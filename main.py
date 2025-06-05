import argparse
from threading import activeCount

import numpy as np
import tensorflow as tf
from data_loader import prepare_datasets
from trainer import train_model
from plots import Plots

activation_functions = {'sigmoid', 'tanh', 'relu'}
inits = {'uniform', 'xavier', 'he'}
#scalers = {'maxabs', 'minmax', 'standard'}

def get_data():
    hidden_layers = []
    hidden = input("Podaj ile warstw ukrytych (domyślnie 2): ")
    hidden = int(hidden) if hidden else 2

    input_layer_activation = input("Podaj metode aktywacji (sigmoid, tanh, relu) warstwy wejsciowej [domyślnie relu]: ")
    input_layer_activation = input_layer_activation if input_layer_activation in activation_functions else "relu"

    for i in range(hidden):
        neutrons = input(f"Podaj liczbę neuronów w warstwie {i+1} (domyślnie 10): ")
        neutrons = int(neutrons) if neutrons else 10

        activation = input("Funkcja aktywacji (sigmoid, tanh, relu) [domyślnie relu]: ")
        activation = activation if activation in activation_functions else "relu"
        while activation not in activation_functions:
            activation = input("Podaj poprawną funkcję aktywacji (sigmoid, tanh, relu): ")

        hidden_layers.append((neutrons, activation))

    batch = input("Podaj rozmiar batcha (domyślnie 40): ")
    batch = int(batch) if batch else 40

    epoch = input("Podaj liczbę epok (domyślnie 40): ")
    epoch = int(epoch) if epoch else 10

    learning_rate = input("Podaj prędkość uczenia (np. 0.0001) [domyślnie 0.0001]: ")
    learning_rate = float(learning_rate) if learning_rate else 0.0001

    initializers = input("Metoda inicjalizacji wag (uniform, xavier, he) [domyślnie he]: ")
    initializers = initializers if initializers in inits else "he"
    while initializers not in inits:
        initializers = input("Podaj poprawną metodę inicjalizacji (uniform, xavier, he): ")

    # scal_method = input("Metoda skalowania (maxabs, minmax, standard) [domyślnie standard]: ")
    # scal_method = scal_method if scal_method in scalers else "standard"
    # while scal_method not in scalers:
    #     scal_method = input("Podaj poprawną metodę skalowania (maxabs, minmax, standard): ")
    scal_method = "standard"

    seed = input("Podaj ziarno losowości (domyślnie brak): ")
    seed = int(seed) if seed else None

    patience = input("Podaj wartość cierpliwości dla wczesnego zatrzymania (domyślnie 10): ")
    patience = int(patience) if patience else 10

    tol = input("Podaj minimalną zmianę metryki (tol/min_delta) do uznania poprawy [domyślnie 0.0001]: ")
    tol = float(tol) if tol else 0.0001

    return {
        'hidden_layers': hidden_layers,
        'input_layer_activation': input_layer_activation,
        'batch_size': batch,
        'epochs': epoch,
        'learning_rate': learning_rate,
        'init_method': initializers,
        'scaling': scal_method,
        'seed': seed,
        'patience': patience,
        'tol': tol
    }


def main():
    data = get_data()

    if data['seed'] is not None:
        np.random.seed(data['seed'])
        tf.random.set_seed(data['seed'])

    (X_train, y_train), (X_test, y_test) = prepare_datasets("dane")
    result = train_model(data, X_train, y_train, X_test, y_test)

    Plots.plot_training_history(result['train_mses'], result['test_mses'])
    Plots.plot_errors_cdf(result['model'], result['scaler_X'], result['scaler_Y'], X_test, y_test)
    Plots.plot_point_chart(result['model'], result['scaler_X'], result['scaler_Y'], X_test, y_test)
    # Plots.plot_research_results()
    # Plots.plot_all_for_model()

    print(f"Końcowe MSE treningowe: {result['train_mses'][-1]:.4f}")
    print(f"Końcowe MSE testowe: {result['test_mses'][-1]:.4f}")
    print(f"Liczba epok: {result['epochs']}")

if __name__ == '__main__':
    main()