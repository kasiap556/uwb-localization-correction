import argparse
import numpy as np
import tensorflow as tf
from data_loader import prepare_datasets
from trainer import train_model

activation_functions = {'sigmoid', 'tanh', 'relu'}
inits = {'uniform', 'xavier', 'he'}
scalers = {'maxabs', 'minmax', 'standard'}

def get_data():
    hidden_layers = []
    hidden = int(input("Podaj ile warstw ukrytych: "))
    for i in range(hidden):
        neutrons = int(input(f"Podaj liczbę neuronów w warstwie {i+1}: "))
        act_fun = input("Funkcje aktywacji: sigmoid, 'tanh, relu ")
        while act_fun not in activation_functions:
            act_fun = input("Podaj poprawna funkcje aktywacji ")
        hidden_layers.append(f"{neutrons}{act_fun[0]}")

    batch = input("Podaj rozmiar batcha ")
    epoch = input("Podaj liczbę epok ")
    learing_rate = float(input("Podaj prędkość uczelnia (najlepiej 0.0001) "))
    initializers = input("Metode inicjalizacji wag: uniform, xavier, he ")
    while initializers not in inits:
        initializers = input("Podaj poprawna metode inicjalizacji ")

    scal_method = input("Metode skalowania: maxabs, minmax, standard ")
    while scal_method not in scalers:
        scal_method = input("Podaj poprawna metode skalowania ")

    seed = input("Podaj ziarno losowości: ")
    seed = int(seed) if seed else None
    patience = input("Podaj wartość cierpliwości dla wczesnego zatrzymania (domyślnie 10): ")
    patience = int(patience) if patience else 10

    return {
        'hidden_layers': hidden_layers,
        'batch_size': int(batch),
        'epochs': int(epoch),
        'learning_rate': learing_rate,
        'init_method': initializers,
        'scaling': scal_method,
        'seed': seed,
        'patience': patience
    }

def main():
    data = get_data()

    if data['seed'] is not None:
        np.random.seed(data['seed'])
        tf.random.set_seed(data['seed'])

    (X_train, y_train), (X_test, y_test) = prepare_datasets("dane")
    result = train_model(data, X_train, y_train, X_test, y_test)

    print(f"Końcowe MSE treningowe: {result['train_mses'][-1]:.4f}")
    print(f"Końcowe MSE testowe: {result['test_mses'][-1]:.4f}")
    print(f"Liczba epok: {result['epochs']}")

if __name__ == '__main__':
    main()