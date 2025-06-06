import json
import numpy as np
import tensorflow as tf
from scipy.linalg import hilbert

from data_loader import prepare_datasets
from trainer import train_model
from plots import Plots


def research():
    (X_train, y_train), (X_test, y_test) = prepare_datasets('dane')

    activations = ['sigmoid', 'tanh', 'relu']
    init_methods = ['uniform', 'xavier', 'he']

    config_list = []

    for i in range(3):
        hidden_neuron_count = input("Podaj ile neuronów w warstwie ukrytej (domyślnie 6): ")
        hidden_neuron_count = int(hidden_neuron_count) if hidden_neuron_count else 6

        learning_rate = input("Podaj prędkość uczenia (np. 0.0001) [domyślnie 0.0001]: ")
        learning_rate = float(learning_rate) if learning_rate else 0.0001

        batch_size = input("Podaj rozmiar batcha (domyślnie 40): ")
        batch_size = int(batch_size) if batch_size else 40

        epochs = input("Podaj liczbe epok (domyślnie 10): ")
        epochs = int(epochs) if epochs else 10

        hidden_layer_activation = input(
            "Podaj metode aktywacji (sigmoid, tanh, relu) warstwy ukrytej [domyślnie relu]: ")
        hidden_layer_activation = hidden_layer_activation if hidden_layer_activation in activations else "relu"

        init_method = input("Podaj metode inicjalizacji wag (uniform, xavier, he) [domyślnie uniform]: ")
        init_method = init_method if init_method in init_methods else "uniform"

        config = {
            'hidden_neurons': hidden_neuron_count,
            'hidden_layer_activation': hidden_layer_activation,
            'batch_size': batch_size, #podzbiór danych, model przetwarza jednorazowo, zanim zaktualizuje swoje wagi
            'epochs': epochs,
            'learning_rate': learning_rate,
            'init_method': init_method,
            'scaling': "standard",
            'patience': 10, #gdy sie nie poprawi model (po tylu) to zatrzymyje trening
            'tol': 0.0001 #jesli poprawa mniejsza niz tol to nie ma postepu
        }

        config_list.append(config)

    results = {}
    best_config = None
    best_mse = float('inf')

    for config in config_list:
        print(f"Training config with {config['hidden_neurons']} hidden neurons...")
        res = train_model(config, X_train, y_train, X_test, y_test)

        avg_test_mse = float(np.mean(res['test_mses']))
        std_test_mse = float(np.std(res['test_mses']))

        config_key = (
            f"{config['hidden_neurons']}_{config['hidden_layer_activation']}_"
            f"lr{config['learning_rate']}_batch{config['batch_size']}"
        )

        X_test_scaled = res['scaler_X'].transform(X_test)
        predictions_scaled = res['model'].predict(X_test_scaled)
        predictions = res['scaler_Y'].inverse_transform(predictions_scaled)

        results[config_key] = {
            'activation': config['hidden_layer_activation'],
            'hidden_units': config['hidden_neurons'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'init_method': config['init_method'],
            'train_mses': res['train_mses'],
            'test_mses': res['test_mses'],
            'reference_mse': float(tf.keras.losses.MeanSquaredError()(X_test, y_test)), #blad mse z wartosci z pliku
            'avg_test_mse': avg_test_mse,
            'std_test_mse': std_test_mse,
            'epochs': res['epochs'],
            'predictions': predictions.tolist(),
            'test_output': y_test.tolist(),
            'test_input': X_test.tolist(),
        }

        if avg_test_mse < best_mse:
            best_mse = avg_test_mse
            best_config = results[config_key]

    with open(f'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=4)

    with open('all_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("zapisaned")

if __name__ == '__main__':
    research()
    Plots.plots_from_file("all_results.json", "best_config.json")