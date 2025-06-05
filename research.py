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

    config_list = [
        {
            'hidden_layers': [(6, activations[2])],
            'input_layer_activation': activations[0],
            'batch_size': 40,
            'epochs': 40,
            'learning_rate': 0.0001,
            'init_method': init_methods[0],
            'scaling': "standard",
            'seed': 42,
            'patience': 10,
            'tol': 0.0001
        },

        {
            'hidden_layers': [(6, activations[2]), (4, activations[2])],
            'input_layer_activation': activations[0],
            'batch_size': 40,
            'epochs': 40,
            'learning_rate': 0.0001,
            'init_method': init_methods[0],
            'scaling': "standard",
            'seed': 42,
            'patience': 10,
            'tol': 0.0001
        }
    ]

    results = {}
    best_config = None
    best_mse = float('inf')

    for config in config_list:
        print(f"Training config with {len(config['hidden_layers'])}")
        res = train_model(config, X_train, y_train, X_test, y_test)

        avg_test_mse = float(np.mean(res['test_mses']))
        std_test_mse = float(np.std(res['test_mses']))

        config_key = (
            f"{config['hidden_layers'][0][1]}_hidden{len(config['hidden_layers'])}_"
            f"lr{config['learning_rate']}_batch{config['batch_size']}_"
            f"init{config['input_layer_activation']}_seed{config['seed']}"
        )

        results[config_key] = {
            'activation': config['hidden_layers'][0][1],
            'hidden_units': len(config['hidden_layers']),
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'init_method': config['init_method'],
            'train_mses': res['train_mses'],
            'test_mses': res['test_mses'],
            'avg_test_mse': avg_test_mse,
            'std_test_mse': std_test_mse,
            'epochs': res['epochs']
        }

        if avg_test_mse < best_mse:
            best_mse = avg_test_mse
            best_config = results[config_key]

    with open(f'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=4)

    with open('all_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("zapisane w plikach json")
    Plots.plot_research_results("all_results.json", "research_results.png")

if __name__ == '__main__':
    research()