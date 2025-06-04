import json
import numpy as np
import tensorflow as tf
from data_loader import prepare_datasets
from trainer import train_model
from plots import Plots

def research():
    (X_train, y_train), (X_test, y_test) = prepare_datasets('data')

    #wartosci dobre?
    activations = ['sigmoid', 'tanh', 'relu']
    hidden_units_list = [1, 2, 3]
    learning_rates = [0.0001]
    batch_sizes = [40]
    init_methods = ['uniform', 'xavier', 'he']
    trials = 3

    results = {}

    for activation in activations:
        activation_results = []

        for hidden_units in hidden_units_list:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    for init_method in init_methods:
                        trial_results = []

                        for trial in range(trials):
                            print(f"Training: {activation}, hidden={hidden_units}, lr={lr}, "
                                  f"batch={batch_size}, init={init_method}, trial={trial + 1}/{trials}")

                            config = {
                                'hidden_units': hidden_units,
                                'activation': activation,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'max_epochs': 20,
                                'scaling': 'standard',
                                'init_method': init_method,
                                'patience': 20,
                                'tol': 1e-5
                            }

                            seed = 42 + trial
                            np.random.seed(seed)
                            tf.random.set_seed(seed)

                            result = train_model(config, X_train, y_train, X_test, y_test)
                            final_test_mse = result['test_mses'][-1]
                            trial_results.append(final_test_mse)

                        avg_test_mse = np.mean(trial_results)
                        std_test_mse = np.std(trial_results)

                        config_key = (f"{activation}_hidden{hidden_units}_lr{lr}_"
                                      f"batch{batch_size}_init{init_method}")

                        results[config_key] = {
                            'activation': activation,
                            'hidden_units': hidden_units,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'init_method': init_method,
                            'trials': trial_results,
                            'avg_test_mse': avg_test_mse,
                            'std_test_mse': std_test_mse
                        }

        best_config = None
        best_mse = float('inf')

        for key, res in results.items():
            if res['activation'] == activation and res['avg_test_mse'] < best_mse:
                best_mse = res['avg_test_mse']
                best_config = res

        with open(f'best_{activation}.json', 'w') as f:
            json.dump(best_config, f, indent=4)

    with open('all_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("end, zapisane w plikach json")
    Plots.plot_research_results("all_results.json", "research_results.png")

if __name__ == '__main__':
    research()