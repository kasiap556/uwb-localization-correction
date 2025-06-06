import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Plots:

    @staticmethod
    def plots_from_file(all_filepath, best_filepath):

        with open(best_filepath) as f:
            best_results = json.load(f)

        with open(all_filepath) as f:
            all_results = json.load(f)

        epochs = list(range(1, best_results['epochs'] + 1))

        # wykres 1 mse train
        plt.figure(figsize=(12, 6))

        for label, results in all_results.items():
            plt.plot(epochs, results['train_mses'],
                     label=f"Model z iloscia neuronów ukrytych =  {results['hidden_units']} i funkcja {results['activation']}", linewidth=2)

        plt.title('Błąd MSE na zbiorze uczącym')
        plt.xlabel('Epoka')
        plt.ylabel('Błąd MSE')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.xlim(1, best_results['epochs'])
        plt.ylim(10000, 80000)
        plt.tight_layout()

        plt.savefig(all_filepath + '1.png')

        # wykres 2 mse test
        plt.figure(figsize=(12, 6))

        for label, results in all_results.items():
            plt.plot(epochs, results['train_mses'],
                     label=f"Model z iloscia neuronów ukrytych =  {results['hidden_units']} i funkcja {results['activation']}", linewidth=2)

        plt.axhline(y=results['reference_mse'], color='black', linestyle='--',
                    label='Wartość błędu MSE dla danych testowych')
        plt.title('Błąd MSE na zbiorze testowym')
        plt.xlabel('Epoka')
        plt.ylabel('Błąd MSE')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.xlim(1, best_results['epochs'])
        plt.ylim(10000, 80000)
        plt.tight_layout()

        plt.savefig(all_filepath + '2.png')



        # wykres 3 - dystrybuanta
        plt.figure(figsize=(16, 12))

        test_output = None
        test_input = None
        for label, results in all_results.items():
            predictions = np.array(results['predictions']) # wartosci skorygowane przez model
            test_output = np.array(results['test_output']) #wartosci rzeczywiste
            test_input = np.array(results['test_input']) # wartosci zmierzone

            errors = np.linalg.norm(predictions - test_output, axis=1)
            sorted_errors = np.sort(errors)
            ps = np.arange(len(sorted_errors)) / float(len(sorted_errors))
            plt.plot(sorted_errors, ps, label=f"Model z iloscia neuronów ukrytych =  {results['hidden_units']} i funkcja {results['activation']}")

        all_errors = np.linalg.norm(test_output - test_input, axis=1)
        all_sorted_errors = np.sort(all_errors)
        ps_all_errors = np.arange(len(all_sorted_errors)) / float(len(all_sorted_errors))
        plt.plot(all_sorted_errors, ps_all_errors, label='Dystrybuanta błędów wszystkich danych testowych', color='black', linestyle='--')

        plt.title("Dystrybuanta błędów dla pomiarów dynamicznych", fontsize=20)
        plt.xlabel("Błąd [mm]", fontsize=18)
        plt.ylabel("Prawdopodobieństwo", fontsize=18)
        plt.xscale('log')

        xmin = np.min(np.concatenate([sorted_errors, all_sorted_errors]))
        xmax = np.max(np.concatenate([sorted_errors, all_sorted_errors])) + 10
        plt.xlim(xmin, xmax)

        plt.legend()
        plt.grid(True)
        plt.savefig(all_filepath + "3.png")


        # wykres 4 (punkty)
        plt.figure(figsize=(14, 8))

        true_values = np.array(best_results['test_output'])
        corrected = np.array(best_results['predictions'])
        measured = np.array(best_results['test_input'])

        plt.scatter(true_values[:, 0], true_values[:, 1], color='blue', label='Wartości rzeczywiste', zorder=4)
        plt.scatter(corrected[:, 0], corrected[:, 1], color='red', label='Wartości skorygowane', zorder=3)
        plt.scatter(measured[:, 0], measured[:, 1], color='green', label='Wartości zmierzone', zorder=2)

        plt.title(f'Porównanie wartości pomiarów (najlepszy model: {best_results["hidden_units"]} neuronów ukrytych / funkcja aktywacji = {best_results["activation"]})', fontsize=16)
        plt.xlabel('x [mm]', fontsize=14)
        plt.ylabel('y [mm]', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(best_filepath + '4.png')