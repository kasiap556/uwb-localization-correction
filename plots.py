import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class Plots:

    @staticmethod
    def plots_from_file(filepath):

        # Wczytanie danych
        with open(filepath) as f:
            results = json.load(f)

        epochs = list(range(1, 11))

        # Wykres 1: MSE na zbiorze uczącym
        plt.figure(figsize=(12, 6))
        # for label, data in models.items():
        #     plt.semilogy(epochs, data['train_mses'], label=label, linewidth=2)
        plt.plot(epochs, results['train_mses'], label="Train MSE", linewidth=2)

        plt.title('Błąd MSE na zbiorze uczącym w kolejnych epokach')
        plt.xlabel('Epoka')
        plt.ylabel('MSE')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.xlim(1, 11)
        plt.ylim(10000, 80000)  # Przycięcie górnej części osi
        plt.tight_layout()
        #plt.show()
        plt.savefig(filepath + '1.png')

        # Wykres 2: MSE na zbiorze testowym
        plt.figure(figsize=(12, 6))
        # for label, data in models.items():
        #     plt.semilogy(epochs, data['test_mses'], label=label, linewidth=2)
        plt.plot(epochs, results['test_mses'], label="Train MSE", linewidth=2)
        plt.axhline(y=results['reference_mse'], color='black', linestyle='--', label='Wartość błędu MSE dla danych testowych')
        plt.title('Błąd MSE na zbiorze testowym w kolejnych epokach')
        plt.xlabel('Epoka')
        plt.ylabel('MSE (skala logarytmiczna)')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.xlim(1, 11)
        plt.ylim(10000, 80000)  # Przycięcie górnej części osi
        plt.tight_layout()
        #plt.show()
        plt.savefig(filepath + '2.png')

        # Generowanie danych zastępczych do wykresów 3 i 4 (symulacja)
        # np.random.seed(42)
        # n_points = 1000
        # measured = np.random.normal(50, 15, epochs)  # Zmierzone wartości
        # true_values = measured + np.random.normal(0, 5, epochs)  # Rzeczywiste wartości
        # corrected = true_values + np.random.normal(0, 2, epochs)  # Skorygowane (najlepszy model)
        #
        # # Błędy dla dystrybuanty
        # errors_measured = np.abs(measured - true_values)
        # errors_corrected = np.abs(corrected - true_values)

        # Wykres 3: Dystrybuanty błędów
        # plt.figure(figsize=(12, 6))
        # plt.plot(sorted(errors_measured), np.linspace(0, 1, len(errors_measured)),
        #                                               label='Błędy zmierzone', linewidth=2)
        # plt.plot(sorted(errors_corrected), np.linspace(0, 1, len(errors_corrected)),
        #                                                label='Błędy skorygowane (2 neurony)', linewidth=2)
        #
        # plt.title('Dystrybuanty błędów dla wyników pomiarów')
        # plt.xlabel('Wartość błędu')
        # plt.ylabel('Dystrybuanta')
        # plt.grid(True, ls="--")
        # plt.legend()
        # plt.xlim(0, 20)  # Dopasowanie do obszaru największego wzrostu
        # plt.tight_layout()
        # #plt.show()
        # plt.savefig(filepath + '3.png')

        plt.figure(figsize=(16, 12))

        predictions = np.array(results['predictions'])
        test_output = np.array(results['test_output'])
        test_input = np.array(results['test_input'])

        errors = np.linalg.norm(predictions - test_output, axis=1)
        sorted_errors = np.sort(errors)
        ps = np.arange(len(sorted_errors)) / float(len(sorted_errors))
        plt.plot(sorted_errors, ps, label=f"Model z iloscia warstw ukrytych =  {results['hidden_units']}")

        all_errors = np.linalg.norm(test_output - test_input, axis=1)
        all_sorted_errors = np.sort(all_errors)
        ps_all_errors = np.arange(len(all_sorted_errors)) / float(len(all_sorted_errors))
        plt.plot(all_sorted_errors, ps_all_errors, label='Wszystkie dane testowe', color='black', linestyle='--')

        plt.title("Dystrybuanta błędów dla pomiarów dynamicznych")
        plt.xlabel("Błąd [mm]")
        plt.ylabel("Prawdopodobieństwo skumulowane")
        plt.xscale('log')

        xmin = np.min(np.concatenate([sorted_errors, all_sorted_errors]))
        xmax = np.max(np.concatenate([sorted_errors, all_sorted_errors])) + 10
        plt.xlim(xmin, xmax)

        plt.legend()
        plt.grid(True)
        plt.savefig(filepath + "3.png")
        plt.show()


        #
        # # Wykres 4: Wartości pomiarów (najlepszy model)
        # plt.figure(figsize=(14, 8))
        # # Warstwy: rzeczywiste -> skorygowane -> zmierzone
        # # plt.scatter(range(n_points), true_values, alpha=0.6, label='Wartości rzeczywiste', s=20)
        # # plt.scatter(range(n_points), corrected, alpha=0.5, label='Wartości skorygowane', s=15)
        # # plt.scatter(range(n_points), measured, alpha=0.4, label='Wartości zmierzone', s=10)
        #
        # plt.scatter(true_values[:,0], true_values[:, 1], color='blue', label='Wartości rzeczywiste', zorder=4)
        # plt.scatter(corrected[:,0], corrected[:, 1], color='red', label='Wartości skorygowane', zorder=3)
        # plt.scatter(measured[:,0], measured[:, 1], color='green', label='Wartości zmierzone', zorder=2)
        #
        # plt.title('Porównanie wartości pomiarów (najlepszy model: 2 neurony)')
        # plt.xlabel('Numer próbki')
        # plt.ylabel('Wartość (skala oryginalna)')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # #plt.show()
        # plt.savefig(filepath + '4.png')