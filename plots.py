import matplotlib.pyplot as plt
import numpy as np
import json
import os
import tensorflow as tf
from tf.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


class Plots:

    @staticmethod
    def plot_training_history(train_mses, test_mses, filename="training_history.png"):
        epochs = range(1, len(train_mses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_mses, 'b-', label='Treningowe MSE')
        plt.plot(epochs, test_mses, 'r-', label='Testowe MSE')
        plt.title('Błąd MSE w kolejnych epokach')
        plt.xlabel('Epoki')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Zapisano wykres: {filename}")

    @staticmethod
    def plot_errors_cdf(model, scaler_X, scaler_y, X_test, y_test, filename="errors_cdf.png"):
        X_test_scaled = scaler_X.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        errors = np.linalg.norm(y_pred - y_test, axis=1)
        sorted_errors = np.sort(errors)
        ps = np.arange(len(sorted_errors)) / float(len(sorted_errors))

        raw_errors = np.linalg.norm(X_test - y_test, axis=1)
        sorted_raw_errors = np.sort(raw_errors)
        ps_raw = np.arange(len(sorted_raw_errors)) / float(len(sorted_raw_errors))

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_errors, ps, 'b-', label='Po korekcji')
        plt.plot(sorted_raw_errors, ps_raw, 'r--', label='Bez korekcji')

        plt.title("Dystrybuanta błędów pozycjonowania")
        plt.xlabel("Błąd [mm]")
        plt.ylabel("Prawdopodobieństwo skumulowane")
        plt.xscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Zapisano wykres: {filename}")

    @staticmethod
    def plot_point_chart(model, scaler_X, scaler_y, X_test, y_test, filename="point_chart.png"):
        X_test_scaled = scaler_X.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        n_points = min(500, len(X_test))
        indices = np.random.choice(len(X_test), n_points, replace=False)

        plt.figure(figsize=(10, 8))
        plt.scatter(y_test[indices, 0], y_test[indices, 1], color='blue', s=15, alpha=0.7, label='Rzeczywiste')
        plt.scatter(X_test[indices, 0], X_test[indices, 1], color='green', s=15, alpha=0.7, label='Zmierzone')
        plt.scatter(y_pred[indices, 0], y_pred[indices, 1], color='red', s=15, alpha=0.7, label='Skorygowane')

        plt.title("Porównanie pozycji rzeczywistych, zmierzonych i skorygowanych")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(filename)
        plt.close()
        print(f"Zapisano wykres: {filename}")

    @staticmethod
    def plot_research_results(results_file="all_results.json", filename="research_results.png"):
        with open(results_file, 'r') as f:
            results = json.load(f)

        configs = []
        avg_mses = []
        std_mses = []

        for key, res in results.items():
            configs.append(key)
            avg_mses.append(res['avg_test_mse'])
            std_mses.append(res['std_test_mse'])

        sorted_indices = np.argsort(avg_mses)
        configs = [configs[i] for i in sorted_indices]
        avg_mses = [avg_mses[i] for i in sorted_indices]
        std_mses = [std_mses[i] for i in sorted_indices]

        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(configs))

        plt.barh(y_pos, avg_mses, xerr=std_mses, align='center', alpha=0.7)
        plt.yticks(y_pos, configs, fontsize=8)
        plt.xlabel('Średni błąd MSE')
        plt.title('Porównanie wyników różnych konfiguracji modelu')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Zapisano wykres: {filename}")

    @staticmethod
    def plot_all_for_model(model_path, data_dir='data'):
        model = load_model(model_path)

        config_path = os.path.splitext(model_path)[0] + '_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        scaler_class = StandardScaler if config['scaling'] == 'standard' else \
            MinMaxScaler if config['scaling'] == 'minmax' else MaxAbsScaler
        scaler_X = scaler_class()
        scaler_y = scaler_class()

        from data_loader import prepare_datasets
        (X_train, y_train), (X_test, y_test) = prepare_datasets(data_dir)

        scaler_X.fit(X_train)
        scaler_y.fit(y_train)

        base_name = os.path.splitext(model_path)[0]
        Plots.plot_training_history(
            config['train_mses'],
            config['test_mses'],
            f"{base_name}_history.png"
        )
        Plots.plot_errors_cdf(
            model, scaler_X, scaler_y, X_test, y_test,
            f"{base_name}_cdf.png"
        )
        Plots.plot_point_chart(
            model, scaler_X, scaler_y, X_test, y_test,
            f"{base_name}_points.png"
        )