import argparse
import numpy as np
import tensorflow as tf
from data_loader import prepare_datasets
from trainer import train_model


def main():
    parser = argparse.ArgumentParser(description='UWB Localization Correction MLP with Keras')
    parser.add_argument('--hidden', type=int, default=10, help='Liczba neuronów w warstwie ukrytej')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'], help='Funkcja aktywacji')
    parser.add_argument('--lr', type=float, default=0.001, help='Szybkość uczenia')
    parser.add_argument('--batch_size', type=int, default=32, help='Rozmiar batcha')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maksymalna liczba epok')
    parser.add_argument('--scaling', type=str, default='standard',
                        choices=['maxabs', 'minmax', 'standard'], help='Metoda skalowania')
    parser.add_argument('--init', type=str, default='uniform',
                        choices=['uniform', 'xavier', 'he'], help='Metoda inicjalizacji wag')
    parser.add_argument('--seed', type=int, default=None, help='Ziarno losowości')
    parser.add_argument('--data_dir', type=str, default='data', help='Katalog z danymi')
    parser.add_argument('--patience', type=int, default=10, help='Cierpliwość dla wczesnego zatrzymywania')
    parser.add_argument('--tol', type=float, default=1e-4, help='Minimalna poprawa dla wczesnego zatrzymywania')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    (X_train, y_train), (X_test, y_test) = prepare_datasets(args.data_dir)

    config = {
        'hidden_units': args.hidden,
        'activation': args.activation,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'scaling': args.scaling,
        'init_method': args.init,
        'patience': args.patience,
        'tol': args.tol
    }

    result = train_model(config, X_train, y_train, X_test, y_test)

    print(f"Końcowe MSE treningowe: {result['train_mses'][-1]:.4f}")
    print(f"Końcowe MSE testowe: {result['test_mses'][-1]:.4f}")
    print(f"Liczba epok: {result['epochs']}")

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


if __name__ == '__main__':
    main()