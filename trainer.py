import numpy as np
from keras import Input
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.initializers import GlorotUniform, HeUniform, RandomUniform

class MetricsLogger(Callback):
    def __init__(self, X_train_scaled, y_train, X_test_scaled, y_test, scaler_y):
        super().__init__()
        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        self.scaler_y = scaler_y
        self.train_mses = []
        self.test_mses = []

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train_scaled, verbose=0)
        y_train_pred_orig = self.scaler_y.inverse_transform(y_train_pred)
        train_mse = np.mean((self.y_train - y_train_pred_orig) ** 2)
        self.train_mses.append(train_mse)

        y_test_pred = self.model.predict(self.X_test_scaled, verbose=0)
        y_test_pred_orig = self.scaler_y.inverse_transform(y_test_pred)
        test_mse = np.mean((self.y_test - y_test_pred_orig) ** 2)
        self.test_mses.append(test_mse)

        print(f"Epoch {epoch+1}, Train MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}")

        logs['val_mse_orig'] = test_mse


def create_scaler(method):
    if method == 'maxabs':
        return MaxAbsScaler()
    elif method == 'minmax':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def create_model(input_dim, hidden_layers, input_layer_activation, init_method):
    if init_method == 'uniform':
        initializer = RandomUniform(minval=-0.1, maxval=0.1)
    elif init_method == 'xavier':
        initializer = GlorotUniform()
    elif init_method == 'he':
        initializer = HeUniform()
    else:
        initializer = 'glorot_uniform'

    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(hidden_layers[0][0], activation=input_layer_activation, kernel_initializer=initializer)) #input_shape=(2,)
    if len(hidden_layers) > 1 :
        for layer in hidden_layers[1:]:
            model.add(Dense(layer[0], activation=layer[1], kernel_initializer=initializer))

    model.add(Dense(2, activation='linear', kernel_initializer=initializer))

    return model


def train_model(config, X_train, y_train, X_test, y_test):
    scaler_X = create_scaler(config['scaling'])
    scaler_X.fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = create_scaler(config['scaling'])
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)

    model = create_model(
        input_dim=2,
        hidden_layers=config['hidden_layers'],
        input_layer_activation=config['input_layer_activation'],
        init_method=config['init_method']
    )

    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='mse'
    )

    metrics_logger = MetricsLogger(X_train_scaled, y_train, X_test_scaled, y_test, scaler_y)

    early_stopping = EarlyStopping(
        monitor='val_mse_orig',
        patience=config['patience'],
        min_delta=config['tol'],
        restore_best_weights=True,
        mode='min'
    )

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=0,
        callbacks=[metrics_logger]
    )

    return {
        'model': model,
        'train_mses': metrics_logger.train_mses,
        'test_mses': metrics_logger.test_mses,
        'scaler_X': scaler_X,
        'scaler_Y': scaler_y,
        'epochs': len(metrics_logger.train_mses)
    }