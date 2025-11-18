"""
Machine Learning Price Prediction Module
Uses LSTM neural network with multiple features to predict next day closing price
Features: Close price, Volume, RSI, MACD, Moving Averages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not installed. ML predictions disabled.")
    print("   Install with: pip install tensorflow")


class MLPredictor:
    """Multi-feature LSTM-based stock price predictor"""

    def __init__(self, lookback_days=60, use_multiple_features=True):
        """
        Initialize the predictor

        Args:
            lookback_days (int): Number of past days to use for prediction
            use_multiple_features (bool): Use multiple features or just Close price
        """
        self.lookback_days = lookback_days
        self.use_multiple_features = use_multiple_features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_columns = None

    def _select_features(self, data):
        """
        Select features for the model

        Args:
            data (pd.DataFrame): Stock data with indicators

        Returns:
            pd.DataFrame: Selected features
        """
        if self.use_multiple_features:
            # Use multiple technical indicators as features
            feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MA50', 'MA200']

            # Check which features are available
            available_features = [col for col in feature_cols if col in data.columns]

            if len(available_features) < 2:
                print("WARNING: Not enough features available, using Close price only")
                self.use_multiple_features = False
                return data[['Close']].copy()

            self.feature_columns = available_features
            print(f"Using {len(available_features)} features: {', '.join(available_features)}")
            return data[available_features].copy()
        else:
            self.feature_columns = ['Close']
            return data[['Close']].copy()

    def prepare_data(self, data):
        """
        Prepare data for LSTM model with multiple features

        Args:
            data (pd.DataFrame): Stock data with indicators

        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaled_data)
        """
        # Select and prepare features
        features_df = self._select_features(data)

        # Drop any rows with NaN values
        features_df = features_df.dropna()

        if len(features_df) < self.lookback_days + 50:
            raise ValueError(f"Not enough data. Need at least {self.lookback_days + 50} rows, got {len(features_df)}")

        # Scale the data
        scaled_data = self.scaler.fit_transform(features_df.values)

        # Create sequences
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_days:i])  # All features for lookback period
            y.append(scaled_data[i, 0])  # Predict Close price (first column)

        X, y = np.array(X), np.array(y)

        # Split into train and test (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, y_train, X_test, y_test, scaled_data

    def build_model(self, n_features):
        """
        Build LSTM model architecture for multiple features

        Args:
            n_features (int): Number of input features
        """
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback_days, n_features)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, data, epochs=50, batch_size=32, verbose=0):
        """
        Train the LSTM model

        Args:
            data (pd.DataFrame): Stock data with indicators
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level (0=silent, 1=progress bar)

        Returns:
            dict: Training metrics
        """
        if not TENSORFLOW_AVAILABLE:
            return None

        print("Training multi-feature ML model...")

        try:
            # Prepare data
            X_train, y_train, X_test, y_test, scaled_data = self.prepare_data(data)

            n_features = X_train.shape[2]

            # Build and train model
            self.model = self.build_model(n_features)
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=verbose
            )

            # Make predictions
            train_predictions = self.model.predict(X_train, verbose=0)
            test_predictions = self.model.predict(X_test, verbose=0)

            # Inverse transform predictions (only Close price)
            # Create dummy array for inverse transform
            train_pred_full = np.zeros((len(train_predictions), n_features))
            train_pred_full[:, 0] = train_predictions.flatten()
            train_predictions_inv = self.scaler.inverse_transform(train_pred_full)[:, 0]

            test_pred_full = np.zeros((len(test_predictions), n_features))
            test_pred_full[:, 0] = test_predictions.flatten()
            test_predictions_inv = self.scaler.inverse_transform(test_pred_full)[:, 0]

            # Get actual values
            y_train_full = np.zeros((len(y_train), n_features))
            y_train_full[:, 0] = y_train
            y_train_actual = self.scaler.inverse_transform(y_train_full)[:, 0]

            y_test_full = np.zeros((len(y_test), n_features))
            y_test_full[:, 0] = y_test
            y_test_actual = self.scaler.inverse_transform(y_test_full)[:, 0]

            # Calculate metrics
            train_mae = mean_absolute_error(y_train_actual, train_predictions_inv)
            test_mae = mean_absolute_error(y_test_actual, test_predictions_inv)
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions_inv))
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions_inv))

            metrics = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_predictions': train_predictions_inv,
                'test_predictions': test_predictions_inv,
                'n_features': n_features,
                'features_used': self.feature_columns
            }

            print(f"Model trained - Test MAE: ${test_mae:.2f}, Test RMSE: ${test_rmse:.2f}")

            return metrics

        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def predict_next_day(self, data):
        """
        Predict the next day's closing price

        Args:
            data (pd.DataFrame): Stock data with indicators

        Returns:
            float: Predicted next day closing price
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None

        try:
            # Select features
            features_df = self._select_features(data)
            features_df = features_df.dropna()

            # Get last lookback_days of data
            last_data = features_df.values[-self.lookback_days:]
            last_data_scaled = self.scaler.transform(last_data)

            # Reshape for prediction
            X_pred = np.reshape(last_data_scaled, (1, self.lookback_days, len(self.feature_columns)))

            # Make prediction
            prediction_scaled = self.model.predict(X_pred, verbose=0)

            # Inverse transform
            pred_full = np.zeros((1, len(self.feature_columns)))
            pred_full[0, 0] = prediction_scaled[0, 0]
            prediction = self.scaler.inverse_transform(pred_full)[0, 0]

            return prediction

        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def get_predictions_for_plotting(self, data):
        """
        Get all predictions aligned with dates for plotting

        Args:
            data (pd.DataFrame): Stock data with indicators

        Returns:
            pd.DataFrame: DataFrame with dates and predictions
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None

        try:
            # Prepare data
            features_df = self._select_features(data)
            features_df = features_df.dropna()
            scaled_data = self.scaler.transform(features_df.values)

            # Create input sequences for all data points
            X_all = []
            for i in range(self.lookback_days, len(scaled_data)):
                X_all.append(scaled_data[i-self.lookback_days:i])

            X_all = np.array(X_all)

            # Predict
            predictions_scaled = self.model.predict(X_all, verbose=0)

            # Inverse transform
            n_features = len(self.feature_columns)
            pred_full = np.zeros((len(predictions_scaled), n_features))
            pred_full[:, 0] = predictions_scaled.flatten()
            predictions = self.scaler.inverse_transform(pred_full)[:, 0]

            # Align with dates
            dates = features_df.index[self.lookback_days:]
            pred_df = pd.DataFrame({
                'Date': dates,
                'ML_Prediction': predictions
            })
            pred_df.set_index('Date', inplace=True)

            return pred_df

        except Exception as e:
            print(f"Error getting predictions for plotting: {e}")
            return None


def predict_stock_price(data, lookback_days=60, epochs=50, verbose=0, use_multiple_features=True):
    """
    Convenience function to train model and make predictions

    Args:
        data (pd.DataFrame): Stock data with indicators
        lookback_days (int): Number of past days to use
        epochs (int): Training epochs
        verbose (int): Verbosity level
        use_multiple_features (bool): Use multiple features or just Close price

    Returns:
        tuple: (predictor, metrics, next_day_prediction, predictions_df)
    """
    if not TENSORFLOW_AVAILABLE:
        return None, None, None, None

    predictor = MLPredictor(lookback_days=lookback_days, use_multiple_features=use_multiple_features)
    metrics = predictor.train(data, epochs=epochs, verbose=verbose)
    
    if metrics is None:
        return None, None, None, None
    
    next_day_pred = predictor.predict_next_day(data)
    predictions_df = predictor.get_predictions_for_plotting(data)
    
    return predictor, metrics, next_day_pred, predictions_df
