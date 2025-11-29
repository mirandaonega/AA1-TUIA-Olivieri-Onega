
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def f1_score_nn(y_true, y_pred):
    """
    Custom F1 score para red neuronal
    """
    y_pred = tf.round(y_pred)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred, 'float32'))
    possible_positives = tf.reduce_sum(tf.cast(y_true, 'float32'))

    precision = tp / (predicted_positives + tf.keras.backend.epsilon())
    recall = tp / (possible_positives + tf.keras.backend.epsilon())

    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

keras.utils.get_custom_objects()['f1_score_nn'] = f1_score_nn

class RegionManagementTransformer(BaseEstimator, TransformerMixin):
    """
    Mergea información de la región desde el dictionario creado a partir de la clusterización
    """

    def __init__(self, region_mapping):
        self.region_mapping = region_mapping.copy()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X = X.merge(self.region_mapping, on='Location', how='left')

        if 'Location' in X.columns:
            X = X.drop(columns=['Location'])

        return X


class DateManagementTransformer(BaseEstimator, TransformerMixin):
    """
    Convierte la columna Date a datetime y crea la columna Season
    """

    def __init__(self, date_column='Date'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.date_column in X.columns:
            X[self.date_column] = pd.to_datetime(X[self.date_column])

            X['Month'] = X[self.date_column].dt.month

            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Verano'
                elif month in [3, 4, 5]:
                    return 'Otoño'
                elif month in [6, 7, 8]:
                    return 'Invierno'
                else:
                    return 'Primavera'

            X['Season'] = X['Month'].apply(get_season)

            X = X.drop(columns=['Month', self.date_column])

        return X


class NullManagementTransformer(BaseEstimator, TransformerMixin):
    """
    Remueve filas donde tengo columnas nulas mayores al threshold (seteado en 50%)
    """

    def __init__(self, null_threshold=0.5):
        self.null_threshold = null_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        threshold = X.shape[1] - int(X.shape[1] * self.null_threshold)

        X = X.dropna(thresh=threshold)
        return X


class GroupedImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Imputa valores faltantes usando estadísticos calculados previamente
    """

    def __init__(self, mean_stats=None, median_stats=None, knn_imputers=None, mode_stats=None):
        self.mean_stats = mean_stats or {}
        self.median_stats = median_stats or {}
        self.knn_imputers = knn_imputers or {}
        self.mode_stats = mode_stats or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['_group_key'] = list(zip(X['Region'], X['Season']))

        for col, stats_dict in self.mean_stats.items():
            if col in X.columns:
                missing_mask = X[col].isna()
                if missing_mask.any():
                    X.loc[missing_mask, col] = X.loc[missing_mask, '_group_key'].map(stats_dict)

        for col, stats_dict in self.median_stats.items():
            if col in X.columns:
                missing_mask = X[col].isna()
                if missing_mask.any():
                    X.loc[missing_mask, col] = X.loc[missing_mask, '_group_key'].map(stats_dict)

        for col, imputers_dict in self.knn_imputers.items():
            if col in X.columns:
                for (region, season), imputer in imputers_dict.items():
                    group_mask = (X['Region'] == region) & (X['Season'] == season) & X[col].isna()

                    if group_mask.any():
                        group_data = X.loc[group_mask, [col]]
                        imputed_values = imputer.transform(group_data)
                        X.loc[group_mask, col] = imputed_values.ravel()

        for col, stats_dict in self.mode_stats.items():
            if col in X.columns:
                missing_mask = X[col].isna()
                if missing_mask.any():
                    X.loc[missing_mask, col] = X.loc[missing_mask, '_group_key'].map(stats_dict)

        X = X.drop(columns=['_group_key'])

        return X


class EncodingTransformer(BaseEstimator, TransformerMixin):
    """
        Transformo variables categóricas según el caso
    """

    def __init__(self, wind_encoding=None, expected_columns=None):
        self.wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        self.dummy_cols = ['Region', 'Season', 'RainToday']

        if wind_encoding is None:
            directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            self.wind_encoding = {d: i * 22.5 for i, d in enumerate(directions)}
        else:
            self.wind_encoding = wind_encoding

        self.expected_columns = expected_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply encoding."""
        X = X.copy()
        
        if 'RainTomorrow' in X.columns:
            X = X.drop(columns=['RainTomorrow'])

        for col in self.wind_cols:
            if col in X.columns:
                angles = X[col].map(self.wind_encoding)
                angles_rad = np.radians(angles)

                X[f'{col}_sin'] = np.sin(angles_rad)
                X[f'{col}_cos'] = np.cos(angles_rad)

                X = X.drop(columns=[col])

        X = pd.get_dummies(X, columns=self.dummy_cols, drop_first=True)

        """
            Agrega las columnas que el modelo espera 
            (esto es para que sea coherente con la función de dummies)
        """
        if self.expected_columns is not None:
            for col in self.expected_columns:
                if col not in X.columns:
                    X[col] = 0

            X = X[self.expected_columns]

        return X


class NumericalScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Scales numerical columns using a pre-fitted StandardScaler.
    """

    def __init__(self, scaler, cols_to_scale):
        self.scaler = scaler
        self.cols_to_scale = cols_to_scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.cols_to_scale:
            existing_cols = [col for col in self.cols_to_scale if col in X.columns]
            if existing_cols:
                X[existing_cols] = self.scaler.transform(X[existing_cols])

        return X
    
class KerasClassifierWrapper(BaseEstimator):
    """
    Wrapper para modelos de Keras que los hace compatibles con sklearn Pipeline.
    Agrega los métodos predict_proba y predict que sklearn espera.
    """
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        """
        Devuelve predicciones de clase (0 o 1).
        """
        y_pred_prob = self.model.predict(X, verbose=0)
        return (y_pred_prob > 0.5).astype(int).ravel()
    
    def predict_proba(self, X):
        """
        Devuelve probabilidades para ambas clases.
        Sklearn espera una matriz de forma (n_samples, n_classes).
        """
        y_pred_prob = self.model.predict(X, verbose=0).ravel()
        # Retornar [prob_clase_0, prob_clase_1] para cada muestra
        return np.column_stack([1 - y_pred_prob, y_pred_prob])