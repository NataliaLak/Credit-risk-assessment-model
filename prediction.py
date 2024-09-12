import joblib
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')

# Функция удаления колонок
class DeletingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_delete):
        self.columns_to_delete = columns_to_delete

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_delete, errors='ignore')

def reduce_memory_usage(df):
    """ 
    Функция для уменьшения использования памяти путем приведения типов данных к более узким типам.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Начальный объем памяти: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Конечный объем памяти: {end_mem:.2f} MB')
    print(f'Экономия памяти: {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df

def main():
    print('Предсказания финальной модели:')

    # Загружаем данные
    df = dd.read_csv('data_final.csv').compute()

    # Подготовка данных
    X = df.drop(['flag'], axis=1)
    y = df['flag']

    # Уменьшаем использование памяти
    X = reduce_memory_usage(X)

    # Признаки для удаления
    columns_to_delete = ['id', 'rn']
    X = DeletingColumns(columns_to_delete=columns_to_delete).transform(X)

    # Загружаем модель
    model_data = joblib.load('final_model.pkl')
    model = model_data['model']

    # Предсказания 
    y_pred = model.predict(X)  

    # Выводим предсказания
    results_df = pd.DataFrame({
        'actual': y,
        'predicted': y_pred
    })

    print(results_df.head())

if __name__ == '__main__':
    main()