import datetime as dt
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
import dask.dataframe as dd
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

def main():
    print('ML_FINAL')
    # Загружаем данные
    df = dd.read_csv('data_final.csv').compute()

    print(df.shape)

    # Даунсэмплинг данных
    df_min = df[df['flag'] == 1]
    df_maj = df[df['flag'] == 0]
    
    # Применение даунсэмплинга
    df_maj_downsample = resample(df_maj, replace=False, n_samples=len(df_min), random_state=42)
    
    # Объединение сбалансированных данных 
    df_balanced = pd.concat([df_maj_downsample, df_min], ignore_index=True).sample(frac=1.0, random_state=42)

    # Разделение данных на X и y
    X = df_balanced.drop(['flag'], axis=1)
    y = df_balanced['flag']

    # Определяем список всех моделей
    models = (
        LGBMClassifier(boosting_type='gbdt', verbose=-100, subsample=0.7, 
                       reg_alpha=0.5, reg_lambda=0.0, num_leaves=64,
                       n_estimators=500, min_child_samples=40, max_depth=5, 
                       learning_rate=0.1, colsample_bytree=0.7),
        GradientBoostingClassifier(n_estimators=200, min_samples_leaf=100, max_features=0.5, 
                                   max_depth=5, loss='log_loss', learning_rate=0.1)
    )

    best_score = .0
    best_pipe = None

    # Признаки для удаления
    columns_to_delete = ['id', 'rn']

    for model in models:
        pipe = Pipeline(steps=[
            ('deleting_columns', DeletingColumns(columns_to_delete=columns_to_delete)),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
        print(f'model: {type(model).__name__},ra_mean: {score.mean():.4f}, ra_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    # обучим лучшую модель на всем объеме датасета
    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, ROC_AUC: {best_score:.4f}')

    # запишем в файл final_model.pkl лучшую модель и её метаданные
    joblib.dump({
        'model': best_pipe,
        'metadata': {
            'name': 'best model',
            'author': 'N.Lakotko',
            'version': 1.7,
            'date': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'ROC AUC': best_score
        }
    }, 'final_model.pkl')

if __name__ == '__main__':
    main()