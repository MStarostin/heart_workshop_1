import re

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TRAIN_DATA_CSV = 'datasets/heart_train.csv'
MODEL_NAME_PKL = 'models/model_heart.pkl'
TARGET_ATTR = 'heart_attack_risk_binary'


class ClassifierModel():
    def __init__(self, random_state):
        self.random_state = random_state
        self.model = None

    def split_data(self, df, target_attr):
        X_train = df.drop([target_attr], axis=1)
        y_train = df[target_attr]
        return X_train, y_train,

    def create_pipeline(
            self,
            X_train,
            y_train,
            ohe_columns=None,
            ord_columns=None,
            categories=[]
            ):
        transformers = []
        num_columns = X_train.select_dtypes(include='number').columns.tolist()
        transformers.append(('num', StandardScaler(), num_columns))
        if ohe_columns:
            ohe_pipe = Pipeline(
                [('simple_imputer_ohe', SimpleImputer(
                    missing_values=np.nan, strategy='most_frequent'
                    )),
                 ('ohe', OneHotEncoder(drop='first',
                                       handle_unknown='ignore',
                                       sparse_output=False))]
                )
            transformers.append(('ohe', ohe_pipe, ohe_columns))
        if ord_columns:
            ord_pipe = Pipeline(
                [('simple_imputer_before_ord', SimpleImputer(
                    missing_values=np.nan, strategy='most_frequent')),
                 ('ord',  OrdinalEncoder(
                            categories=categories,
                            handle_unknown='use_encoded_value',
                            unknown_value=np.nan
                        )),
                 ('simple_imputer_after_ord', SimpleImputer(
                    missing_values=np.nan,
                    strategy='most_frequent'))]
            )
            transformers.append(('ord', ord_pipe, ord_columns))
        data_preprocessor = ColumnTransformer(
            transformers,
            remainder='passthrough'
        )
        pipeline = Pipeline([
            ('preprocessor', data_preprocessor),
            ('models', CatBoostClassifier(
                random_state=self.random_state,
                depth=10,
                learning_rate=0.1,
                iterations=500,
                loss_function='Logloss',
                verbose=False
            ))
        ])
        return pipeline

    def fit(self, df, target_attr):
        X_train, y_train = self.split_data(df, target_attr)
        self.model = self.create_pipeline(X_train, y_train)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X_pred):
        if self.model is None:
            raise ValueError(
                "Модель не обучена. Сначала вызовите метод fit()")
        return self.model.predict(X_pred)

    def predict_proba(self, X_pred):
        if self.model is None:
            raise ValueError(
                "Модель не обучена. Сначала вызовите метод fit()")
        return self.model.predict_proba(X_pred)


def load_model(model_path):
    return joblib.load(model_path)


def to_snake_case(df):
    df.columns = [
        re.sub(r'[^a-z0-9_]', '', column_name.replace(' ', '_').lower())
        for column_name in df.columns
    ]
    return df


def pre_pocessing(data_path):
    df = pd.read_csv(data_path)
    df = to_snake_case(df)
    df_id = df['id']
    df = df.drop(['id', 'unnamed_0'], axis=1)
    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})
    df['gender'] = pd.to_numeric(df['gender'])
    return df, df_id


def post_processing(df_id, y_prediction):
    y_prediction = pd.Series(y_prediction)
    y_prediction.info()
    predictions = pd.concat([df_id, y_prediction], axis=1)
    predictions.columns = ['id', 'prediction']
    predictions['prediction'] = predictions['prediction'].astype(int)
    return predictions


def predict_class(model, data_path):
    df, df_id = pre_pocessing(data_path)
    predictions = model.predict(df)
    return post_processing(df_id, predictions)


def train_and_save_model():
    df = pre_pocessing(TRAIN_DATA_CSV)
    model = ClassifierModel(random_state=RANDOM_STATE)
    model.fit(df=df, target_attr=TARGET_ATTR)
    joblib.dump(model, MODEL_NAME_PKL)
