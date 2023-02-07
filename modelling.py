import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, model_selection

def eda(df):
    print(df.head())
    print()
    print(df.info())
    print()
    print(df.describe())

def generate_clean_X_y(df):

    df = df.fillna(df.mean(numeric_only=True))
    le = preprocessing.LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'].to_list())
    df = df.drop(['Cabin'], axis=1)
    df = df.dropna(axis=0)
    df['Embarked'] = le.fit_transform(df['Embarked'])

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']]
    y = df['Survived']
    return X, y

def logistic_regression_eval(X_train, X_test, y_train, y_test):
    logistic_model = linear_model.LogisticRegression()
    param_grid = {'penalty': ['l1', 'l2'],
                  'C': [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]}
    gs = model_selection.GridSearchCV(estimator=logistic_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)

    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

if __name__ == "__main__":
    
    df = pd.read_csv('train.csv')

    X, y = generate_clean_X_y(df)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    print(logistic_regression_eval(X_train, X_test, y_train, y_test))