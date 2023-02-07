import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, model_selection, tree, ensemble, naive_bayes

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

def decision_tree_eval(X_train, X_test, y_train, y_test):
    decision_tree_model = tree.DecisionTreeClassifier()
    param_grid = {'max_depth': [5, 10, 25, 50, 75, 100, 150]}
    gs = model_selection.GridSearchCV(estimator=decision_tree_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)
    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

def random_forest_eval(X_train, X_test, y_train, y_test):
    random_forest_model = ensemble.RandomForestClassifier()
    param_grid = {'n_estimators': [5, 10, 25, 50, 75, 100, 150],
                  'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}
    gs = model_selection.GridSearchCV(estimator=random_forest_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)
    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

def gradient_boost_eval(X_train, X_test, y_train, y_test):
    gradient_boost_model = ensemble.GradientBoostingClassifier()
    param_grid = {'n_estimators': [5, 10, 25, 50, 75, 100, 150],
                  'loss': ['log_loss', 'exponential'],
                  'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]}
    gs = model_selection.GridSearchCV(estimator=gradient_boost_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)
    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

def gaussian_process_eval(X_train, X_test, y_train, y_test):
    gaussian_process_model = naive_bayes.GaussianNB()
    gaussian_process_model.fit(X_train, y_train)
    return {'score': gaussian_process_model.score(X_test, y_test),
            'model': gaussian_process_model}

if __name__ == "__main__":
    
    df = pd.read_csv('train.csv')

    X, y = generate_clean_X_y(df)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    print(gaussian_process_eval(X_train, X_test, y_train, y_test))