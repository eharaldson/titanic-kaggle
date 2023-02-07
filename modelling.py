import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, model_selection

def eda(df):
    print(df.head())
    print()
    print(df.info())
    print()
    print(df.describe())

if __name__ == "__main__":
    
    df = pd.read_csv('train.csv')

    eda(df)

    # le = preprocessing.LabelEncoder()
    # df['Sex'] = le.fit_transform(df['Sex'])

    # print(df.head())

    # X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']]
    # y = df['Survived']

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # logistic_model = linear_model.LogisticRegression()

    # logistic_model.fit(X_train, y_train)

    # print(logistic_model.score(X_test, y_test))