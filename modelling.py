import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, model_selection, tree, ensemble, naive_bayes, metrics
import torch
import torch.utils.data
import torch.utils.tensorboard

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

def logistic_regression_eval(X_train, y_train):
    logistic_model = linear_model.LogisticRegression()
    param_grid = {'penalty': ['l1', 'l2'],
                  'C': [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]}
    gs = model_selection.GridSearchCV(estimator=logistic_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)

    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

def decision_tree_eval(X_train, y_train):
    decision_tree_model = tree.DecisionTreeClassifier()
    param_grid = {'max_depth': [5, 10, 25, 50, 75, 100, 150]}
    gs = model_selection.GridSearchCV(estimator=decision_tree_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)
    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

def random_forest_eval(X_train, y_train):
    random_forest_model = ensemble.RandomForestClassifier()
    param_grid = {'n_estimators': [5, 10, 25, 50, 75, 100, 150],
                  'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}
    gs = model_selection.GridSearchCV(estimator=random_forest_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)
    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

def gradient_boost_eval(X_train, y_train):
    gradient_boost_model = ensemble.GradientBoostingClassifier()
    param_grid = {'n_estimators': [5, 10, 25, 50, 75, 100, 150],
                  'loss': ['log_loss', 'exponential'],
                  'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]}
    gs = model_selection.GridSearchCV(estimator=gradient_boost_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)
    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

def gaussian_process_eval(X_train, y_train):
    gaussian_process_model = naive_bayes.GaussianNB()
    gaussian_process_model.fit(X_train, y_train)
    return {'score': gaussian_process_model.score(X_train, y_train),
            'model': gaussian_process_model}

def ada_boost_eval(X_train, y_train):
    ada_boost_model = ensemble.AdaBoostClassifier()
    param_grid = {'n_estimators': [5, 10, 25, 50, 75, 100, 150],
                  'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]}
    gs = model_selection.GridSearchCV(estimator=ada_boost_model, param_grid=param_grid, n_jobs=-1)
    gs.fit(X_train, y_train)
    return {'score': gs.best_score_,
            'model': gs.best_estimator_,
            'params': gs.best_params_}

class NNClassification(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, features):
        return self.layers(features.float())

class TitanicDataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super().__init__()
        df = pd.read_csv('train.csv')
        self.X, self.y = generate_clean_X_y(df)

    def __getitem__(self, index):
        return torch.tensor(self.X.iloc[index,:]), torch.tensor(self.y.iloc[index])

    def __len__(self) -> int:
        return len(self.y)

def train(model, dataloader, lr=0.001, epochs=30):

    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)

    writer = torch.utils.tensorboard.SummaryWriter()

    batch_index = 0

    validation_losses = []

    for epoch in range(epochs):
        for batch in dataloader['train']:
            features, labels = batch
            predictions = model(features)
            predictions = predictions.view(-1)

            loss = torch.functional.F.binary_cross_entropy(predictions.float(), labels.float())

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            val_features, val_labels = next(iter(dataloader['validation']))
            val_predictions = model(val_features)   
            val_predictions = val_predictions.view(-1)

            val_loss = torch.functional.F.binary_cross_entropy(val_predictions.float(), val_labels.float())
            validation_losses.append(val_loss.item())

            writer.add_scalar(tag='Loss', scalar_value=loss.item(), global_step=batch_index)
            writer.add_scalar(tag='Validation Loss', scalar_value=val_loss.item(), global_step=batch_index)
            batch_index += 1

    return model

def label_from_proba(probs):
    positive_class = probs > 0.5
    return positive_class.int()

def nn_eval():

    data = TitanicDataset()

    train_data, validation_data, test_data = torch.utils.data.random_split(data, [0.7, 0.15, 0.15])

    batch_size = 32
    data_loaders = {
        'train': torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        ),
        'validation': torch.utils.data.DataLoader(
            validation_data,
            batch_size=len(validation_data),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        ),
        'test': torch.utils.data.DataLoader(
            test_data,
            batch_size=len(test_data),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
    }

    nn = NNClassification()

    nn = train(nn, data_loaders, epochs=60)

    features, labels = next(iter(data_loaders['validation']))
    predictions = nn(features)

    predictions = label_from_proba(predictions)

    validation_accuracy = metrics.accuracy_score(labels.numpy(), predictions.detach().numpy())

    print(validation_accuracy)

if __name__ == "__main__":

    nn_eval()