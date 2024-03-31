# imports
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

train_data = None
test_data = None
le = None
x_train = x_valid = y_train = y_valid = x_test = y_test = None
x_train_std = x_test_std = None
forest = None
random_grid = None
rf_best_values = None


# import data
def import_data():
    global train_data, test_data
    train_data = pd.read_csv('data/train.csv', index_col='id')
    test_data = pd.read_csv('data/test.csv', index_col='id')
    get_data_meta_info()


def get_data_meta_info():
    print(train_data.info())


def peek_at_data(rows=5):
    print(train_data.head(rows))


def encode_convert_features():
    # All the features are either float type or object type
    global le
    le = LabelEncoder()
    for column in train_data.columns:
        # Check if the column dtype is object
        if train_data[column].dtype == 'object':
            # Get unique values of the column
            unique_values = train_data[column].unique()
            print(f"Unique values in '{column}': {unique_values}")
            if column != 'NObeyesdad':
                train_data[column] = le.fit_transform(train_data[column])
                train_data[column] = train_data[column].astype(float)

    peek_at_data(5)


def check_data_missing():
    # Data Preparation
    # 1. check if there are any missing/null values, if found,
    # drop the row containing null
    # 2. check duplicate values, different rows in data could
    # have same exact values for all features, find them, get rid of them

    # Null values feature wise
    print(f'Missing value(s): {train_data.isna().sum()}')

    # find duplicate rows
    print(f'Total dup row(s): {train_data.duplicated().sum()}')


def split_data_train_validate():
    global x_train, x_valid, y_train, y_valid, x_test, y_test
    # 3 Way hold out
    # Target Feature
    print(train_data.shape)
    y = train_data['NObeyesdad']
    print(train_data.NObeyesdad.value_counts())

    train_data.drop(columns=['NObeyesdad'], inplace=True)
    print(train_data.head(5))

    x_temp, x_test, y_temp, y_test = train_test_split(train_data, y,
                                                      test_size=0.2,
                                                      shuffle=True,
                                                      random_state=121,
                                                      stratify=y)
    x_train, x_valid, y_train, y_valid = train_test_split(train_data, y,
                                                          test_size=0.2,
                                                          shuffle=True,
                                                          random_state=129,
                                                          stratify=y)


# TODO:
# 1. might need to keep track of what value got encoded to a number using LabelEncoder
# 2. Test the trained model with validation data set too


def scale_data(x1, x2):
    # global x_train_std, x_test_std
    sc = StandardScaler()
    if x1 is not None:
        new_x1 = sc.fit_transform(x1)
    else:
        new_x1 = None

    if x2 is not None:
        new_x2 = sc.transform(x2)
    else:
        new_x2 = None

    return new_x1, new_x2


def train_model(params=None):
    global rf_best_values, forest
    if params is not None:
        print('Using GridSearch values')
        forest = RandomForestClassifier(**params)
    else:
        forest = RandomForestClassifier(n_estimators=100, random_state=1)
    forest.fit(x_train_std, y_train)


def get_evaluation_metrics(input_local, target, data_type, show_confusion=False):
    # input_local : x_test_std
    # target : y_test
    global forest
    # print('Training accuracy:', forest.score(X_train_std, y_train))
    rf_score = forest.score(input_local, target)
    print(f'Test accuracy RForest for {data_type}: {round((rf_score * 100), 3)}%')
    y_pred_all_features = forest.predict(input_local)
    cm_all_features = confusion_matrix(target, y_pred_all_features)
    if show_confusion:
        print(f'Confusion matrix for {data_type}: {cm_all_features}')

    return round((rf_score * 100), 3)


def get_hyper_params_tune():
    print('Setting up Grid to fine tune the classifier')
    global random_grid
    # Set of hyperparameters to search
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1500, num=15)]
    max_features = ['sqrt', 'log2', None]
    max_depth = [int(x) for x in np.linspace(start=10, stop=800, num=10)]
    # this makes sure that number of samples as in list below should be
    # available to split the node else the node becomes the leaf node.
    min_samples_split = [2, 5, 10, 14]
    # prevents overfitting
    min_samples_leaf = [2, 4, 6, 8]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': ['gini', 'entropy', 'log_loss']
    }
    print(f'Grid values {random_grid}')


def do_tune():
    global x_train_std, y_train, random_grid, rf_best_values
    # Initialize Random Forest classifier
    rf = RandomForestClassifier()

    # Initialize RandomizedSearchCV
    rf_gird_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                        n_iter=100, verbose=False,
                                        random_state=42, n_jobs=None)

    # Fit RandomizedSearchCV to the training data
    rf_best_values = rf_gird_search.fit(x_train_std, y_train)

    print("Best parameters found by RandomizedSearchCV:")
    print(rf_best_values.best_params_)

    print("Best score found by RandomizedSearchCV:")
    print(rf_best_values.best_score_)


def save_model():
    global forest
    dump(forest, 'model.joblib')


if __name__ == "__main__":
    do_grid = False
    import_data()
    encode_convert_features()
    check_data_missing()
    split_data_train_validate()
    x_train_std, x_test_std = scale_data(x_train, x_test)
    train_model(None)
    score_test = get_evaluation_metrics(x_test_std, y_test, 'test data', True)

    # test model on validation data
    x_valid_std, _ = scale_data(x_valid, None)
    score_valid = get_evaluation_metrics(x_valid_std, y_valid, 'validation data', False)

    if score_valid < 95.0 and do_grid:
        print('Validation score is less than 95.0%')
        print('Find best possible params to train the RF classifier')
        get_hyper_params_tune()
        do_tune()
        train_model(None)
        score = get_evaluation_metrics(x_valid_std, y_valid, 'validation data after grid search', True)
        if score > score_valid:
            save_model()

    best_params = {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2',
                   'max_depth': 185, 'criterion': 'gini'}
    train_model(best_params)
    score = get_evaluation_metrics(x_valid_std, y_valid, 'validation data after grid search', False)
    save_model()
    dump(le, 'le.joblib')
