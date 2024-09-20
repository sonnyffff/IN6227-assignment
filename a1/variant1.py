import pandas as pd
import numpy as np

# Load the training and testing dataset using panda
def load_data():
    train_data = pd.read_csv('census+income/adult.data', sep=',', header=None)
    test_data = pd.read_csv('census+income/adult.test', sep=',', skiprows=1, header=None)
    # set column names
    train_data.columns =  ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country',
                        'income']
    test_data.columns =  ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country',
                        'income']


    return train_data, test_data

# map unique labels to a number
def get_unique_label(col):
    unique_labels = col.unique()
    ret = {}
    for idx, value in enumerate(unique_labels):
        ret[value] = idx

    return ret


# preprocess the datasets
def preprocess(train, test):

    label_encoders = {}
    categorical_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country',
                         'income']

    for col in categorical_cols:
        # use a numbers to represent class attributes
        label_encoders[col] = get_unique_label(train[col])
        train[col] = train[col].map(label_encoders[col])
        test[col] = test[col].map(label_encoders[col])


    train_feat = train.iloc[:, :-1]  # Training features

    train_label = train.iloc[:, -1]  # Training labels

    test_feat = test.iloc[:, :-1]  # Test features
    test_label = test.iloc[:, -1]  # Test labels

    print(label_encoders)

    return train_feat, train_label, test_feat, test_label



if __name__ == "__main__":
    test_data, train_data = load_data()
    preprocess(test_data, train_data)
    # print(test_data.head())
    # print(train_data.head())



