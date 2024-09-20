import pandas as pd
import numpy as np

# Load the training and testing dataset using panda
def load_data():
    train_data = pd.read_csv('census+income/adult.data', sep=',', header=None)
    test_data = pd.read_csv('census+income/adult.test', sep=',', skiprows=1, header=None)
    # train_data.loc[0] = categorical_cols
    # test_data.loc[0] = categorical_cols


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
    # categorical_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    #                     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
    #                     'native-country',
    #                     'income']

    for col in range(len(categorical_cols)):
        label_encoders[col] = get_unique_label(train[col])
        train[col] = train[col].map(get_unique_label[col])
        test[col] = test[col].map(get_unique_label[col])


    train_feat = train.iloc[:, :-1]  # Training features
    train_label = train.iloc[:, -1]  # Training labels

    test_feat = test.iloc[:, :-1]  # Test features
    test_label = test.iloc[:, -1]  # Test labels

    print(train_feat)

    return train_feat, train_label, test_feat, test_label



if __name__ == "__main__":
    test_data, train_data = load_data()
    # preprocess(test_data, train_data)
    print(test_data.head())
    print(train_data.head())



