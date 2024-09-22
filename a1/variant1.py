import pandas as pd
import numpy as np
from collections import Counter

ATTRIBUTES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                      'native-country',
                      'income']

# Load the training and testing dataset using panda
def load_data():
    t_data = pd.read_csv('census+income/adult.data', sep=',', header=None)
    tes_data = pd.read_csv('census+income/adult.test', sep=',', skiprows=1, header=None)
    # set column names
    t_data.columns = ATTRIBUTES
    tes_data.columns = ATTRIBUTES

    tes_data['income'] = tes_data['income'].apply(lambda x: x.rstrip('.'))

    return t_data, tes_data


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

    train_f = train.iloc[:, :-1]  # Training features

    train_l = train.iloc[:, -1]  # Training labels
    test_f = test.iloc[:, :-1]  # Test features
    # use median value to replace nan numbers
    test_f = test_f.fillna(test_f.median())
    test_l = test.iloc[:, -1]  # Test labels

    # print(label_encoders)

    return train_f, train_l, test_f, test_l


# calculate euclidean distance of the node to its neighbor
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# an KNN class to do predictions
class KNN:
    def __init__(self, k, train_f, train_l):
        self.k = k
        self.train_feat = train_f
        self.train_label = train_l

    def predict(self, test_f):
        predictions = []
        for i in range(test_f.shape[0]):
            print(f"{i}/{len(test_f)}")
            point_1 = test_f.iloc[i].values
            distances = []
            # random sample a fraction of training data to reduce running time
            sample_train = self.train_feat.sample(frac=0.01, random_state=42)
            for t in range(len(sample_train)):
                point_2 = train_feat.iloc[t].to_numpy()
                d = euclidean_distance(point_1, point_2)
                distances.append(d)
            k_indices = np.argsort(distances)[:self.k]

            k_nearest_labels = [self.train_label[i] for i in k_indices]
            common = Counter(k_nearest_labels).most_common(1)

            predictions.append(common[0][0])

        #
        # for x in test_f.iloc[1:].iterrows():
        #     distances = [euclidean_distance(x, t) for t in self.train_feat]
        #     k_indices = np.argsort(distances)[:self.k]
        #     k_nearest_labels = [self.train_label[i] for i in k_indices]
        #
        #     common = Counter(k_nearest_labels).most_common(1)
        #     predictions.append(common[0][0])

        return np.array(predictions)


# test accuracy of the prediction
def accuracy_test(predictions, test_l):
    return np.sum(predictions == test_l) / len(test_l)


if __name__ == "__main__":
    test_data, train_data = load_data()

    train_feat, train_label, test_feat, test_label = preprocess(test_data, train_data)
    # print(test_feat)
    #
    # # print(test_feat.shape[0])
    knn = KNN(3, train_feat, train_label)
    prediction = knn.predict(test_feat)
    # print(test_label)
    print(accuracy_test(prediction, test_label))
    # print(test_data.head())
    # print(train_data.head())
