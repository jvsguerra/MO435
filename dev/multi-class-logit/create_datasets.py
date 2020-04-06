from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import math

# Debug
import pandas as pd
from sklearn.model_selection import train_test_split

class Create_Datasets:
    """Class to create datasets and partition test and train sets"""
    def __init__(self, n_samples=150, method='classification', seed=0):
        self.seed = seed
        self.n_samples = n_samples
        if method == 'classification':
            self.x, self.y = self.classification()
        elif method == 'blobs':
            self.x, self.y = self.blobs()
        else:
            raise ValueError("invalid positional argument: \'method\'")
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None

    def classification(self):
        x, y = make_classification(n_samples=self.n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3, random_state=self.seed)
        return x, y

    def blobs(self):
        x, y = make_blobs(n_samples=self.n_samples, n_features=2, centers=None, random_state=self.seed)
        return x, y

    def scatter(self):
        plt.scatter(x=self.x[:, 0], y=self.x[:, 1], c=self.y, marker='o')
        plt.show()

    # def split_test_and_training(self, test_size=0.5):
    #     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=self.seed)

    def split_test_and_training(self, test_frac=0.5):
        # Sort features (x) and classes (y)
        self.x = np.asarray([ [x1, x2] for _,x1,x2 in sorted(zip(self.y, self.x[:, 0], self.x[:, 1]))])
        self.y = np.sort(self.y)
        # Get classes of dataset (0,1,2)
        classes = np.unique(self.y)
        # Get indexes for test partition
        initial = 0
        samples = 0
        total_samples = self.y.shape[0]
        test_samples = math.ceil(total_samples*test_frac)
        indexes = np.zeros([test_samples], dtype=int)
        for c in range(len(classes)):
            np.random.seed(self.seed)
            class_size = self.x[ self.y == classes[c], :].shape[0]
            class_test_size = math.ceil(class_size*test_frac)
            while (initial + class_test_size > test_samples):
                class_test_size -= 1
            end = initial + class_test_size
            indexes[initial:end] = samples+np.random.choice(class_size, class_test_size, replace=False)
            initial += class_test_size
            samples += class_size
        # Prepare test dataset (x_test, y_test)
        self.x_test = self.x[indexes, :]
        self.y_test = self.y[indexes]
        # Prepare train dataset (x_train, y_train)
        self.x_train = self.x[[ i for i in range(self.n_samples) if i not in indexes ], :]
        self.y_train = self.y[[ i for i in range(self.n_samples) if i not in indexes ]]

    def export_csv(self, name):
        # Export dataset 
        data = pd.DataFrame(np.column_stack((self.x, self.y)))
        data = data.astype({0:float, 1:float, 2:int})
        data.to_csv(f"{name}.csv")

    def export_test_and_train(self, name):
        # Export test partition
        data = pd.DataFrame(np.column_stack((self.x_test, self.y_test)))
        data = data.astype({0:float, 1:float, 2:int})
        data.to_csv(f"{name}_test.csv")
        # Export train partition
        data = pd.DataFrame(np.column_stack((self.x_train, self.y_train)))
        data = data.astype({0:float, 1:float, 2:int})
        data.to_csv(f"{name}_train.csv")


if __name__ ==  "__main__":
    # Set seed 
    seed=0
    # Create dataset with make_classifcation
    dataset1 = Create_Datasets(n_samples=150, method='classification', seed=seed)
    # Create dataset with make_blobs
    dataset2 = Create_Datasets(n_samples=150, method='blobs', seed=seed)
    # Split data in test and train partitions
    dataset1.split_test_and_training(test_frac=0.5)
    dataset2.split_test_and_training(test_frac=0.5)
    # Scatter plots
    # dataset1.scatter()
    # dataset2.scatter()
    
    # Debug: export dataset and partitions
    # dataset1.export_csv('input/data1')
    # dataset1.export_test_and_train(name='input/data1')
    # dataset2.export_csv('input/data2')
    # dataset2.export_test_and_train(name='input/data2')