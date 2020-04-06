import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Debug
from create_datasets import Create_Datasets
from sklearn.metrics import confusion_matrix

class MultiClassGenerativeModel:

    """Class to handle MultiClass Generative Model"""
    def __init__(self, x_train, y_train):
        # Training data
        self.x = x_train
        self.y = y_train
        self.y_pred = np.zeros(y_train.shape)
        # Data information
        self.n_classes = None
        self.n_features = None
        self.classes = None
        # Model parameter
        self.prior = None
        # Gaussian parameters
        self.mean = None
        self.cov = None

    def fit(self):
        # Calculate priors for each class (p(c0), p(c1), p(c2))
        self.get_prior()
        # Find mean and covariance (same covariance for all classes)
        self.get_mean_and_cov()
        # Predict classes of training set (self.x)
        self.y_pred = self.predict(self.x)

    def get_prior(self):
        # Get classes and frequency
        self.classes, freq = np.unique(self.y, return_counts=True)
        # Get number of classes
        self.n_classes = len(self.classes)
        # Create empty vector for prior
        self.prior = np.zeros([self.n_classes])
        # Calculate priors
        for i in range(self.n_classes):
            self.prior[i] = freq[i] / freq.sum()

    def get_mean_and_cov(self):
        # Get number of features (x)
        self.n_features = self.x.shape[1]
        # Get mean vector of each class
        self.mean = np.zeros([self.n_classes, self.n_features])
        # Get cov matrix (cov = (cov1 + cov2 + cov3)*1/N)
        self.cov = np.zeros([self.n_features, self.n_features])
        for c in self.classes:
            xc = self.x[self.y == c, : ]
            self.mean[c, ] = xc.mean(axis=0)
            xE = xc - self.mean[c]
            self.cov += xE.transpose().dot(xE)
        self.cov/self.x.shape[0]

    def get_class_conditional(self, c, index):
        # Calculate class conditional densities for class c
        constant = 1 / ((2 * math.pi) ** (self.n_features / 2))
        constant *= 1 / (np.linalg.det(self.cov) ** (1 / 2))
        xE = self.x[index, :] - self.mean[c] 
        mahalanobis = math.exp(-0.5 * xE.dot(np.linalg.inv(self.cov)).dot(xE.T))
        class_conditional = constant*mahalanobis
        return class_conditional

    def predict(self, x):
        y_pred = np.zeros(self.y.shape)
        for index in range(self.x.shape[0]):
            class_conditional = np.zeros(self.n_classes)
            conditional_probability = np.zeros(self.n_classes)
            numerator = np.zeros(self.n_classes)
            # Calculate each class conditional
            for c in self.classes:
                class_conditional[c] = self.get_class_conditional(c, index)
                numerator[c] = class_conditional[c] * self.prior[c]
            for c in self.classes:
                conditional_probability[c] = numerator[c] / numerator.sum()
            # print(conditional_probability.max)
            y_pred[index] = np.argmax(conditional_probability)
        return y_pred                  

    def score(self, y, y_pred):
        # score = correct_predictions / all_predictions
        score = np.sum(y == y_pred) / y.shape[0]
        return score
    
    def scatter(self, x, y, c, ax = None):
        ax = ax
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis([-5, 5, -5, 5])
        scatter = ax.scatter(x=x, y=y, c=c)
        return scatter


# def run_analysis(seed):
#     ######## Create Datasets ########
#     # Dataset 1
#     classification = Create_Datasets(n_samples=150, method='classification', seed=seed)
#     classification.split_test_and_training(test_frac=0.5)
#     # Dataset 2
#     blobs = Create_Datasets(n_samples=150, method='blobs', seed=seed)
#     blobs.split_test_and_training(test_frac=0.5)

#     ######## Create Models ########
#     # Dataset 1
#     c_model = MultiClassGenerativeModel(classification.x_train, classification.y_train)
#     # Dataset 2
#     b_model = MultiClassGenerativeModel(blobs.x_train, blobs.y_train)

#     ######## Fit Models ########
#     # Dataset 1
#     c_model.fit()
#     # Dataset 2
#     b_model.fit()

#     ######## Score Training set ########
#     # Dataset 1
#     c_score = c_model.score(c_model.y, c_model.y_pred)
#     # Dataset 2
#     b_score = b_model.score(b_model.y, b_model.y_pred)


if __name__ == "__main__":
    # NOTE: debug! Create a dataset here
    seed = 0
    dataset = Create_Datasets(n_samples=150, method='classification', seed=seed)
    dataset.split_test_and_training(test_frac=0.5)
    dataset.scatter('output/scatter_dataset.png')
    #######

    # Create model
    model = MultiClassGenerativeModel(dataset.x_train, dataset.y_train)

    # Fit model
    model.fit()

    # Score training set (accuracy)
    score = model.score(model.y, model.y_pred)
    print(score)

    # Print scatter (testing data)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    model.scatter(model.x[:, 0], model.x[:, 1], model.y, ax1)
    model.scatter(model.x[:, 0], model.x[:, 1], model.y_pred, ax2)
    fig.tight_layout()
    fig.savefig('output/scatter_trainset.png')

    # Confusion matrix (training set set)
    cm = confusion_matrix(model.y, model.y_pred)

    # Plot confusion matrix
    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    # plt.show()

    # Predict test set
    y_pred_test = model.predict(dataset.x_test)

    # Score testing set (accuracy)
    score_test = model.score(dataset.y_test, y_pred_test)
    print(score_test)

    # Print scatter (testing data)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    model.scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], dataset.y_test, ax1)
    model.scatter(dataset.x_test[:, 0], dataset.x_test[:, 1], y_pred_test, ax2)
    fig.tight_layout()
    fig.savefig('output/scatter_testset.png')

    # Confusion matrix (testing set)
    cm = confusion_matrix(dataset.y_test, y_pred_test)

    # Plot confusion matrix
    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    # plt.show()

    