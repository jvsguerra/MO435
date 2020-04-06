from create_datasets import Create_Datasets
import numpy as np
import matplotlib.pyplot as plt


# import numpy as np
# import math

# # Debug
# from create_datasets import Create_Datasets

# class MultiClassGenerativeModel:

#     """Class to handle MultiClass Generative Model"""
#     def __init__(self, x_train, y_train):
#         # Training data
#         self.x = x_train
#         self.y = y_train
#         self.pred = np.zeros(y_train.shape)
#         # Data information
#         self.n_classes = None
#         self.n_features = None
#         self.classes = None
#         # Model parameter
#         self.prior = None
#         # Gaussian parameters
#         self.mean = None
#         self.cov = None

#     def fit(self):
#         # Calculate priors for each class (p(c0), p(c1), p(c2))
#         self.get_prior()
#         # Find mean and covariance (same covariance for all classes)
#         self.get_mean_and_cov()

#     def get_prior(self):
#         # Get classes and frequency
#         self.classes, freq = np.unique(self.y, return_counts=True)
#         # Get number of classes
#         self.n_classes = len(self.classes)
#         # Create empty vector for prior
#         self.prior = np.zeros([self.n_classes])
#         # Calculate priors
#         for i in range(self.n_classes):
#             self.prior[i] = freq[i] / freq.sum()

#     def get_mean_and_cov(self):
#         # Get number of features (x)
#         self.n_features = self.x.shape[1]
#         # Get mean vector of each class
#         self.mean = np.zeros([self.n_classes, self.n_features])
#         # Get cov matrix (cov = (cov1 + cov2 + cov3)*1/N)
#         self.cov = np.zeros([self.n_features, self.n_features])
#         for c in self.classes:
#             xc = self.x[self.y == c, : ]
#             self.mean[c, ] = xc.mean(axis=0)
#             xE = xc - self.mean[c]
#             self.cov += xE.transpose().dot(xE)
#         self.cov/self.x.shape[0]

#     def get_class_conditional(self, c, index):
#         # Calculate class conditional densities for class c
#         constant = 1 / ((2 * math.pi) ** (self.n_features / 2))
#         constant *= 1 / (np.linalg.det(self.cov) ** (1 / 2))
#         xE = self.x[index, :] - self.mean[c] 
#         mahalanobis = math.exp(-0.5 * xE.dot(np.linalg.inv(self.cov)).dot(xE.T))
#         class_conditional = constant*mahalanobis
#         return class_conditional

#     def predict(self):
#         for index in range(self.x.shape[0]):
#             class_conditional = np.zeros(self.n_classes)
#             conditional_probability = np.zeros(self.n_classes)
#             numerator = np.zeros(self.n_classes)
#             # Calculate each class conditional
#             for c in self.classes:
#                 class_conditional[c] = self.get_class_conditional(c, index)
#                 numerator[c] = class_conditional[c] * self.prior[c]
#             for c in self.classes:
#                 conditional_probability[c] = numerator[c] / numerator.sum()
#             print(conditional_probability.max)
#             self.pred[index] = np.argmax(conditional_probability)                    
#         return
    
#     def scatter(self):
#         return

# if __name__ == "__main__":
#     # NOTE: debug! Create a dataset here
#     seed = 0
#     dataset = Create_Datasets(n_samples=150, method='classification', seed=seed)
#     dataset.split_test_and_training(test_frac=0.5)
#     dataset.scatter()
#     #######

#     # Create model
#     model = MultiClassGenerativeModel(dataset.x_train, dataset.y_train)

#     # Fit model
#     model.fit()

#     # Predict
#     model.predict()
#     # print(model.pred)
#     # print(model.y)



class MultiClassLogisticRegression:

    def __init__(self):
        return
    







if __name__ == "__main__":
    # Create dataset
    dataset = Create_Datasets(150, method='classification', seed=0)

    # Split in train and test partitions
    dataset.split_test_and_training(test_frac=0.5)
    # print(dataset.x)

    print(dataset.x_train[dataset.y_train == 0])
    print(dataset.x_train[dataset.y_train == 1])
    print(dataset.x_train[dataset.y_train == 2])
    print(dataset.x_train[dataset.y_train == 0].shape)
    print(dataset.x_train[dataset.y_train == 1].shape)
    print(dataset.x_train[dataset.y_train == 2].shape)

