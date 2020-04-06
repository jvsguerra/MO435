import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from create_datasets import Create_Datasets

# Debug
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report

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
        self.prior = freq/freq.sum()

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
        self.cov = self.cov/self.x.shape[0]

    def get_class_conditional(self, x):
        """
        Class conditional density has the following form:
        p(x|Ci) = constant*exp(mahalanobis)
        """
        # Create an empty array   
        class_conditional = np.zeros((x.shape[0], self.n_classes))
      
        # Get constant value
        constant = 1 / ((2 * math.pi) ** (self.n_features) * np.linalg.det(self.cov) ** (1/2))

        # Get class conditional densities
        prec = np.linalg.inv(self.cov)
        for c in range(self.n_classes):
            X = x - self.mean[c]
            class_conditional[:, c] = constant*np.exp(-0.5 * np.einsum('ij,jk,ki->i', X, prec, X.T))
        return class_conditional

    def get_posterior(self, x):
        """
        Posterior probability p(Ci|x) has the following form:
        p(Ci|x) = p(x|Ci)*p(Ci)/sum(p(x|Cj)*p(Cj))
        numerator = p(x|Ci)*p(Ci)
        denominator = sum(p(x|Cj)*p(Cj))
        """
        # Get class conditional densities
        posterior = self.get_class_conditional(x)
        
        # Get numerator
        posterior = np.dot(posterior, np.diag(self.prior))
        posterior = np.dot(np.diag(1 / posterior.sum(axis=1)), posterior)

        return posterior

    def predict(self, x):
        y_pred = np.argmax(self.get_posterior(x), axis=1)
        return y_pred          

    def score(self, y, y_pred):
        # score = correct_predictions / all_predictions
        score = np.sum(y == y_pred) / y.shape[0]
        return score


class PerformanceEvaluation:
    """
    Class to evaluate models
    """
    def __init__(self, y_expected_train, y_predicted_train, y_expected_test, y_predicted_test):
        self.y_expected_train = y_expected_train
        self.y_predicted_train = y_predicted_train
        self.y_expected_test = y_expected_test
        self.y_predicted_test = y_predicted_test

    def train_metrics(self):
        print("> Performance evaluation of training set:")
        cm = self.confusion_matrix(self.y_expected_train, self.y_predicted_train)
        p, r = self.precision_and_recall(cm)
        print("{:>8}\t{:>8}\t{:>8}\n".format("classes", "precision", "recall"), end='')
        for i in range(3):
            print("{:>8}\t{:>8.1%}\t{:>8.1%}\n".format(i, p[i], r[i]), end='')
        return

    def test_metrics(self):
        print("> Performance evaluation of testing set:")
        cm = self.confusion_matrix(self.y_expected_test, self.y_predicted_test)
        p, r = self.precision_and_recall(cm)
        print("{:>8}\t{:>8}\t{:>8}\n".format("classes", "precision", "recall"), end='')
        for i in range(3):
            print("{:>8}\t{:>8.1%}\t{:>8.1%}\n".format(i, p[i], r[i]), end='')
        return

    def confusion_matrix(self, expected, predicted):
        """
        Determine a confusion matrix.
        """
        # Get number of classes
        n_classes = len(np.unique(expected))
        # Create empty vector
        confusion_matrix = np.zeros((n_classes, n_classes))
        # Get confusion matrix
        for index in range(expected.shape[0]):
            confusion_matrix[expected[index]][predicted[index]] += 1
        return confusion_matrix

    def precision_and_recall(self, confusion_matrix):
        # Get number of classes
        n_classes = confusion_matrix.shape[0]
        # Create empty vector
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        # Get precision and recall
        for i in range(n_classes):
            precision[i] = confusion_matrix[i, i]/confusion_matrix[:, i].sum()
            recall[i] = confusion_matrix[i, i]/confusion_matrix[i, :].sum()
        return precision, recall


def run_generative_analysis(seed, method, base_name, lim):
    # Start analysis
    print(f"[==> Running analysis for dataset: {base_name}")

    # Create dataset
    dataset = Create_Datasets(n_samples=150, n_classes=3, method=method, seed=seed)
    dataset.split_test_and_training(test_frac=0.5)
    
    # Scatter plot of dataset
    dataset.scatter(lim=lim, output=f"output/scatter_{base_name}.png", )

    # Create model
    model = MultiClassGenerativeModel(dataset.x_train, dataset.y_train)

    ####### Training step #######
    # Fit model
    model.fit()

    # # Score training set (accuracy)
    score = model.score(model.y, model.y_pred)
    print(f"> Accuracy of the training set: {score:.1%}")
    #############################    
    
    ####### Testing step #######
    # Predict testing set
    y_pred_test = model.predict(dataset.x_test)

    # # Score testing set (accuracy)
    score_test = model.score(dataset.y_test, y_pred_test)
    print(f"> Accuracy of the testing set: {score_test:.1%}")
    #############################

    ####### Scatter plots with contour #######
    pts = 200
    x1 = np.linspace(*(lim[0], lim[1], pts))
    x2 = np.linspace(*(lim[2], lim[3], pts))
    x1, x2 = np.meshgrid(x1, x2)
    x = np.array([x1.reshape(pts*pts), x2.reshape(pts*pts)]).T
    interval = pts // 100
    gap = int(pts * pts / interval)
    prediction = np.zeros([pts*pts])
    for i in range(interval):
        start = int(i * gap)
        end = int(start + gap)
        prediction[start:end] = model.predict(x[start:end, :])
    prediction = prediction.reshape(pts, pts)

    # Plot 1: Data with contour
    fig, ax = plt.subplots()
    ax.axis(lim)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.contourf(x1, x2, prediction, alpha=0.2)
    ax.contour(x1, x2, prediction, colors='k')
    scatter = ax.scatter(x=dataset.x[:, 0], y=dataset.x[:, 1], c=dataset.y)
    ax.legend(*scatter.legend_elements(), loc='best', title='Classes')
    fig.tight_layout()
    fig.savefig(f"output/scatter_with_contour_{base_name}", dpi=300)

    # Plot 2: predictions with contour
    fig, ax = plt.subplots()
    ax.axis(lim)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.contourf(x1, x2, prediction, alpha=0.2)
    ax.contour(x1, x2, prediction, colors='k')
    scatter = ax.scatter(x=model.x[:, 0], y=model.x[:, 1], c=model.y_pred)
    ax.scatter(x=dataset.x_test[:, 0], y=dataset.x_test[:, 1], c=y_pred_test)
    ax.legend(*scatter.legend_elements(), loc='best', title='Classes')
    fig.tight_layout()
    fig.savefig(f"output/scatter_predictions_with_contour_{base_name}", dpi=300)
    ##########################################

    ####### Performance Evaluation #######
    metrics = PerformanceEvaluation(model.y, model.y_pred, dataset.y_test, y_pred_test)
    metrics.train_metrics()
    metrics.test_metrics()
    ######################################
   