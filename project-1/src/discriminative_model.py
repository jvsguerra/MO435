import numpy as np
import matplotlib.pyplot as plt
import math
from create_datasets import Create_Datasets

class LogisticRegressionDiscriminativeModel:

    def __init__(self, x_train, y_train, learning_rate):
        # Training data
        self.size = float(len(y_train))
        self.X = np.c_[np.ones(int(self.size)), x_train]
        self.y = y_train.reshape(len(y_train), 1) 
        self.learning_rate = learning_rate
        self.theta = np.random.randn(len(self.X[0]),1)

    def fit(self, return_costs=False):
        all_costs = self.gradient_descent()
        if return_costs:
            return all_costs

    # Following chapter 8.4.4.1 of "Machine Learning: A Probabilistic Perspective" [Murphy 2012-08-24]
    # Following this paper: https://www.researchgate.net/publication/228556635_Efficient_Monte_Carlo_Methods_for_Conditional_Logistic_Regression
    def predict(self, X, n_samples=1000, return_normal=False):
        # TODO: Define cov
        # Predicting using Monte Carlo
        X = np.c_[np.ones(len(X)), X]
        cov = abs(np.diag(self.theta.reshape(-1))/10)
        normal = np.random.multivariate_normal(self.theta.reshape(-1), cov, n_samples)

        prediction = (1/float(n_samples))*np.sum(1 / (1 + pow(math.e, -1*X.dot(normal.T))), axis=1) >= 0.5
        if return_normal:
            return normal, prediction
        else:
            return prediction

    def hypothesis(self, X):
        return 1 / (1 + pow(math.e, -1*X.dot(self.theta)))

    def cost(self):
        hypothesis_value = self.hypothesis(self.X)
        return (1/self.size)*np.sum(-1 * (1 - self.y) * np.log(1 - hypothesis_value) - self.y * np.log(hypothesis_value))

    # Following this paper: https://iopscience.iop.org/article/10.1088/1757-899X/495/1/012003/pdf
    # Following chapter 8.3.2 of "Machine Learning: A Probabilistic Perspective" [Murphy 2012-08-24]
    def gradient_descent(self, n_iterations=20000):
        all_costs = []
        all_costs.append(self.cost())
        for i in range(n_iterations):
            self.theta = self.theta - (1/self.size)*self.learning_rate*(self.X.T.dot(self.hypothesis(self.X) - self.y))
            new_cost = self.cost()

            # Stop criterea
            if abs(new_cost - all_costs[-1]) < 1e-9:
                break
            all_costs.append(new_cost)

        return all_costs

def run_discriminative_analisys():

    learning_rates = [0.0000001, 0.00001, 0.0001, 0.001, 1, 10, 30, 32, 34, 34.5, 34.8, 34.9]
    samples = [100, 1000, 10000, 100000]

    dataset = Create_Datasets(n_samples=100, n_classes=2)
    dataset.split_test_and_training(test_frac=0.5)
    lim = [-4, 5, -3.5, 4]
    dataset.scatter(lim=lim, output=f"output/scatter_3.png", )
    print("[")
    costs = []
    for learning_rate in learning_rates:
        for sample in samples:

            model = LogisticRegressionDiscriminativeModel(dataset.x_train, dataset.y_train, learning_rate)

            model.fit()

            predicted_train = model.predict(dataset.x_train, sample)
            accuracy_train = np.sum(dataset.y_train == predicted_train)/float(len(dataset.y_train))
            normal_test, predicted_test = model.predict(dataset.x_test, sample, return_normal=True)
            accuracy_test = np.sum(dataset.y_test == predicted_test)/float(len(dataset.y_test))
            #print("learning_rate:", learning_rate, "sample:", sample, "accuracy_train:", accuracy_train, "accuracy_test:", accuracy_test)

            #print()
            precision0t, precision1t, recall0t, recall1t, accuracy0t, accuracy1t = test_measures(predicted_train, dataset.y_train)
            #print("For train:")
            #print("precision0:",precision0, "precision1:",precision1,"recall0:", recall0, "recall1",recall1, "accuracy0:", accuracy0, "accuracy1:", accuracy1 )
            #print()

            precision0, precision1, recall0, recall1, accuracy0, accuracy1 = test_measures(predicted_test, dataset.y_test)
            #print("For test:")
            print("[",precision0t, ",",precision1t,",", recall0t, ",",recall1t, ",", accuracy0t, ",", accuracy1t,",",precision0, ",",precision1,",", recall0, ",",recall1, ",", accuracy0, ",", accuracy1 ,"]")
            #print()
            # plot normal
            plt.close()
            plt.clf()
            plt.hist(normal_test[:,0], bins=100)
            plt.xlabel("w")
            plt.ylabel("Quantity")
            plt.title("Histogram of W related to bias")
            plt.savefig(f"output/hist_featbias_{int(learning_rate*10000000)}_sample_{sample}.png", dpi=300)
            plt.close()
            plt.clf()
            plt.hist(normal_test[:,1], bins=100)
            plt.xlabel("w")
            plt.ylabel("Quantity")
            plt.title("Histogram of W related to feature 1")
            plt.savefig(f"output/hist_feat0_{int(learning_rate*10000000)}_sample_{sample}.png", dpi=300)
            plt.close()
            plt.clf()
            plt.hist(normal_test[:,2], bins=100)
            plt.xlabel("w")
            plt.ylabel("Quantity")
            plt.title("Histogram of W related to feature 2")
            plt.savefig(f"output/hist_feat1_{int(learning_rate*10000000)}_sample_{sample}.png", dpi=300)
            plt.close()
            plt.clf()

            # Plot Contours
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

            # Plot 2: predictions with contour
            fig, ax = plt.subplots()
            ax.axis(lim)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.contourf(x1, x2, prediction, alpha=0.2)
            ax.contour(x1, x2, prediction, colors='k')
            scatter = ax.scatter(x=dataset.x_train[:, 0], y=dataset.x_train[:, 1], c=predicted_train)
            ax.scatter(x=dataset.x_test[:, 0], y=dataset.x_test[:, 1], c=predicted_test)
            ax.legend(*scatter.legend_elements(), loc='best', title='Classes')
            fig.tight_layout()
            fig.savefig(f"output/scatter_predictions_with_contour_3_learning_rate_{int(learning_rate*10000000)}_sample_{sample}.png", dpi=300)
            plt.close()
            plt.clf()
        costs.append(model.fit(return_costs=True)[-1])
    print("]")
    plt.clf()
    plt.close()
    plt.ylabel("Loss function")
    plt.xlabel("Learning Rate")
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(learning_rates, costs)
    plt.savefig("output/learning_rates.png")
    plt.clf()
    plt.close()
def test_measures(predicted, y):
    TP0 = 0
    TP1 = 0
    FN0 = 0
    FN1 = 0
    TN0 = 0
    TN1 = 0
    FP0 = 0
    FP1 = 0
    for index in range(len(predicted)):
        if predicted[index] == 0:
            if y[index] == predicted[index]:
                TP0 += 1
                TN1 += 1
            else:
                FP0 += 1
                FN1 += 1
        else:
            if y[index] == predicted[index]:
                TP1 += 1
                TN0 += 1
            else:
                FP1 += 1
                FN0 += 1
    
    try: 
        precision0 = TP0/(TP0 + FP0) 
    except:
        precision0 = 0
    try: 
        precision1 = TP1/(TP1 + FP1) 
    except:
        precision1 = 0

    try: 
        recall0 = TP0/(TP0 + FN0) 
    except:
        recall0 = 0
    try: 
        recall1 = TP1/(TP1 + FN1) 
    except: 
        recall1 = 0
    accuracy0 = (TP0 + TN0)/len(y)
    accuracy1 = (TP1 + TN1)/len(y)

    return precision0, precision1, recall0, recall1, accuracy0, accuracy1