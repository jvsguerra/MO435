import numpy as np
import pandas as pd
from classifiers import read_spam_data, transform_log, transform_binary, LogisticRegression

train_data, test_data = read_spam_data()
# print(train_data.head())

#### Default setting
logReg = LogisticRegression(regularization = 1)
logReg.fit(train_data.drop('spam', axis = 1), train_data['spam'])
print(logReg.intercept_)
print(logReg.coef_)
print(1-logReg.score(train_data.drop('spam', axis = 1), train_data['spam']))
print(1-logReg.score(test_data.drop('spam', axis = 1), test_data['spam']))

####  Pre-processing
from sklearn import preprocessing

# transform the data
ytrain = train_data['spam']
ytest = test_data['spam']
Xtrain_raw = train_data.drop('spam', axis = 1)
Xtest_raw = test_data.drop('spam', axis = 1)
Xtrain_standard = preprocessing.scale(Xtrain_raw, axis=0)
Xtest_standard = preprocessing.scale(Xtest_raw, axis=0)
Xtrain_log = np.apply_along_axis(transform_log, axis = 0, arr=Xtrain_raw)
Xtest_log = np.apply_along_axis(transform_log, axis = 0, arr=Xtest_raw)
Xtrain_binary = np.apply_along_axis(transform_binary, axis = 0, arr=Xtrain_raw)
Xtest_binary = np.apply_along_axis(transform_binary, axis = 0, arr=Xtest_raw)

data_transform = ['Raw', 'Standard', 'Log', 'Binary']
Xtrain = [Xtrain_raw, Xtrain_standard, Xtrain_log, Xtrain_binary]
Xtest = [Xtest_raw, Xtest_standard, Xtest_log, Xtest_binary]

## now run lots of models to find regularization parameter
regularization = np.linspace(0, 20, num=41)
misclassification_rates = pd.DataFrame(dtype=np.float64,
                                       index = np.arange(len(regularization)),
                                       columns = ['Regularization'] + 
                                       list(map(lambda x : x + ' Train',  data_transform)) + 
                                       list(map(lambda x : x + ' Test',  data_transform)))
for i in range(len(regularization)):
    misclassification_rates.iloc[i]['Regularization'] = regularization[i]
    if regularization[i] == 0:
        regularization[i] += 0.01 # hack when there's no convergence.
    logReg = LogisticRegression(regularization = regularization[i])
    for j in range(len(data_transform)):
        logReg.fit(Xtrain[j], ytrain)
        misclassification_rates.iloc[i][data_transform[j] + ' Train'] = 1 - logReg.score(Xtrain[j], ytrain)
        misclassification_rates.iloc[i][data_transform[j] + ' Test'] = 1 - logReg.score(Xtest[j], ytest)
misclassification_rates

# Plot results
import matplotlib.pyplot as plt
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
plt.figure(figsize=(12,8))
for i in range(len(data_transform)):
    plt.plot(misclassification_rates['Regularization'], misclassification_rates[data_transform[i] + ' Train'],
             color=colors[i], linestyle='--', linewidth=2, marker='.')
    plt.plot(misclassification_rates['Regularization'], misclassification_rates[data_transform[i] + ' Test'],
             color=colors[i], linestyle='-', linewidth=2, marker='.')
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5), title="Dataset")
plt.ylabel("Misclassification Rate")
plt.xlabel("Regularization Parameter ($\lambda$)")
plt.title("Misclassification Rates for Various Transforms and Regularizations")
plt.show()

