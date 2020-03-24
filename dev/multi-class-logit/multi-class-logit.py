import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import data
digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# print(digits.data.shape)
# print(X_train.shape)
# print(len(X_train))
# print(len(X_test))

# Create model
model = LogisticRegression(solver='liblinear')
# print(model)

# Train
model.fit(X_train, y_train)

# Score
score = model.score(X_test, y_test)
print(score)

# Predict
# pred = model.predict([digits.data[67]])
# print(pred)
y_predicted = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# Plot confusion matrix
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
