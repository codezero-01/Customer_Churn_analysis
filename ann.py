
##DATA PREPROCESSING 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data (which is independent variables for this set)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
x[:, 1] = labelencoder_x1.fit_transform(x[:, 1])
labelencoder_x2= LabelEncoder()
x[:, 2] = labelencoder_x2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1]) #creating the dummy variables
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]        #removing dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


## BUILDING ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initializing ANN

classifier = Sequential()

#first input layer and hidden layer for ann
classifier.add(Dense(activation= 'relu', input_dim =11, units = 6, kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))   #for overfitting

#adding second hidden layer
classifier.add(Dense(activation='relu', units= 6, kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1)) #for overfitting

#adding the output layer
classifier.add(Dense(units = 1 , activation='sigmoid', kernel_initializer='uniform'))

#compiling the ann
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fiting the ann to training set
classifier.fit(x_train, y_train, batch_size=10 , epochs=100)


# making predictions and evaluating ann
# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred >0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""data visualisations"""


#training set visualisations
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#test set visualisations
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

""" test case to check whether the customer will leave thebank or not:"""

custom_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 300, 1, 39, 2, 453400, 3, 1, 1, 3000]])))
custom_pred = (custom_pred > 0.5)

#evaluating the ann
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation= 'relu', input_dim =11, units = 6, kernel_initializer='uniform'))
    classifier.add(Dense(activation= 'relu', units= 6, kernel_initializer='uniform'))
    classifier.add(Dense(units = 1 , activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier 
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch=100)
accuracies = cross_val_score(estimator = classifier , X = x_train , y = y_train , cv = 10 , n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

