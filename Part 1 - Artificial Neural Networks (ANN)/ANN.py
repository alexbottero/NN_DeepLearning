
"""
Created on Thu Jun  7 11:08:32 2018

@author: alexandre
"""
#Artificial Neural Network

#Installing Theano (numerical calculation module based on Numpy (CPU and GPU) )
 
#Installing Tensorflow (Google numerical calculation module)

#Installing Keras (Create Neural Network)


#Part 1: Preparing data

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the DataSet
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Part 2: Building neural network

#Import modules Keras
import keras
from keras.models import Sequential
from keras.layers import Dense


#Init NN
classifier=Sequential()

#Add headen layer 
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform",input_dim=11))

#Add a second headen layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

#Add output layer
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

#Compilation 
classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

#Training 
classifier.fit(X_train,y_train, batch_size=10,epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#prediction for 1 client
""" Country : France
	Credit Score :600
	Genre : Male
	Old : 40
	Time in this bank: 3 years
	Balance:600000€
	number of product : 2
	credit card: yes
	actif Member:yes
	Estimated avergae : 50 000€
	"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))	
new_prediction = (new_prediction > 0.5)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score 

def build_classifier():
	classifier=Sequential()
	classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform",input_dim=11))
	classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
	classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
	classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])
	return classifier

classifier2 = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
precision =cross_val_score(classifier2, X=X_train,Y=y_train,cv=10,n_jobs=-1)
moyenne = precision.mean()
ecart_type =precision.std()


