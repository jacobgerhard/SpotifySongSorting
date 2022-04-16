
# # MODEL CREATION


# ## Data Preprocessing


# We again begin by importing the necessary libraries required for training and evaluating the TensorFlow deep learning model. We also import the data sets from the `SongCluter.ipynb` file, giving us the full data set of songs and their associated labels.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
np.random.seed(1)


model_data_num = pd.read_csv('/Users/jacobgerhard/Documents/Projects/SpotifySongSorting/Clusters.csv')

model_data_num = model_data_num.iloc[:,1:]
# We split the data into a training and test set, and separate the explanatory variable and response variable.

model_data_num = model_data_num.drop('track_id', axis = 1)
train, test = train_test_split(model_data_num, train_size = 0.75)
xtrain = train.iloc[:,:-1]
ytrain = train.iloc[:,-1].astype(int)

xtest = test.iloc[:,:-1]
ytest = test.iloc[:,-1].astype(int)


# We now dummy encode the response variable, which will allow us to perform multi-class classification within the TensorFlow algorithm.


ydummy = pd.get_dummies(ytrain)
ydummy_test = pd.get_dummies(ytest)


xtrain = xtrain.values
xtest = xtest.values
ydummy = ydummy.values
ydummy_test = ydummy_test.values


# ## Model Training


# We generate the TensorFlow keras model using the following code, with input of 51 attributes and output of 25 attributes, and compile the model. Our loss function is categorical crossentropy since we are working with multi-class classification.


def baseline_model():
    model = tf.keras.Sequential()
    model.add(Dense(20, input_dim=24, activation = 'relu'))
    model.add(Dense(20, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                  metrics = ['accuracy'])
    return model

tfmodel = baseline_model()



# Another method of training the model using KerasClassifier
# Also shows results when testing using the kfold cross validation resampling method,
# which isn't needed since we have an abundance of data

'''
KCmodel = KerasClassifier(build_fn=baseline_model, epochs = 100,
                            batch_size = 500, verbose = 0)

kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)

results = cross_val_score(tfmodel, xtrain, ydummy, cv=kfold)
print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

results_test = cross_val_score(tfmodel, xtest, ydummy_test, cv=kfold)
print("Result: %.2f%% (%.2f%%)" % (results_test.mean()*100, results_test.std()*100))
'''


# We now fit the model to the training data. We choose 5 epochs, meaning we will fit the model by running through the training set 5 times. Since we are already working with a large data set, this will only sacrifice a slight loss in accuracy in exchange for reducing run time.


tfmodel_fit = tfmodel.fit(xtrain, ydummy,epochs= 5, 
                            validation_data=(xtest, ydummy_test))


tfmodel.evaluate(xtest, ydummy_test)


# The accuracy of our training and test data are both well above $95\%$, so we can accept this model.


# Since the TensorFlow algorithm produces the probabilities that a certain song belongs in any of the clusters, we extract the cluster with the highest probability and assign that to `clus_pred` as our predicted cluster. We then output the total number of clusters that were miscalculated in the algorithm.


clus_pred = pd.DataFrame(tfmodel.predict(xtest)).idxmax(axis=1)

clus_real = pd.DataFrame(ydummy_test).idxmax(axis=1)

#sum(clus_real != clus_pred)



# At this point, we've determined our model is accurate and sufficient enough to move onto the next stage: pulling a playlist from a spotify user and assigning which cluster the songs in that playlist belong to.

tfmodel.save('/Users/jacobgerhard/Documents/Projects/SpotifySongSorting/tfmodel')

print('Model has been trained with the training data')
