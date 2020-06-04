from __future__ import print_function

import warnings

from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax

#from sklearn.model_selection import train_test_split
import sklearn.model_selection as model_selection


"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 4
"""
class BLSTM:
    def __init__(self, name=""):
        self.name = name
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(50, 50)))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        # Lower learning rate to prevent divergence
        adamax = Adamax(lr=0.002)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model based on training data
    """
    def train(self, data, batch_size=64, epochs=1):
        
        print("data: ",data)

        vectors = np.stack(data.iloc[:, 0].values)
        print("vectors: ",vectors)

        labels = data.iloc[:, 1].values
        print("labels: ",labels)

        positive_idxs = np.where(labels == 1)[0]
        print(len(positive_idxs))
        #print((positive_idxs))
        negative_idxs = np.where(labels == 0)[0]
        print(len(negative_idxs))

        undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
        print(len(undersampled_negative_idxs))

        resampled_idxs = np.concatenate([positive_idxs, negative_idxs])
        print(len(resampled_idxs))


        #print ("vectors"+vectors)
        
        #print ("labels"+labels)

        #print ("positive_idxs"+positive_idxs)

        #print ("negative_idxs"+negative_idxs)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(vectors[resampled_idxs, ], labels[resampled_idxs],test_size=0.2, stratify=labels[resampled_idxs])
     
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weight)
        self.model.save_weights(self.name + "_model.h5")
        values = self.model.evaluate(X_test, y_test, batch_size=batch_size)
        print("Accuracy is...", values[1])
        predictions = (self.model.predict(X_test, batch_size=batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print('False positive rate is...', fp / (fp + tn))
        print('False negative rate is...', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('True positive rate is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))
		
    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def load(self, model_path):
        self.model.load_weights(model_path)

    def predict(self, data, batch_size=64):
        vectors = np.stack(data.iloc[:, 1].values)
        X = vectors
        predictions = (self.model.predict(X, batch_size=batch_size)).round()
        print(predictions)
        