import numpy
import math
import keras
from keras.datasets import imdb
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


seed = 2
numpy.random.seed(seed)

def sigmoid(x):
    return 1/(1+ numpy.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def classif(x):
    return 1 * (x > 0.5)

def relu(x):
    return x * (x > 0)

def relu_derivative(x):
    return 1. * (x > 0)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

#phase d'embedding, on obtient un unique vecteur par critique
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.compile('rmsprop','mse')
X_train2=model.predict(X_train)
X_test2=model.predict(X_test)

print(X_train2.shape)

yy=y_train.reshape(y_train.shape[0],1)
yyy=y_test.reshape(y_train.shape[0],1)


#100 neurones dans la couche cachée
w1 = numpy.random.rand(16000,100)
w2 = numpy.random.rand(100,1)
dw1 = numpy.random.rand(16000,100)
dw2 = numpy.random.rand(100,1)
l2 = 0

x=X_train2
y=yy
#nombre d'epochs
for k in range(50):



            #feedforward pour les couches l1 et l2
            l1 = sigmoid(x.dot(w1))
            l2 = sigmoid(l1.dot(w2))


            #backprop
            dw2 = l1.T.dot(2*(y - l2) * sigmoid_derivative(l2))
            dw1 = x.T.dot(numpy.dot(2*(y - l2) * sigmoid_derivative(l2), w2.T) * sigmoid(l1))


            #vitesse de descente
            w1 += 0.5*dw1
            w2 += 0.5*dw2
            
            
            
