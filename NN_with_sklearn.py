from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import numpy as numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelBinarizer

def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def classif(x):
    return 1 * (x > 0.5)

def relu(x):
    return x * (x > 0)

def relu_derivative(x):
    return 1. * (x > 0)

def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')

    return text

df = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3) #https://www.kaggle.com/c/word2vec-nlp-tutorial/data


df['cleaned_review'] = df['review'].apply(clean_review)


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)
vectorizer = CountVectorizer(lowercase=True, binary=True)
vectorizer.fit(X_train)

X_train2=vectorizer.transform(X_train)
X_test2=vectorizer.transform(X_test)

y_train=list(y_train.values)
y_test=list(y_test.values)



x=X_train2
n1,n2=x.shape
binarizer_ = LabelBinarizer().fit(y_train)
y = binarizer_.transform(y_train)
yy = binarizer_.transform(y_test)

#100 neurones dans la couche cachée, n2 features
w1 = numpy.random.randn(n2,100)
w2 = numpy.random.randn(100,1)
dw1 = numpy.random.randn(n2,100)
dw2 = numpy.random.randn(100,1)
l2 = 0


#nombre epochs
for k in range(200):



            #feedforward pour les couches l1 et l2 (relu l1 et sigmoid l2)
            l1 = relu(x.dot(w1))
            l2 = sigmoid(l1.dot(w2))


            #backprop
            dw2 = l1.T.dot(2*(y - l2) * sigmoid_derivative(l2))
            dw1 = x.T.dot(numpy.dot(2*(y - l2) * sigmoid_derivative(l2), w2.T) * relu_derivative(l1))

            #vitesse de descente
            w1 += 0.005*dw1
            w2 += 0.005*dw2


#validation du modèle
l1 = sigmoid(X_test2.dot(w1))
l2 = sigmoid(l1.dot(w2))
error = abs(yy-classif(l2))
print(numpy.mean(error))
