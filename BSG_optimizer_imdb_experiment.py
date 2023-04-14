import tensorflow as tf
print(tf.__version__)

from keras.layers import Dense, Dropout, Flatten, ReLU
from keras import backend as K
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from BSG import BSG

def vectorize(sequences, dimension = 10000):
 results = np.zeros((len(sequences), dimension))
 for i, sequence in enumerate(sequences):
  results[i, sequence] = 1
 return results

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

data = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

def create_logistic_dropout_model():
    model = Sequential([Dropout(0.5, input_shape=(10000,)),
    Dense(1, activation='sigmoid')
])
    return model

def create_logistic_model():
    model = Sequential([ Dense(1, activation='sigmoid', input_shape=(10000,))])
    return model


#For logistic dropout model
model = create_logistic_dropout_model()
#For logistic model
model = create_logistic_model()

model.compile(optimizer = BSG(alpha=80000, type=1), #CHANGE HERE TYPE AND ALPHA
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_x,
          y=train_y,
          epochs=50,
          batch_size = 4096 , #CHANGE HERE BATCH SIZE
          validation_data=(test_x, test_y),
          callbacks=[tensorboard_callback]
          )