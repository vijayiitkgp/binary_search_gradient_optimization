import tensorflow as tf
print(tf.__version__)

from keras.datasets import mnist, cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from BSG import BSG

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

def create_model(input_shape = (32,32,3)):
   
    model = Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((3,3), strides=2),
    # Conv2D(64, (5, 5), activation='relu'),
    # MaxPooling2D((3,3), strides=2),
    Conv2D(128, (5, 5), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((3,3), strides=2),
    Flatten(),
    Dense(300),
    BatchNormalization(),
    ReLU() ,
    Dense(10, activation='softmax')
])
    return model

def create_model_new(input_shape = (28,28,1)):
  model = Sequential()
  model.add(Conv2D(16, kernel_size=(3, 3),
                  activation='elu',
                  input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='elu',
                  input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, kernel_size=(3, 3),
                  activation='elu',
                  input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(5, activation='elu'))
  model.add(BatchNormalization())
  # model.add(Dropout(0.5))
  model.add(Dense(10, activation='softmax'))
  return model

model = create_model()


input_shape = (32,32,3)
model = create_model()

model.compile(optimizer = BSG(alpha=10000, type=1),  #CHANGE HERE
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images,
          y=train_labels, 
          epochs=50,
          batch_size = 1024 , #CHANGE HERE
          validation_data=(test_images, test_labels)
          # callbacks=[tensorboard_callback]
          )
