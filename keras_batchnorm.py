import numpy as np
np.random.seed(123)  # for reproducibility
import keras
'''
Next, import the Sequential model type from Keras.
This is simply a linear stack of neural network layers,
and it's perfect for the type of feed-forward CNN we're building in this tutorial.
'''
from keras.models import Sequential

'''
Next, let's import the "core" layers from Keras.
These are the layers that are used in almost any neural network:
'''
from keras.layers import Dense, Dropout, Activation, Flatten

'''
Then, we'll import the CNN layers from Keras.
These are the convolutional layers that will help us efficiently train on
image data:
'''
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers

# Finally, we'll import some utilities. This will help us transform our data later:
from keras.utils import np_utils

# Load MNIST data:
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
# (60000, 28, 28)
# plotting the first sample:

from matplotlib import pyplot as plt
#plt.imshow(X_train[0])
#plt.show()

'''
When using the Theano backend, you must explicitly declare a dimension for the
depth of the input image. For example, a full-color image with all 3 RGB
channels will have a depth of 3. Our MNIST images only have a depth of 1, but
we must explicitly declare that. In other words, for Theano we want to transform
our dataset from having shape (n, width, height) to (n, depth, width, height).
When using the TensorFlow backend, the number of input channels must be last so
we want to reshape from (n, width, height) to (n, width, height, depth)
Here's how we can do that easily:
'''
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# to confirm the reshape operation:
print(X_train.shape)
# (60000, 28, 28, 1)
# The final preprocessing step for the input data is to convert our data type to float32
# and normalize our data values to the range [0, 1].
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print(Y_train.shape)
# (60000, 10)

'''
Now we're ready to define our model architecture. In actual R&D work, researchers
will spend a considerable amount of time studying model architectures.

To keep this tutorial moving along, we're not going to discuss the theory or
math here. This alone is a rich and meaty field, and we recommend the CS231n
class mentioned earlier for those who want to learn more.

Plus, when you're just starting out, you can just replicate proven architectures
from academic papers or use existing examples. Here's a list of example
implementations in Keras https://github.com/fchollet/keras/tree/master/examples

Let's start by declaring a sequential model format:
'''
model = Sequential()

# L1, the input layer:
#model.add(Convolution2D(32, (4, 4), input_shape=(28,28,1), activation='relu'))
model.add(Convolution2D(32, (6, 6), input_shape=(28,28,1), activation='relu'))
print('shape after 1st layer:', model.output_shape)
#shape after 1st layer: (None, 25, 25, 32)

model.add(BatchNormalization())

# L2, the first hidden layer:
model.add(Convolution2D(64, (3, 3), activation='relu'))
print('shape after 2nd layer:', model.output_shape)
#shape after 2nd layer: (None, 22, 22, 40)
#model.add(BatchNormalization())

# L3, the second hidden layer:
#model.add(Convolution2D(64, (3, 3), activation='relu'))
#print('shape after 3rd layer:', model.output.shape)
#shape after 3rd layer: (?, 20, 20, 64)

#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
#print('final output shape after maxpooling:', model.output_shape)
#final output shape after maxpooling: (None, 10, 10, 64)
'''
Again, we won't go into the theory too much, but it's important to highlight
the Dropout layer we just added.
This is a method for regularizing our model in order to prevent overfitting.
You can read more about it here:
https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning

MaxPooling2D is a way to reduce the number of parameters in our model by
sliding a 2x2 pooling filter across the previous
layer and taking the max of the 4 values in the 2x2 filter.

So far, for model parameters, we've added two Convolution layers.
To complete our model architecture, let's add a fully connected layer and then
the output layer:
'''
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

'''
For Dense layers, the first parameter is the output size of the layer.
Keras automatically handles the connections between layers.

Note that the final layer has an output size of 10, corresponding to the 10
classes of digits. Also note that the weights from the Convolution layers must
be flattened (made 1-dimensional) before passing them to the fully connected
Dense layer. We just need to compile the model and we'll be ready to train it.
When we compile the model, we declare the loss function and the optimizer
(SGD, Adam, etc.).
'''
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)
# default: adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)


model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
              optimizer=adam,
              metrics=['accuracy'])
'''
Keras has a variety of loss functions (https://keras.io/objectives/) and
out-of-the-box # optimizers (https://keras.io/optimizers/) to choose from.
To fit the model, all we have to do is declare the batch size and number of
epochs to train for, then pass in our training data.
'''
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

# We then instantiate the callback like so:
history = AccuracyHistory()
import time
start = time.time()
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[history])
end = time.time()
print('training time (min):', (end - start)/60.)
'''
You can also use a variety of callbacks (https://keras.io/callbacks/) to
set early-stopping rules,
save model weights along the way, or log the history of each training epoch.

Finally, we can evaluate our model on the test data:
'''
score = model.evaluate(X_test, Y_test, verbose=0)
print('Loss:', score[0], 'accuracy:', score[1])

#L1: 32, k=4,4, relu
#BN
#L2: 40, k=4,4, relu
#BN
#L3: 64, k=3,3, relu
# MaxPooling + drop 0.2
#shape after 3rd layer: (?, 4, 4, 64)
#L6: flatten
#BN
#L7: dense 128, relu
# drop 0.5
#L8: dense 10, softmax
#optimizer adam (lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
#Epoch 7/10
#60000/60000 [==============================] - 602s - loss: 0.0477 - acc: 0.9858 - val_loss: 0.0195 - val_acc: 0.9939
#Epoch 8/10
#60000/60000 [==============================] - 614s - loss: 0.0462 - acc: 0.9856 - val_loss: 0.0197 - val_acc: 0.9932
#Epoch 9/10
#60000/60000 [==============================] - 613s - loss: 0.0430 - acc: 0.9869 - val_loss: 0.0215 - val_acc: 0.9935
#Epoch 10/10
#60000/60000 [==============================] - 601s - loss: 0.0408 - acc: 0.9880 - val_loss: 0.0191 - val_acc: 0.9938
#training time (min): 99.33410190343857
#Loss: 0.019136534208 accuracy: 0.9938

#L1: 32, k=4,4, relu
#BN
#L2: 40, k=4,4, relu
# MaxPooling
#L3: flatten
#BN
#L4: dense 128, relu
#BN
#L5: dense 10, softmax
#optimizer adam (lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
#shape after 1st layer: (None, 25, 25, 32)
#shape after 2nd layer: (None, 22, 22, 40)
#final output shape after maxpooling: (None, 11, 11, 40)
#Epoch 8/10
#60000/60000 [==============================] - 332s - loss: 0.0034 - acc: 0.9992 - val_loss: 0.0254 - val_acc: 0.9919
#Epoch 9/10
#60000/60000 [==============================] - 332s - loss: 0.0022 - acc: 0.9995 - val_loss: 0.0221 - val_acc: 0.9938
#Epoch 10/10
#60000/60000 [==============================] - 333s - loss: 0.0017 - acc: 0.9996 - val_loss: 0.0242 - val_acc: 0.9934
#training time (min): 56.68383150498072
#Loss: 0.0242435377537 accuracy: 0.9934


#L1: 32, k=4,4, relu
#BN
#L2: 40, k=5,5, relu
# MaxPooling
#L3: flatten
#BN
#L4: dense 128, relu
#L5: dense 10, softmax
#optimizer adam (lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
#shape after 1st layer: (None, 25, 25, 32)
#shape after 2nd layer: (None, 21, 21, 40)
#final output shape after maxpooling: (None, 10, 10, 40)
#Epoch 8/10
#60000/60000 [==============================] - 403s - loss: 0.0043 - acc: 0.9986 - val_loss: 0.0292 - val_acc: 0.9927
#Epoch 9/10
#60000/60000 [==============================] - 394s - loss: 0.0035 - acc: 0.9989 - val_loss: 0.0234 - val_acc: 0.9929
#Epoch 10/10
#60000/60000 [==============================] - 392s - loss: 0.0025 - acc: 0.9994 - val_loss: 0.0229 - val_acc: 0.9931
#training time (min): 66.84674318234126
#Loss: 0.0228747450394 accuracy: 0.9931

#L1: 32, k=3,3, relu
#BN
#L2: 64, k=3,3, relu
# MaxPooling
#L3: flatten
#BN
#L4: dense 128, relu
#L5: dense 10, softmax
#optimizer adam (lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
#shape after 1st layer: (None, 26, 26, 32)
#shape after 2nd layer: (None, 24, 24, 64)
#final output shape after maxpooling: (None, 12, 12, 64)
#Epoch 7/10
#60000/60000 [==============================] - 391s - loss: 0.0061 - acc: 0.9981 - val_loss: 0.0538 - val_acc: 0.9889
#Epoch 8/10
#60000/60000 [==============================] - 379s - loss: 0.0049 - acc: 0.9985 - val_loss: 0.0646 - val_acc: 0.9879
#Epoch 9/10
##60000/60000 [==============================] - 388s - loss: 0.0043 - acc: 0.9988 - val_loss: 0.0558 - val_acc: 0.9886
#Epoch 10/10
#60000/60000 [==============================] - 412s - loss: 0.0033 - acc: 0.9991 - val_loss: 0.0548 - val_acc: 0.9907
#training time (min): 64.5347384373347
#Loss: 0.0547986747986 accuracy: 0.9907

#L1: 32, k=4,4, relu
#BN
#L2: 64, k=4,4, relu
# MaxPooling
#L3: flatten
#BN
#L4: dense 128, relu
#L5: dense 10, softmax
#optimizer adam (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
#shape after 1st layer: (None, 25, 25, 32)
#shape after 2nd layer: (None, 22, 22, 64)
#final output shape after maxpooling: (None, 11, 11, 64)
#Epoch 8/10
#60000/60000 [==============================] - 444s - loss: 0.0037 - acc: 0.9989 - val_loss: 0.0382 - val_acc: 0.9909
#Epoch 9/10
#60000/60000 [==============================] - 439s - loss: 0.0026 - acc: 0.9993 - val_loss: 0.0352 - val_acc: 0.9926
#Epoch 10/10
#60000/60000 [==============================] - 428s - loss: 0.0022 - acc: 0.9994 - val_loss: 0.0349 - val_acc: 0.9921
#training time (min): 72.39179325898489
#Loss: 0.0348949823188 accuracy: 0.9921

# L1: Conv 32, (8, 8)
# BN
# L2: Conv 64, (4, 4)
# L3: flatten
# BN
# L4: Fully connected 128 (relu)
# L5: Fully connected 10 (softmax)
# shape after 1st layer: (None, 21, 21, 32)
# shape after 2nd layer: (None, 18, 18, 64)
#Epoch 7/10
#60000/60000 [==============================] - 438s - loss: 0.0849 - acc: 0.9925 - val_loss: 0.0908 - val_acc: 0.9923
#Epoch 8/10
#60000/60000 [==============================] - 439s - loss: 0.0665 - acc: 0.9940 - val_loss: 0.1015 - val_acc: 0.9909
#Epoch 9/10
#60000/60000 [==============================] - 433s - loss: 0.0599 - acc: 0.9945 - val_loss: 0.0826 - val_acc: 0.9923
#Epoch 10/10
#60000/60000 [==============================] - 449s - loss: 0.0533 - acc: 0.9952 - val_loss: 0.0946 - val_acc: 0.9917
#training time (min): 73.92254432837169
#Loss: 0.0945816576937 accuracy: 0.9917


# L1: Conv 32, (6, 6)
# BN
# L2: Conv 64, (3, 3)
# L3: flatten
# BN
# L4: Fully connected 128 (relu)
# L5: Fully connected 10 (softmax)
# shape after 1st layer: (None, 23, 23, 32)
# shape after 2nd layer: (None, 21, 21, 64)
#Epoch 8/10
#60000/60000 [==============================] - 492s - loss: 0.1054 - acc: 0.9916 - val_loss: 0.1261 - val_acc: 0.9908
#Epoch 9/10
#60000/60000 [==============================] - 490s - loss: 0.0860 - acc: 0.9932 - val_loss: 0.1146 - val_acc: 0.9911
#Epoch 10/10
#60000/60000 [==============================] - 497s - loss: 0.0784 - acc: 0.9940 - val_loss: 0.1162 - val_acc: 0.9908
#training time (min): 82.55580251216888
#Loss: 0.116169631735 accuracy: 0.9908

# We recommend studying other example models in Keras
# (https://github.com/fchollet/keras/tree/master/examples)
# and Stanford's computer vision class (http://cs231n.stanford.edu/).

plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
print('done!')
