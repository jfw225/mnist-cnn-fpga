from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers as reg
# model=Sequential()
#
# model.add(Flatten(input_shape=(28,28)))
#
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(10,activation='softmax'))

# model = Sequential([
# Flatten(input_shape=(28,28)),
# Dense(784,activation='relu'),
# Dense(10,activation='softmax')
# ])
"""
>> X.shape
(6000, 28, 28, 1)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_2 (Conv2D)            (None, 27, 27, 8)         40
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 8)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1352)              0
_________________________________________________________________
dense_4 (Dense)              (None, 10)                13530
=================================================================
Total params: 13,570
Trainable params: 13,570
Non-trainable params: 0
_________________________________________________________________

>>> model.weights[0].numpy().shape
(2, 2, 1, 8)
>>> model.weights[1].numpy().shape
(8,)
>>> model.weights[2].numpy().shape
(1352, 10)
>>> model.weights[3].numpy().shape
(10,)

"""
# model = Sequential([
# Conv2D(filters=8, kernel_size=2, activation='sigmoid',padding='valid',
# kernel_regularizer=reg.l2(0.005), input_shape=(28, 28, 1), ),
# MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
# Dropout(0.2),
# Flatten(),
# Dense(10,activation='softmax')
# ])

model = Sequential([
Conv2D(filters=8, kernel_size=2, activation='relu',padding='valid',
 input_shape=(28, 28, 1), kernel_regularizer=reg.l2(0.005)),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(filters=8, kernel_size=2, activation='relu',padding='valid'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Dropout(0.2),
Flatten(),
Dense(10,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
model.summary()
# model.load_weights('weights.h5') #for training again once already trained

def load_data():
    X = []
    for n in ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixs', 'sevens', 'eights', 'nines']:
        x = cv2.imread(f'data/{n}/{n[:-1]}0.jpg')
        x =cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).reshape(1,28,28)
        for i in range(1,600):
            new_x = cv2.imread(f'data/{n}/{n[:-1]}{i}.jpg')
            new_x = cv2.cvtColor(new_x ,cv2.COLOR_BGR2GRAY).reshape(1,28,28)
            x = np.concatenate((x, new_x), axis=0)
        X.append(x)
    X = np.array(X).reshape(-1, 28, 28)
    Y = np.hstack( [np.repeat(i, 600) for i in range(10) ] )
    return X, Y

X, Y = load_data()
X = X.reshape(6000, 28, 28, 1)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

(train_data,train_target),(test_data,test_target)= (X_train, Y_train) , (X_val, Y_val) # mnist.load_data()
train_data , test_data = train_data.reshape(-1, 28, 28, 1), test_data.reshape(-1, 28, 28, 1)
new_train_target=np_utils.to_categorical(train_target)
new_test_target=np_utils.to_categorical(test_target)
new_train_data=train_data/256
new_test_data=test_data/256

model.fit(new_train_data,new_train_target,epochs=1000, batch_size=128)

model.save_weights('weights.h5')
