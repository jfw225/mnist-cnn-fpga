import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from matplotlib import pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# (X, Y), (X_test, Y_test) = mnist.load_data() # X:(60000, 28, 28), Y: (60000,)


def load_data():
    X = []
    for n in ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixs', 'sevens', 'eights', 'nines']:
        x = cv2.imread(f'data/{n}/{n[:-1]}0.jpg')
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).reshape(1, 28, 28)
        for i in range(1, 600):
            new_x = cv2.imread(f'data/{n}/{n[:-1]}{i}.jpg')
            new_x = cv2.cvtColor(new_x, cv2.COLOR_BGR2GRAY).reshape(1, 28, 28)
            x = np.concatenate((x, new_x), axis=0)
        X.append(x)
    X = np.array(X).reshape(-1, 28, 28)
    Y = np.hstack([np.repeat(i, 600) for i in range(10)])
    return X, Y


# X, Y = load_data()

# Do this by convention, but specify random state for debugging
# X_train, X_val, Y_train, Y_val = train_test_split(
#     X, Y, test_size=0.15, random_state=42)

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
#Sigmoid and derivative
def sigmoid(x): return 1/(np.exp(-x)+1)


def d_sigmoid(x): return sigmoid(x) * (1 - sigmoid(x))

# relu


def relu(x): return np.where(x >= 0, x, 0)

#Softmax and derivative


def softmax(x):
    exp_element = np.exp(x)
    return exp_element / np.sum(exp_element)


def d_softmax(x):
    return softmax(x) * (1 - softmax(x))




# model = Sequential([
#     Conv2D(8, 2, activation='sigmoid',
#            padding='valid', input_shape=(28, 28, 1)),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(10, activation='softmax')
# ])
# model.compile(loss='categorical_crossentropy',
#               optimizer='SGD', metrics=['accuracy'])
# model.load_weights('weights_our_data_96.h5')
#
# w1 = model.weights[0].numpy().reshape(2, 2, 8)
# b1 = model.weights[1].numpy()
# w3 = model.weights[2].numpy()
# b3 = model.weights[3].numpy()


def convolve(img, kernel: np.array) -> np.array:
    tgt_size = img.shape[1]-1  # we're using
    # To simplify things
    k = kernel.shape[0]
    img_len, img_width, img_depth = img.shape  # (128, 28, 28)
    kernel_len, kernel_width, kernel_depth = kernel.shape  # (2, 2, 8)
    # 2D array of zeros
    convolved_img = np.zeros(
        shape=(img_len, tgt_size, tgt_size, kernel_depth))  # (128, 27, 27, 8)
    for t in range(img_len):
        for i in range(tgt_size):  # rows of image
            for j in range(tgt_size):  # columns of image
                # img[i, j] = individual pixel value
                # Get the current matrix
                mat = img[t, i:i+k, j:j +
                          k].reshape(kernel_len, kernel_width, 1)  # (2, 2, 1)
                # Apply the convolution - element-wise multiplication and summation of the result
                # Store the result to i-th row and j-th column of our convolved_img array
                convolved_img[t, i, j] = np.sum(mat * kernel, axis=(0, 1))
    return convolved_img
    # need to activation function and max batch


def maxpool(image, f=2, s=2):
    x, h_prev, w_prev, n_c = image.shape
    x, img_len, img_width, n_filters = image.shape
    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1
    # create a matrix to hold the values of the maxpooling operation.
    # downsampled = np.zeros((n_c, h, w))
    # (128, 13, 13, 8)
    output = np.zeros((x, img_len//2, img_width//2, n_filters))
    # slide the window over every part of the image using stride s. Take the maximum value at each step.
    for t in range(x):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + f <= w_prev:
                # choose the maximum value within the window at each step and store it to the output matrix
                # downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                output[t, out_y, out_x] = np.max(
                    image[t, curr_y:curr_y+f, curr_x:curr_x+f], axis=(0, 1))
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return output


# def forward(x):
#     Z1 = convolve(x, w1)  # (128, 27, 27, 8) = (128, 28, 28) , (2, 2, 8)
#     print("MAX and MIN", np.max(Z1 + b1), np.min(Z1 + b1))
#     A1 = (Z1 + b1)  # (128, 27, 27, 8) = (128, 27, 27, 8)
#     Z2 = maxpool(A1)  # (128, 13, 13, 8) = (128, 27, 27, 8)
#     Z3 = Z2.reshape(Z2.shape[0], -1) @ w3  # (128, 10) = (128, 1352) (1352, 10)
#     print(Z3 + b3)
#     out = Z3 + b3 #no softmax
#     return out



model = Sequential([
Conv2D(filters=8, kernel_size=2, activation='relu',padding='valid',
 input_shape=(28, 28, 1) ),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(filters=8, kernel_size=2, activation='relu',padding='valid'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(10,activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer='SGD', metrics=['accuracy'])
model.load_weights('weights.h5')
w1, b1, w3, b3, w5, b5 = [ model.weights[i].numpy() for i in range(6) ]
w1 =w1.reshape(2,2,8)
def forward(x):
    global w1, b1, w3, b3, w5, b5
    x, w1, b1, w3, b3, w5, b5 = [i.astype(np.float16) for i in (x, w1, b1, w3, b3, w5, b5) ]
    Z1 = convolve(x, w1).astype(np.float16)  # (1, 27, 27, 8) = (1, 28, 28) , (2, 2, 8)
    A1 = (Z1 + b1).astype(np.float16)  # (1, 27, 27, 8) = (1, 27, 27, 8)
    print('Layer 1: ', np.max(Z1+b1), np.min(Z1+b1), np.max(Z1), np.min(Z1))
    Z2 = maxpool(A1).astype(np.float16)  # (1, 13, 13, 8) = (128, 27, 27, 8)
    print('Layer 2: ', np.max(Z2), np.min(Z2))
    arr = [convolve(Z2[:, :, :, i], w3[:, :, :, i] ) for i in range(8)]
    # Z3 = convolve(Z2, w3)  # (1, 12, 12, 8) =  (1, 13, 13, 8), (2, 2, 8, 8)
    Z3 = arr[0]
    for i in range(1,8): Z3 += arr[i].astype(np.float16)
    A3 = (Z3 + b3).astype(np.float16) # (1, 12, 12, 8) = (1, 12, 12, 8)
    print('Layer 3: ', np.max(Z3), np.min(Z3), np.max(Z3 + b3), np.min(Z3 + b3))
    Z4 = maxpool(A3).astype(np.float16)  # (1, 6, 6, 8) = (1, 12, 12, 8)
    print('Layer 4: ', np.max(Z4), np.min(Z4))
    Z5 = Z4.reshape(Z4.shape[0], -1).astype(np.float16) @ w5.astype(np.float16)  # (1, 10) = (1, 288) (288, 10)
    out = (Z5 + b5).astype(np.float16) #softmax(Z5 + b5)
    print('Layer 5: ', np.max(Z5), np.min(Z5), np.max(out), np.min(out))
    return out
# w3_alt = np.swapaxes(w3, 2,3)
# for i in range(8):
#     print(np.all(w3[:, :, i, :] == w3_alt[:, :, :, i]))
# model.set_weights([w1, b1, w3_alt, b3, w5, b5])
# new_model.predict( image.reshape(1, 28, 28, 1) )
# XX = model.input
# YY = model.layers[0].output
# new_model = Model(XX, YY)
# new_model2 = Sequential([
# model.layers[0],
# model.layers[1] ])
# new_model2.predict(new_x)
# new_model3 = Sequential([
# model.layers[0],
# model.layers[1],
# model.layers[2] ] )
# new_model3.predict(image.reshape(1, 28, 28, 1))


if __name__ == '__main__':
    # training
    epochs = 50000  # best 100000 #second best was 12000
    lr = 0.0001  # best 0.0001 #second best was 0.001
    batch = 128

    # this is basically the same as forward, except that it
    def inference(x): return softmax(sigmoid(x.dot(w1)).dot(w2))
    # only returns 10 element list of probabilities
    losses, accuries, val_accuracies = [], [], []

    for i in range(epochs):
        # randomize and create batches
        sample = np.random.randint(
            0, X_train.shape[0], size=(batch))  # random indices
        x = X_train[sample].reshape((-1, 28, 28))
        y = Y_train[sample]

        forward_values = forward(x, y)
        out, update_w1, update_w2 = backward(x, y, forward_values)
        category = np.argmax(out, axis=1)

        accuracy = (category == y).mean()
        accuries.append(accuracy.item())

        loss = ((category-y)**2).mean()
        losses.append(loss.item())

        # Stochastic gradient descent
        w1 = w1 - lr*update_w1
        w2 = w2 - lr*update_w2

        # testing our model using the validation set every 20 epochs
        if(i % 20 == 0):
            X_val = X_val.reshape((-1, 28*28))
            val_out = np.argmax(inference(X_val), axis=1)
            val_acc = (val_out == Y_val).mean()
            val_accuracies.append(val_acc.item())
        if(i % 1000 == 0):
            print(
                f'For {i}th epoch: train accuracy: {accuracy:.3f}| validation accuracy:{val_acc:.3f}')
            print(np.max(x.dot(w1)), np.min(x.dot(w1)))

    X_test = X.reshape(-1, 28*28)  # X_test.reshape(-1, 28*28)
    Y_test = Y

    test_out = np.argmax(inference(X_test), axis=1)
    test_acc = (test_out == Y_test).mean().item()
    print(f'Test accuracy = {test_acc:.4f}')
    np.savez('weights', w1, w2)
"""
weights = np.load('weights.npz')
w1, w2 = weights[weights.files[0]], weights[weights.files[1]]

fig, axes = plt.subplots(2)
axes[0].imshow(X_test[11].reshape(28, 28) )
axes[1].imshow(X_test[7].reshape(28, 28) )

np.argmax( inference(X_test[11]) )




"""
