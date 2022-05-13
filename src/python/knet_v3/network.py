from verilog.core.vsyntax import V_FixedPoint
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


def load_data():
    X = []
    for n in ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixs', 'sevens', 'eights', 'nines']:
        x = cv2.imread(f'./python_model/data/{n}/{n[:-1]}0.jpg')
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).reshape(1, 28, 28)
        for i in range(1, 600):
            new_x = cv2.imread(f'./python_model/data/{n}/{n[:-1]}{i}.jpg')
            new_x = cv2.cvtColor(new_x, cv2.COLOR_BGR2GRAY).reshape(1, 28, 28)
            x = np.concatenate((x, new_x), axis=0)
        X.append(x)
    X = np.array(X).reshape(-1, 28, 28)
    Y = np.hstack([np.repeat(i, 600) for i in range(10)])
    return X, Y


def get_weights():
    model = Sequential([
        Conv2D(filters=8, kernel_size=2, activation='sigmoid', padding='valid',
               input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=8, kernel_size=2, activation='sigmoid', padding='valid'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    model.load_weights('./python_model/weights.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD', metrics=['accuracy'])
    model.load_weights('./python_model/weights.h5')
    w1, b1, w3, b3, w5, b5 = [model.weights[i].numpy() for i in range(6)]
    w1 = w1.reshape(2, 2, 8)

    return w1, b1, w3, b3, w5, b5


def sigmoid(x):
    return 1/(np.exp(-x)+1)


def softmax(x):
    exp_element = np.exp(x)
    return exp_element / np.sum(exp_element)


def convolve(img, kernel: np.array) -> np.array:
    tgt_size = 27  # we're using
    # To simplify things
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
                mat = img[t, i:i + kernel_len, j:j +
                          kernel_len].reshape(kernel_len, kernel_width, 1)  # (2, 2, 1)
                # Apply the convolution - element-wise multiplication and summation of the result
                # Store the result to i-th row and j-th column of our convolved_img array
                convolved_img[t, i, j] = np.sum(mat * kernel, axis=(0, 1))

    return convolved_img
    # need to activation function and max batch


def convolve_flat(img, kernel):
    """
    A = [[a, b]
        [c, d]]

    W = [[ W1 = [o1, ..., o8],
           W2 = [p1, ..., p8]],
         [ W3 = [q1, ..., q8],
           W4 = [r1, ..., r8]]]

    prod = A * W
         = [[a * W1,
             b * W2,
             c * W3
             d * W4]]

    prod.sum(axis=(0, 1)) =
        [
            a * o1 + b * p1 + c * q1 + d * r1,
            ...,
            a * o8 + b * p8 + c * q8 + d * r8
        ]
    """

    assert len(img.shape) == 2, img.shape

    tgt_size = img.shape[-1] - 1  # 27

    img_width, img_depth = img.shape
    k_len, k_width, k_depth = kernel.shape

    c_img = np.zeros(shape=(tgt_size, tgt_size, k_depth))  # (27, 27, 8)

    for i in range(tgt_size):  # rows if image
        for j in range(tgt_size):  # columns of image
            sub = img[i:i + k_len, j:j + k_len]

            sub = sub.reshape(*sub.shape, 1)

            c_img[i, j] = np.sum(sub * kernel, axis=(0, 1))

    return c_img


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


def maxpool_flat(image: np.ndarray, f=2, s=2):
    assert image.ndim == 3, image

    img_len, img_width, n_filters = image.shape

    output = np.zeros(shape=(img_len // 2, img_width // 2, n_filters))

    # slide the window over every part of the image using stride s
    # take the maximum value at each step
    curr_y = out_y = 0

    # slide the max pooling window vertically across the image
    while curr_y + f <= img_len:
        curr_x = out_x = 0

        # slide the max pooling window horizontally across the image
        while curr_x + f <= img_width:
            # k = image[curr_y:curr_y + f, curr_x:curr_x + f]
            # k_flat = [V_FixedPoint(k, 44, 44) for k in k.flatten()]
            # print(k, k.shape)
            # print(k.flatten())
            # for i, k in enumerate(k_flat):
            #     k = str(k).replace("88'b", "").replace("_", "")
            #     print(f"Window at {curr_y}, {curr_x}, {i}: {int(k, 2)}")
            # print("===")

            # choose the max value in the window at each step
            # store it to output matrix
            output[out_y, out_x] = np.max(
                image[curr_y:curr_y + f, curr_x:curr_x + f], axis=(0, 1))

            # j = output[out_y, out_x]
            # print(j, j.shape)
            # indd = out_y * (img_width // 2) * n_filters + out_x * n_filters
            # for i, k in enumerate([V_FixedPoint(k, 44, 44) for k in j]):
            #     k = str(k).replace("88'b", "").replace("_", "")
            #     print(f"Window at {indd}: {int(k, 2)}")
            curr_x += s
            out_x += 1
        # exit()

        curr_y += s
        out_y += 1

    print(curr_y, curr_x)

    return output


def forward(x, w1, b1, w3, b3):
    Z1 = convolve(x, w1)  # (128, 27, 27, 8) = (128, 28, 28) , (2, 2, 8)
    print(np.max(Z1 + b1), np.min(Z1 + b1))
    A1 = sigmoid(Z1 + b1)  # (128, 27, 27, 8) = (128, 27, 27, 8)
    Z2 = maxpool(A1)  # (128, 13, 13, 8) = (128, 27, 27, 8)
    Z3 = Z2.reshape(Z2.shape[0], -1) @ w3  # (128, 10) = (128, 1352) (1352, 10)
    out = softmax(Z3 + b3)
    return out
