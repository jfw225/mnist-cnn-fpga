from conv_network import *
import tkinter as tk
import numpy as np
import cv2

from PIL import ImageTk, Image, ImageDraw
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import random

from test_modelv3 import export_sim
import subprocess


def dec2bin(num, integer=10, k_prec=10):
    negative = True if num < 0 else False
    num = num * -1 if negative else num
    binary = ""
    Integral = int(num)
    fractional = num - Integral
    while (Integral):
        rem = Integral % 2
        binary += str(rem)
        Integral //= 2
    binary = binary[:: -1]
    binary += '_'
    while (k_prec):
        fractional *= 2
        fract_bit = int(fractional)
        if (fract_bit == 1):
            fractional -= fract_bit
            binary += '1'
        else:
            binary += '0'
        k_prec -= 1
    binary = '0' * (integer - len(binary.split('_')[0])) + binary

    def twos_complement(binary):
        def invert(x): return '1' if x == '0' else '0' if x == '1' else '_'
        first_one = False
        result = ''
        for i in range(len(binary)-1, -1, -1):
            if first_one:
                result += invert(binary[i])
            else:
                result += binary[i]
            if binary[i] == '1' and first_one == False:
                first_one = True
        return result[::-1]
    binary = twos_complement(binary) if negative else binary
    return binary
# model=Sequential()
#
# model.add(Flatten(input_shape=(28,28)))
#
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(10,activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.load_weights('FFNN-MNIST.h5')

# model = Sequential([
# Flatten(input_shape=(28,28)),
# Dense(784,activation='relu'),
# Dense(10,activation='softmax')
# ])
# model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
# model.load_weights('weights.h5')

# model = Sequential([
# Conv2D(8, 2, activation='sigmoid',padding='valid', input_shape=(28, 28, 1)),
# MaxPooling2D((2, 2)),
# Flatten(),
# Dense(10,activation='softmax')
# ])
# model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
# model.load_weights('weights.h5')


# weights = np.load('weights.npz')
# w1, w2 = weights[weights.files[0]], weights[weights.files[1]]
# prediction = lambda x: np.argmax( softmax(sigmoid(x.dot(w1)).dot(w2)) )


def event_function(event):

    x = event.x
    y = event.y

    SIZE = 15
    x1 = x-SIZE
    y1 = y-SIZE

    x2 = x+SIZE
    y2 = y+SIZE

    canvas.create_oval((x1, y1, x2, y2), fill='black')
    img_draw.ellipse((x1, y1, x2, y2), fill='white')


def save():

    global count

    img_array = np.array(img)
    img_array = cv2.resize(img_array, (28, 28))

    cv2.imwrite(str(count)+'.jpg', img_array)
    # np.savez('arr',img_array)

    count = count+1


def clear():

    global img, img_draw

    canvas.delete('all')
    img = Image.new('RGB', (500, 500), (0, 0, 0))
    img_draw = ImageDraw.Draw(img)

    label_status.config(text='PREDICTED DIGIT: NONE')


def predict():
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (28, 28)) / 256
    # export_sim(img_array)
    with open("./src/c/test.txt", "w") as f:
        f.write(f"{random.randint(0, 1000)}\n")
        for v in img_array.flatten():
            v = dec2bin(v, 10, 10)
            v = "".join([c for c in f"{v}" if c == "0" or c == "1"])
            f.write(f"{int(v, 2)}\n")

    subprocess.run("scp -r src/c/* root@10.253.17.26:~/final".split())

    # subprocess.run("ssh root@10.253.17.26".split() + ["\"cd final && make\""])

    # ------------- tensorflow ------------------------
    # img_array= img_array/4
    # print(img_array.shape)

    # img_array= img_array.reshape(1,28,28, 1) #img_array.reshape(1,28,28)
    # result=model.predict(img_array)
    # label=np.argmax(result,axis=1)

    # ------------- model from scratch ------------------------
    # print(softmax(sigmoid(img_array.flatten().dot(w1)).dot(w2)) )
    # print(forward(img_array.reshape(1, 28, 28)))
    # label = prediction(img_array.flatten()) #np.argmax(result,axis=1)
    label = np.argmax(forward(img_array.reshape(1, 28, 28)))

    # print(img_array.flatten())
    # plt.imshow(img_array)
    # plt.show()
    # np.savez('arr',img_array)
    label_status.config(text='PREDICTED DIGIT:'+str(label))


def func(e):
    global img, img_draw, photo
    num = random.choice(['zeros', 'ones', 'twos', 'threes',
                        'fours', 'fives', 'sixs', 'sevens', 'eights', 'nines'])
    img = Image.open(
        f'./data/{num}/{num[:-1]}{random.randint(0,9)}.jpg').resize((500, 500))
    img_draw = ImageDraw.Draw(img)
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=photo, anchor='nw')
# me = np.load('arr.npz')
# img_array = me[me.files[0]]
# n = np.concatenate([np.concatenate([[x]*4 for x in y]*4) for y in n])
# fig, ax = plt.subplots(2)
# ax[0].imshow(img_array.reshape(28,28))
# ax[1].imshow(n.reshape(28,28))


count = 0

win = tk.Tk()

canvas = tk.Canvas(win, width=500, height=500, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

button_save = tk.Button(win, text='SAVE', bg='green',
                        fg='white', font='Helvetica 20 bold', command=save)
button_save.grid(row=1, column=0)

button_predict = tk.Button(win, text='PREDICT', bg='blue',
                           fg='white', font='Helvetica 20 bold', command=predict)
button_predict.grid(row=1, column=1)

button_clear = tk.Button(win, text='CLEAR', bg='yellow',
                         fg='white', font='Helvetica 20 bold', command=clear)
button_clear.grid(row=1, column=2)

button_exit = tk.Button(win, text='EXIT', bg='red', fg='white',
                        font='Helvetica 20 bold', command=win.destroy)
button_exit.grid(row=1, column=3)

label_status = tk.Label(win, text='PREDICTED DIGIT: NONE',
                        bg='white', font='Helvetica 24 bold')
label_status.grid(row=2, column=0, columnspan=4)


win.bind('<Left>', func)
win.bind('<Right>', func)

canvas.bind('<B1-Motion>', event_function)
img = Image.new('RGB', (500, 500), (0, 0, 0))
img_draw = ImageDraw.Draw(img)

win.mainloop()
