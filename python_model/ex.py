import tkinter as tk
import numpy as np
import cv2
from PIL import ImageTk,Image,ImageDraw



# X = []
# for n in ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixs', 'sevens', 'eights', 'nines']:
#     x = cv2.imread(f'data/{n}/{n[:-1]}0.jpg')
#     x =cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).reshape(1,28,28)
#     for i in range(1,500):
#         new_x = cv2.imread(f'data/{n}/{n[:-1]}{i}.jpg')
#         new_x = cv2.cvtColor(new_x ,cv2.COLOR_BGR2GRAY).reshape(1,28,28)
#         x = np.concatenate((x, new_x), axis=0)
#     X.append(x)
# X = np.array(X).reshape(-1, 28, 28)

def event_function(event):

    x=event.x
    y=event.y

    SIZE = 15
    x1=x-SIZE
    y1=y-SIZE

    x2=x+SIZE
    y2=y+SIZE

    canvas.create_oval((x1,y1,x2,y2),fill='black')
    img_draw.ellipse((x1,y1,x2,y2),fill='white')

def save():

    global count

    img_array=np.array(img)
    img_array=cv2.resize(img_array,(28,28))

    cv2.imwrite(f'data/twos/two{count}.jpg',img_array)
    print(count)

    count=count+1

def clear():

    global img,img_draw

    canvas.delete('all')
    img=Image.new('RGB',(500,500),(0,0,0))
    img_draw=ImageDraw.Draw(img)



def predict():
    label_status.config(text=f'Count: {count}')

double = 0
def release(event):
    global double
    double = (double + 1 )% 1
    if double == 0 :
        save()
        clear()
        predict()



count=eval( input("Enter count: ") )

win=tk.Tk()

canvas=tk.Canvas(win,width=500,height=500,bg='white')
canvas.grid(row=0,column=0,columnspan=4)

button_save=tk.Button(win,text='SAVE',bg='green',fg='white',font='Helvetica 20 bold',command=save)
button_save.grid(row=1,column=0)

button_predict=tk.Button(win,text='PREDICT',bg='blue',fg='white',font='Helvetica 20 bold',command=predict)
button_predict.grid(row=1,column=1)

button_clear=tk.Button(win,text='CLEAR',bg='yellow',fg='white',font='Helvetica 20 bold',command=clear)
button_clear.grid(row=1,column=2)

button_exit=tk.Button(win,text='EXIT',bg='red',fg='white',font='Helvetica 20 bold',command=win.destroy)
button_exit.grid(row=1,column=3)

label_status=tk.Label(win,text='PREDICTED DIGIT: NONE',bg='white',font='Helvetica 24 bold')
label_status.grid(row=2,column=0,columnspan=4)

canvas.bind('<B1-Motion>',event_function)
canvas.bind('<ButtonRelease-1>', release)
img=Image.new('RGB',(500,500),(0,0,0))
img_draw=ImageDraw.Draw(img)

win.mainloop()
