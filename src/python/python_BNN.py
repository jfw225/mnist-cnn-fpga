#ECE 5760 final backprop producer

import numpy as np
import skimage.measure
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_img = train_X
test_img = test_X
train_lbl = train_y

X_trn = train_img.reshape((60000, 28, 28, 1))
X_tst = test_img.reshape((10000, 28, 28, 1))

X_trn_bin = np.where(X_trn > 127, 1, -1)
X_tst_bin = np.where(X_tst > 127, 1, -1)



#conv0_w = np.zeros((3, 3, 32))
#conv1_w = np.zeros((3,3,64))
#conv2_w = np.zeros((3, 3, 128))
#fc1_w = np.zeros((8, 8, 128))
#fc2_w = np.zeros((10, 128))



def convolution_layer(a, w):
    a_len, a_width, a_depth = a.shape
    w_len, w_width, w_depth = w.shape
    #print(str(a[:,:,0]))
    #print(str(w[:,:,0]))
    a_padded = np.full((a_len+2, a_width+2, a_depth), -1)
    a_padded[1:a_len+1, 1:a_width+1, :] = a
    bin_w = np.where(w > 0, 1, -1)
    out = np.zeros((a_len, a_width, w_depth))
    for k in range(w_depth):
        for i in range(w_len):
            for j in range(w_width):
                out[:,:,k] += bin_w[i,j,k]*a_padded[i:(a_len+i), j:(a_width+j), k%a_depth]
    return out

def batch_norm(a):
    a_len, a_width, a_depth = a.shape
    #out = np.zeros(int(a_len), int(a_width), int(a_depth))
    batch_mean = np.sum(a)/a.size
    norm_a = a-batch_mean
    return np.where(norm_a>0, 1, -1)

def batch_norm_max(a, norm_map):
    a_len, a_width, a_depth = a.shape
    out = np.zeros((int(a_len/2), int(a_width/2), int(a_depth)))
    batch_mean = np.sum(a)/a.size
    norm_a = a-batch_mean
    for k in range(a_depth):
        for i in range(0, a_len, 2):
            for j in range(0, a_width, 2):
                max_val = 0
                act_val = 0
                max_ind = (0,0)
                for i_x in range(2):
                    for j_y in range(2):
                        if(abs(norm_a[i+i_x, j+j_y, k]) > max_val):
                            max_val = abs(norm_a[i+i_x, j+j_y, k])
                            act_val = norm_a[i+i_x, j+j_y, k]
                            max_ind = (i+i_x, j+j_y)
                max_i, max_j = max_ind
                norm_map[max_i, max_j, k] = 1
                out[int(i/2),int(j/2),k] = act_val
    return np.where(out>0, 1, -1)

def fc1_layer(a, w):
    a_len, a_width, a_depth = a.shape
    w_len, w_width, w_depth = w.shape
    product= np.multiply(a, w)
    sum_a0 = np.sum(product, axis = 0)
    sum_a1 = np.sum(sum_a0, axis = 0)
    return sum_a1

def fc2_layer(a, w):
    a_len = a.shape
    out = np.zeros((10))
    for i in range(10):
        out[i] = np.sum(a[i]*w[i, :])
    return out

def softmax(a):
    return np.divide(np.exp(a),np.sum(np.exp(a)))

def cross_entropy_loss(labels,scores):
    #labels - true values (0 or 1) of data
    #scores - outputs of the NN after the softmax layer
    return -(labels*np.log(scores) + (1-labels)*np.log(1-scores))

def fc2_backprop(a, w, labels, loss_out):
    global lamda
    global fc2_w_g
    loss_grad = np.subtract(loss_out,labels)
    #print(str(loss_grad))
    #out = np.multiply(np.sum(w, axis=1), loss_out)
    out = np.sum(np.multiply(w, loss_out[:, None]), axis = 0)
    for i in range(10):
        #print(str( w[i]+(lamda * loss_out[i] * a)))
        w[i] = w[i]+(lamda * loss_out[i] * a)
        #fc2_w_g[i] += (lamda * loss_out[i] * a)
    return out

def fc1_backprop(a, w, q):
    global lamda
    global fc1_w_g
    #fc1_w_g += lamda * q[None, None, :] * a
    w = w + lamda * q[None, None, :] * a
    return w * q[None, None, :]

def norm_backprop(q, norm_map):
    q = np.repeat(q, 2, axis=0)
    q = np.repeat(q, 2, axis=1)
    return np.multiply(q, norm_map)


def conv_backprop(a, w, q):
    global conv0_w_g
    global conv1_w_g
    global conv2_w_g
    a_len, a_width, a_depth = a.shape
    w_len, w_width, w_depth = w.shape
    q_len, q_width, q_depth = q.shape
    a_padded = np.full((a_len+2, a_width+2, a_depth), 0)
    a_padded[1:a_len+1, 1:a_width+1, :] = a
    q_padded = np.zeros((q_len+2, q_width+2, q_depth))
    q_padded[1:q_len+1, 1:q_width+1, :] = q
    a_dub = np.zeros((a_len+2, a_width+2, a_depth*2))
    a_dub[:,:,:a_depth] = a_padded
    a_dub[:,:,a_depth:] = a_padded
    for i in range(w_len):
        for j in range(w_width):
            mul = np.multiply(a_dub[i:(a_len+i), j:(a_width+j)], q_padded[i:(a_len+i), j:(a_width+j)])
            sum_0 = np.sum(mul, axis=0)
            sum_1 = np.sum(sum_0, axis=0)
            w[i, j, :] += lamda * sum_1*.0001
    np.clip(w, -10, 10, out = w)
    out = np.zeros((a_len, a_width, a_depth*2))
    for i in range(w_len):
        for j in range(w_width):
            #out += a_dub[(2-i):(a_len+(2-i)), (2-j):(a_len+(2-j))] * w[i, j]
            out += q_padded[(2-i):(a_len+(2-i)), (2-j):(a_len+(2-j))] * w[i, j]
            #out += w[i,j]
    #out = np.multiply(out, q)
    out = out[:,:,:a_depth] + out[:, :, a_depth:]
    return out

#Hyperparameters
lamda = 10
batch_size = 100

conv0_w = np.random.normal(0, .001, (3, 3, 32))
conv1_w = np.random.normal(0, .001,(3,3,64))
conv2_w = np.random.normal(0, .001, (3, 3, 128))
fc1_w = np.random.normal(0, .001, (8, 8, 128))
fc2_w = np.random.normal(0, .001, (10, 128))
conv0_w_g = np.zeros((3, 3, 32))
conv1_w_g = np.zeros((3,3,64))
conv2_w_g = np.zeros((3, 3, 128))
fc1_w_g = np.zeros((8, 8, 128))
fc2_w_g = np.zeros((10, 128))

for i in range(1000):
    labels = np.zeros((10))
    labels[int(train_lbl[i])] = 1

    norm_map0 = np.zeros((32, 32, 32))
    norm_map1 = np.zeros((16, 16, 64))

    image = np.zeros((32, 32, 16))
    image[2:30, 2:30] = X_trn_bin[i] # copy (28, 28, 1) image to all 16 layers

    #Forward Propogation
    #OUTPUT SHAPES
    #conv: (a_len, a_width, w_depth)
    #batch_norm_max: ( (a_len/2), (a_width/2), (a_depth) )
    #batch_norm: ( (a_len), (a_width), (a_depth) )
    #fc1_layer:(a_depth)
    #fc2_layer: 

    conv0_out = convolution_layer(image, conv0_w) # (32, 32, 32) = (32, 32, 16), (3, 3, 32)
    batch0_out = batch_norm_max(conv0_out, norm_map0) # (16, 16, 32) = (32, 32, 32) , (32, 32, 32)
    conv1_out = convolution_layer(batch0_out, conv1_w) # (16, 16, 64) = (16, 16, 32) , (3,3,64)
    batch1_out = batch_norm_max(conv1_out, norm_map1) # (8, 8, 64) = (16, 16, 64) , (16, 16, 64)
    conv2_out = convolution_layer(batch1_out, conv2_w) # (8, 8, 128) = (8, 8, 64) , (3, 3, 128)
    batch2_out = batch_norm(conv2_out) # (8, 8, 128) = (8, 8, 128)
    fc1_out = fc1_layer(batch2_out, fc1_w) # (128,) = (8, 8, 128) , (8, 8, 128)
    fc2_out = fc2_layer(fc1_out, fc2_w) # (10,) = (128,) , (10, 128)  
    soft_out = softmax(fc2_out) # (10,) =  (10,)
    loss_out = cross_entropy_loss(labels, soft_out) #(10,) = (10,), (10,)
    print(loss_out)
    #print("Test")
    #print(str(fc2_w[0,0]))
    #print(str(soft_out))
    #print(str(loss_out))

    #Back propogation - updates weights by reference
    fc2_bp = fc2_backprop(fc1_out, fc2_w, labels, loss_out)# (128,) = (128,) (10, 128)
    fc1_bp = fc1_backprop(batch2_out, fc1_w, fc2_bp) # (8, 8, 128) = (8, 8, 128), (8, 8, 128), (128,)


    #conv2_bp = conv_backprop(batch1_out, conv2_w, fc1_bp)
    #print(str(conv2_bp))
    #norm1_bp = norm_backprop(conv2_bp, norm_map1)
    #print(str(norm1_bp))
    #conv1_bp = conv_backprop(batch0_out, conv1_w, norm1_bp)
    #norm0_bp = norm_backprop(conv1_bp, norm_map0)
    #conv0_bp = conv_backprop(image, conv0_w, norm0_bp)

    if(i % batch_size == 0):
        lamda = lamda * .9
        print(str(i))

print("Finished!")
