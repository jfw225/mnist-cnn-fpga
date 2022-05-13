import numpy as np

x = np.array([[1,0,1], [0,1,0], [0,0,1], [1,0,0]])
y = np.array([[1,0,1,0]]).T
weights = np.random.random((3,1))


sigmoid = lambda x: 1/(1+np.exp(-x))
for epoch in range(10000):
    z = x @ weights # Matrix multiplication is equivalent to taking dot products of each training example with weights
    a = sigmoid(z)
    error = (y - a)
    da_dz = a * (1 - a)
    weights += np.dot(x.T, error*da_dz) #weights -= np.dot(x.T, -error*sigmoidDerivative)

print("Predicting [0,1,1]...")
newZ = np.dot(np.array([0,1,1]), weights)
prediction = sigmoid(newZ)
print(f'The input [0,1,1] is {prediction[0]*100}% likely to be 1')




# np.array(  [ [i,j,k] for i in [0,1] for j in [0,1] for k in [0,1] ] )

"""
weights = np.random.random((3,1))
LEARNING_RATE = 0.01
a = x
for epoch in range(1000):
    #FORWARD
    z = a @ weights #np.dot(x, weights)
    sigmoid = 1/(1+np.exp(-z))
    #BACKWARD
    Cost = 1/len(y) * np.sum( np.square(y - sigmoid) )
    dC_dSigmoid = 2 * (y - sigmoid)
    dSigmoid_dz = sigmoid * (1 - sigmoid)
    dz_dw = a
    dC_dw = dC_dSigmoid * dSigmoid_dz * dz_dw
    weights -=  LEARNING_RATE* dC_dw #np.dot(x.T, error*sigmoidDerivative) #w = w - LEARNING_RATE* dC/dw , -> dC/dw = dC/dSigmoid * dSigmoid/dz * dz/dweights
    a = sigmoid
"""
