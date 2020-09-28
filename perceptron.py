import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(12)
num_observations = 500

x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

X1 = np.vstack((x1, x2)).astype(np.float32)
Y1 = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
Y = Y1.reshape((len(Y1), 1))
insertion = np.ones((X1.shape[0], 1))
X = np.hstack((insertion, X1))
w = np.zeros((3, 1))

def perceptron(z):
    if z > 0:
        return 1
    else:
        return 0

def calc_error(X, Y, w):
    result = np.dot(X, w)
    error = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        error[i] = abs(Y[i] - perceptron(result[i])) 
    return np.sum(error) / error.shape[0]

def show_clf(X, Y, w):
    # plot the weighted data points
    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.4)

    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
    xy = np.c_[xx.ravel(), yy.ravel()]

    Z = np.dot(np.hstack((np.ones((xy.shape[0], 1)),  xy)), w)
    
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='b')

    plt.show()


def iteration(gamma, X, Y, w, lr=0.1):
    i = 0
    while(calc_error(X, Y, w) > gamma):
        z = np.dot(X[i], w)
        pred = perceptron(z)
        #print(X[i].reshape(3,1))
        
        w += lr * (Y[i] - pred) * X[i].reshape(3,1)
    
        if i == 999:
            print("error = ",calc_error(X, Y, w))
            i = 0
        else: 
            i += 1
    print("final error = ", calc_error(X, Y, w))


#print(calc_error(X, Y, [[0.1],[0.1],[0.1]]))
iteration(0.01, X, Y, w)
show_clf(X1, Y1, w)