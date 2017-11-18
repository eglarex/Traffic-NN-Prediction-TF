from numpy import *
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
import os 
import math
import matplotlib.pyplot as plt
#from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

df=pd.read_table("C:\\Users\\Rex Arthur\\test_BdBKkAj.csv",sep="-|[ ]|:|,",engine="python",skiprows=1,header=None)
test_year=df.values[:,0]
test_month=df.values[:,1]
test_date=df.values[:,2]
test_hour=df.values[:,3]
test_junction=df.values[:,6]
df=pd.read_table("C:\\Users\\Rex Arthur\\train_aWnotuB.csv",sep="-|[ ]|:|,",engine="python",skiprows=1,header=None)
train_year=df.values[:,0]
train_month=df.values[:,1]
train_date=df.values[:,2]
train_hour=df.values[:,3]
train_junction=df.values[:,6]
train_vehicles=df.values[:,7]

X_train=np.zeros((48120,5))
#X_train=(train_year,train_month,train_date,train_hour,train_junction)
X_train[:,0]=train_year
X_train[:,1]=train_month
X_train[:,2]=train_date
X_train[:,3]=train_hour
X_train[:,4]=train_junction
X_train=np.transpose(X_train)
print(X_train.shape)

Y_train=np.zeros((train_year.shape[0],180))
for i in range (train_year.shape[0]):
    n=train_vehicles[i]
    Y_train[i,n-1]=1
Y_train=np.transpose(Y_train)
print(Y_train.shape)

X_test=np.zeros((test_year.shape[0],5))
X_test[:,0]=test_year
X_test[:,1]=test_month
X_test[:,2]=test_date
X_test[:,3]=test_hour
X_test[:,4]=test_junction
#X_test=np.transpose(X_test)
print(X_test.shape)

def cost(logits, labels):

    z = tf.placeholder(tf.float64, name = "z")
    y = tf.placeholder(tf.float64, name = "y")
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)
    sess = tf.Session()
    cost = sess.run(cost,feed_dict = {z:logits,y:labels})
    sess.close()
    
    return cost

def create_placeholders(n_x,n_y):
    X=tf.placeholder(tf.float64,shape=(n_x,None),name="X")
    Y=tf.placeholder(tf.float64,shape=(n_y,None),name="Y")
    return X,Y

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1",[1000,5],dtype=float64,initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[1000,1],dtype=float64,initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",[np.max(train_vehicles),1000],dtype=float64,initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2",[180,1],dtype=float64,initializer=tf.zeros_initializer())
    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = tf.add(tf.matmul(W1,X),b1,name="Z1")                                             
    A1 = tf.nn.relu(Z1,name="A1")                                              
    Z2 = tf.add(tf.matmul(W2,A1),b2,name="Z2")                                 

    return Z2                                 
          
def comput_cost(Z2,Y):
    logits=tf.transpose(Z2)
    labels=tf.transpose(Y)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]                  
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model(X_train,Y_train, X_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32,print_cost=True):

    ops.reset_default_graph()
    tf.set_random_seed(1)    
    seed = 3                 
    (n_x, m) = X_train.shape 
    n_y = Y_train.shape[0]   
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z2=forward_propagation(X,parameters)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
