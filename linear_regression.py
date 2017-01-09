'''
Linear regression learning algorithm using TensorFlow.
Author: Wu Tenghu
Git: https://github.com/wutenghu
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.05
training_epochs = 500
display_step = 100
batch_size = 100

# Training Data
noise = np.random.uniform(-0.05,0.05,[100]).astype(np.float32)
train_X = np.random.uniform(-10.0,10.0,[100]).astype(np.float32)
train_Y = train_X*0.1 + 0.3 + noise

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Set model weights
# W = tf.Variable(tf.random_uniform([1],-1.0,1.0), name="weight")
W = tf.Variable(tf.zeros([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_mean(tf.square(pred-Y))

# Gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
# init = tf.initialize_all_variables() # old version, will get expired soon
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
	total_batch = int(len(train_X)/batch_size)
	for i in range(total_batch):
            batch_X = train_X[i*batch_size:(i+1)*batch_size-1]
            batch_Y = train_Y[i*batch_size:(i+1)*batch_size-1]
            sess.run(optimizer, feed_dict={X: batch_X, Y: batch_Y})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
