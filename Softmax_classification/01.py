import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder("float", [None, 3]) # 1(for bias), x1, x2
Y = tf.placeholder("float", [None, 3]) # A, B, C --> 3 classes

W = tf.Variable(tf.zeros([3, 3]))

# Hypothesis: Softmax H(x) = exp(Wx) / sum(exp(Wx))
hypothesis = tf.nn.softmax(tf.matmul(X, W))
# hypothesis = tf.nn.softmax(tf.matmul(W, X))

# Cost: cross-entropy cost = -sum(Y * logH(x))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

# Minimize
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
"""
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
"""

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

# inference
    print('---------------------')
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print a, sess.run(tf.arg_max(a, 1))

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print b, sess.run(tf.arg_max(b, 1))

    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print c, sess.run(tf.arg_max(c, 1))

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print all, sess.run(tf.arg_max(all, 1))

