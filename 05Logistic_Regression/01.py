import tensorflow as tf
import numpy as np

# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
"""
script_dir = os.path.dirname(os.path.abspath(__file__))
train_data = np.loadtxt(script_dir + '/train.txt', unpack=True, dtype='float32')
"""
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# Our hypothesis                        H(X) = 1 / (1 + exp(-W^T * X))
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# Cost function                         cost(W) = ...
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))


# Minimize                              W := W - a * gradient(cost)
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initialize
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Logistic regression
for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

###########################
# Inference
###########################
print '-----------------------------'
# study_hour attendance
print sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5
print sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5

print sess.run(hypothesis, feed_dict={X:[[1, 1], [4, 3], [3, 5]]}) > 0.5

