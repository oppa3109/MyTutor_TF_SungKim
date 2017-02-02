import tensorflow as tf
import matplotlib.pyplot as plt

# TF Graph input
X = [1., 2., 3.]
Y = [1., 2., 3.]

m = n_samples = len(X)

# Set model weights
W = tf.placeholder(tf.float32)

# Construct a linear hypothesis: H(x) = Wx
hypothesis = tf.mul(X, W)

# Cost function: cost(W) = 1/m * sum(H(x) - y)^2
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m

# Initializing the variables
init = tf.initialize_all_variables()

# For graphs
W_val = []
cost_val = []

# Launch the graph
sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    print i * 0.1, sess.run(cost, feed_dict={W: i * 0.1})
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

# Graph display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()

'''
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.mul(X, W)

cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m

W_val = []
cost_val = []

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(-30, 50):
    print i * 0.1, sess.run(cost, feed_dict={W: i * 0.1})
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()
'''