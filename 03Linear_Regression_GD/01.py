import tensorflow as tf
import matplotlib.pyplot as plt

# TF Graph input
X = [1., 2., 3.]
Y = [1., 2., 3.]

# Set model weights
W = tf.placeholder(tf.float32)

# Construct a linear hypothesis: H(x) = Wx
hypothesis = X * W

# Cost function: cost(W) = 1/m * sum(H(x) - y)^2
#m = n_samples = len(X)
#cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Initializing the variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# For graphs
W_val = []
cost_val = []

# Launch the graph
sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    print feed_W, curr_cost
    W_val.append(curr_W)
    cost_val.append(curr_cost)

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