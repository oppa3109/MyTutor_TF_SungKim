import tensorflow as tf

X = [1., 2., 3.]
Y = [1., 2., 3.]

#W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
# Set wrong model weights
W = tf.Variable(5.)

hypothesis = W * X

gradient = tf.reduce_mean((hypothesis - Y) * X) * 2
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.01

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# Get gradients
gvs = optimizer.compute_gradients(cost)
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in xrange(100):
    print step, sess.run([gradient, W, gvs])
    sess.run(apply_gradients)
