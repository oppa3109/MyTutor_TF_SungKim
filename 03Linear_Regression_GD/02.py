import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.1

# Minimize
# W := W - a/m * sum{(H(xi) - yi) * xi}
#1 descent = W - tf.mul(0.1, tf.reduce_min(tf.mul((hypothesis - Y), X)))

#2 descent = W - learning_rate * tf.reduce_mean((hypothesis - Y) * X)
#2 update = W.assign(descent)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in xrange(100):
#2    sess.run(update, feed_dict={X:x_data, Y:y_data})
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

