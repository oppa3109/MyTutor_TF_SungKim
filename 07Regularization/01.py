# https://gitub.com/aymericdamien/TensorFlow-Examples
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # MNIST data image of shape 28 x 28 = 784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition -> 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Computer average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y:batch_ys})
        avg_cost /= total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print "Epoch: ", '%04d' % (epoch+1), "cost = ", "{:.9f}".format(avg_cost)

    print "Optimzation Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy: ", accuracy.eval({x: mnist.test.images, y:mnist.test.labels})

    # Get one and predict
    r = np.random.randint(0, mnist.test.num_examples-1)
    print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
    print "Prediction: ", sess.run(tf.argmax(hypothesis, 1), {x: mnist.test.images[r:r+1]})

    # Show the image
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

'''
#--- OK in python not OK in docker
import matplotlib.pyplot as plt
import numpy as np
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C,S = np.cos(X), np.sin(X)
plt.plot(X,C)
plt.show()
'''