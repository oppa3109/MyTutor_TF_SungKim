import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print hello
print sess.run(hello)
