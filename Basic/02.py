import tensorflow as tf

# Everything is operation!!
sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a + b

# Print out operation
print c

print sess.run(c)