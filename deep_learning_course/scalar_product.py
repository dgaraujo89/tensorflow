
import tensorflow as tf

inputs = tf.constant([[-1.0, 7.0, 5.0]], name = "inputs")
weights = tf.constant([[0.8, 0.1, 0.0]], name = "weights")

multiply = tf.multiply(inputs, weights)

sum = tf.reduce_sum(multiply)

with tf.Session() as sess:
    print(sess.run(inputs))
    print(sess.run(weights))
    print(sess.run(multiply))
    print(sess.run(sum))