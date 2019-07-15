
import tensorflow as tf

p1 = tf.placeholder('float', None)

op1 = p1 + 2

with tf.Session() as sess:
    result = sess.run(op1, feed_dict = {p1: [1, 2, 3]})
    print(result)

p2 = tf.placeholder('float', [None, 5])

op2 = p2 * 5

with tf.Session() as sess:
    result = sess.run(op2, feed_dict = {p2: [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]})
    print(result)
