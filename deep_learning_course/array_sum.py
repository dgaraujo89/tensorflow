
import tensorflow as tf

a = tf.constant([9, 8, 7], name = 'a')
b = tf.constant([1, 2, 3], name = 'b')

sum = a + b

with tf.Session() as sess:
    print(sess.run(sum))


a1 = tf.constant([[1, 2, 3], [4, 5, 6]], name = 'a1')
b1 = tf.constant([[1, 2, 3], [4, 5, 6]], name = 'b1')

sum = tf.add(a1, b1)

with tf.Session() as sess:
    print(sess.run(sum))


a2 = tf.constant([[1],[2]])

sum = tf.add(a1, a2)

with tf.Session() as sess:
    print(sess.run(sum))