
import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[-1, 3], [4 , 2]])

mult = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(mult))

mult2 = tf.matmul(b, a)

with tf.Session() as sess:
    print(sess.run(mult2))


c = tf.constant([[2, 3], [0, 1], [-1, 4]])
d = tf.constant([[1, 2, 3], [-2, 0, 4]])

mult3 = tf.matmul(c, d)

with tf.Session() as sess:
    print(sess.run(mult3))