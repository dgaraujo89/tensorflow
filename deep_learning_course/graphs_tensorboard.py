
import tensorflow as tf

tf.reset_default_graph()

a = tf.add(2, 2, name = 'add')
b = tf.multiply(a, 3, name = 'mult1')
c = tf.multiply(b , a, name = 'mult2')

with tf.name_scope('Operation'):
    d = tf.add(a, 5)
    e = tf.multiply(d, 3)
    
    with tf.name_scope('Final_Operation'):
        f = tf.subtract(e, 10)
        
        with tf.Session() as sess:
            print(sess.run(f))

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    print(sess.run(c))
    writer.close()