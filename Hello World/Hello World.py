import tensorflow as tf

session = tf.Session()

hello = tf.constant('Hello World!')
print(session.run(hello))

a = tf.constant(20)
b = tf.constant(30)
print(session.run(a + b))