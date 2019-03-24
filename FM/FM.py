import numpy as np
import tensorflow as tf


x_data = np.matrix([
#    Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
#   A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
])
# ratings
y_data = np.array([5, 3, 1, 4, 5, 1, 5])
# Let's add an axis to make tensoflow happy.
y_data.shape += (1, )

n, p = x_data.shape
# number of latent factors
k = 5
# design matrix
X = tf.placeholder('float32', [n, p])
# target vector
y = tf.placeholder('float32', [n, 1])
# bias and weights
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))
 # interaction factors, randomly initialized
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))
# estimate of y, initialized to 0.
y_hat = tf.Variable(tf.zeros([n, 1]))
linear_terms = tf.add(w0,
                        tf.reduce_sum(
                            tf.multiply(W, X), 1, keepdims=True))
interactions = (tf.multiply(0.5,
                tf.reduce_sum(
                    tf.subtract(
                        tf.pow(tf.matmul(X, tf.transpose(V)), 2),
                        tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                    1, keepdims=True)))
y_hat = tf.add(linear_terms, interactions)

# L2 regularized sum of squares loss function over W and V
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

l2_norm = (tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w, tf.pow(W, 2)),
                tf.multiply(lambda_v, tf.pow(V, 2)))))

error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
loss = tf.add(error, l2_norm)

eta = tf.constant(0.1)
optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

# that's a lot of iterations
N_EPOCHS = 1000
# Launch the graph.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(N_EPOCHS):
        # indices = np.arange(n)
        # np.random.shuffle(indices)
        # x_data, y_data = x_data[indices], y_data[indices]
        sess.run(optimizer, feed_dict={X: x_data, y: y_data})

    print('MSE: ', sess.run(error, feed_dict={X: x_data, y: y_data}))
    print('Loss (regularized error):', sess.run(loss, feed_dict={X: x_data, y: y_data}))
    print('Predictions:', sess.run(y_hat, feed_dict={X: x_data, y: y_data}))
    print('Learnt weights:', sess.run(W, feed_dict={X: x_data, y: y_data}))
    print('Learnt factors:', sess.run(V, feed_dict={X: x_data, y: y_data}))