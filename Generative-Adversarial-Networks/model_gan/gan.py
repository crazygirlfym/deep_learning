import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


## training a generative adversarial network to sample from a Gaussian distribution
mu, sigma = -1, 1
xs = np.linspace(-5, 5, 1000)
plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma))
# plt.show()

TRAIN_ITERS = 10000
M= 200

def mlp(input, output_dim):
    w1 = tf.get_variable("w0",[input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
    w2 = tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("b1", [5],initializer=tf.constant_initializer(0.0))

    w3 = tf.get_variable("w2", [5, output_dim], initializer=tf.random_normal_initializer())
    b3 = tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))

    fc1 = tf.nn.tanh(tf.matmul(input, w1) + b1)
    fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
    fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)

    return fc3, [w1, b1, w2, b2, w3, b3]

def momentum_optimizer(loss, var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.001, batch, TRAIN_ITERS//4, 0.95, staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.6).minimize(loss, global_step=batch, var_list=var_list)
    return optimizer

with tf.variable_scope("D_pre"):
    input_node = tf.placeholder(tf.float32, shape=(M,1))
    train_labels = tf.placeholder(tf.float32, shape=(M,1))
    D, theta = mlp(input_node, 1)
    loss = tf.reduce_mean(tf.square(D-train_labels))

optimizer = momentum_optimizer(loss, None)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def plot_d0(D, input_node):
    f, ax = plt.subplots(1)
    xs = np.linspace(-5, 5, 1000)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label = "p_data")
    r = 1000
    xs = np.linspace(-5, 5, r)
    ds = np.zeros((r,1))
    for i in range(r//M):
        x = np.reshape(xs[M*i:M*(i+1)], [M,1])
        ds[M * i:M * (i + 1)] = sess.run(D, feed_dict={input_node: x})
    ax.plot(xs, ds, label="decision boundary")
    ax.set_ylim(0,1.1)
    plt.legend()

plot_d0(D, input_node)
plt.show()

lh = np.zeros(1000)
for i in range(1000):
    d = (np.random.random(M) -0.5) * 10
    labels = norm.pdf(d, loc=mu, scale=sigma)

    lh[i], _ = sess.run([loss, optimizer], {input_node: np.reshape(d, (M,1)), train_labels : np.reshape(labels, (M,1))})

plt.plot(lh)
plt.title('Training Loss')
plot_d0(D, input_node)


weightsD = sess.run(theta)
sess.close()


# build generative adversarial network
with tf.variable_scope("G"):
    z_node = tf.placeholder(tf.float32, shape=(M,1))   # M uniform floats
    G, theta_g = mlp(z_node, 1)
    G = tf.multiply(5.0, G) # scale up by 5 to match range

with tf.variable_scope("D") as scope:
    x_node = tf.placeholder(tf.float32, shape=(M,1))  # input M normally distribution floats
    fc, theta_d = mlp(x_node, 1)
    D1 = tf.maximum((tf.minimum(fc, .99)), 0.01)   # clamp as a probability

    # make a copy of D that uses the same variables, but takes in G as input
    scope.reuse_variables()
    fc, theta_d = mlp(G, 1)
    D2 = tf.maximum((tf.minimum(fc, .99)), 0.01)  # clamp as a probability

obj_d = tf.reduce_mean(tf.log(D1) + tf.log(1-D2))
obj_g = tf.reduce_mean(tf.log(D2))


opt_d = momentum_optimizer(1-obj_d, theta_d)
# maximize log(D(G(a))
opt_g = momentum_optimizer(1-obj_g, theta_g)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# copy weights from pre-training over to new D network
for i, v in enumerate(theta_d):
    sess.run(v.assign(weightsD[i]))

def plot_fig():
    f, ax = plt.subplots(1)

    # p_data
    xs = np.linspace(-5, 5, 1000)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')

    # decison boundary
    r = 5000
    xs = np.linspace(-5, 5, r)
    ds = np.zeros((r,1))  # decision surface

    # process multiple points in parallel in same minibatch
    for i in range(r//M):
        x = np.reshape(xs[M*i:M*(i+1)], (M, 1))
        ds[M*i: M*(i+1)] = sess.run(D1, {x_node: x})

    ax.plot(xs, ds, label='decision boundary')

    # distribution of inverse-mapped points
    zs = np.linspace(-5, 5, r)
    gs = np.zeros((r,1))

    for i in range(r//M):
        z = np.reshape(zs[M*i:M*(i+1)], (M, 1))
        gs[M * i: M * (i + 1)] = sess.run(G, {z_node: z})

    histc, edges = np.histogram(gs, bins=10)
    ax.plot(np.linspace(-5, 5, 10), histc/float(r), label='p_g')

    ax.set_ylim(0, 1.1)
    plt.legend()
    plt.show()

plot_fig()
plt.title('Before Training')

k = 1
histd, histg = np.zeros(TRAIN_ITERS), np.zeros(TRAIN_ITERS)

for i in range(TRAIN_ITERS):
    for j in range(k):
        x = np.random.normal(mu, sigma, M)
        x.sort()
        z = np.linspace(-5.0, 5.0, M) + np.random.random(M) * 0.01
        histd[i], _ = sess.run([obj_d, opt_d], {x_node: np.reshape(x, (M, 1)), z_node:np.reshape(z,(M,1))})
    z = np.linspace(-5, 5, M) + np.random.random(M) * 0.01
    histg[i], _ = sess.run([obj_g, opt_g], {z_node: np.reshape(z, (M,1))})

plt.plot(range(TRAIN_ITERS), histd, label = "obj_d")
plt.plot(range(TRAIN_ITERS), 1- histg, label ="obj_g")
plt.legend()

plot_fig()




















