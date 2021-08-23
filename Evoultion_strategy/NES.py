# -*- coding:utf-8 -*-
# @Time : 2021/8/24 12:23 上午
# @Author : huichuan LI
# @File : NES.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

DNA_SIZE = 2  # parameter (solution) number
N_POP = 20  # population size
N_GENERATION = 100  # training step
LR = 0.02  # learning rate


def get_fitness(pred): return -((pred[:, 0]) ** 2 + pred[:, 1] ** 2)


mean = tf.Variable(tf.random.normal([2, ], 13., 1.), dtype=tf.float32, trainable=True, name='mean')
cov = tf.Variable(5. * tf.eye(DNA_SIZE), dtype=tf.float32, trainable=True, name='cov')

mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)

tfkids = tf.keras.Input(shape=[N_POP, DNA_SIZE])

tfkids_fit = tf.keras.Input(shape=[N_POP, ])


def FocalLoss():
    def custom_loss(tfkids, tfkids_fit):
        loss = -tf.reduce_mean(mvn.log_prob(tfkids) * tfkids_fit)
        return loss

    return custom_loss


optimizer = tf.keras.optimizers.SGD(LR)

n = 300
x = np.linspace(-20, 20, n)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for i in range(n):
    for j in range(n):
        Z[i, j] = get_fitness(np.array([[x[i], x[j]]]))
plt.contourf(X, Y, -Z, 100, cmap=plt.cm.rainbow);
plt.ylim(-20, 20);
plt.xlim(-20, 20);
plt.ion()

loss_object = FocalLoss()

# training
for g in range(N_GENERATION):
    kids = mvn.sample(N_POP)
    kids_fit = get_fitness(kids)
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(mvn.log_prob(tfkids) * tfkids_fit)

        gradients = tape.gradient(loss, [mean, cov])
        print(mean, cov, loss)
        optimizer.apply_gradients(zip(gradients, [mean, cov]))
    # plotting update
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(kids[:, 0], kids[:, 1], s=30, c='k');
    plt.pause(0.01)

print('Finished');
plt.ioff();
plt.show()
# training
