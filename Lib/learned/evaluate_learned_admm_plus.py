import sys,os
sys.path.insert(0,'..')

"""  Learned ADMM Plus 
Reference : Learned Primal-Dual Reconstruction (Jonas Adler,Ozan Öktem)
https://github.com/adler-j/learned_primal_dual

Run this script to evauluate parameters trained in learned_admm_plus.py.

 """

import adler
adler.util.gpu.setup_one_gpu()

from adler.odl.phantom import random_phantom
from adler.tensorflow import prelu, cosine_decay

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
import xlsxwriter


np.random.seed(0)
name = 'learned_admm_plus'

sess = tf.InteractiveSession()

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=60)
operator = odl.tomo.RayTransform(space, geometry)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint,
                                                                  'RayTransformAdjoint')

# User selected paramters
n_data = 5
n_iter = 10
memory = 5
# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)


def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    y_arr = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = random_phantom(space)
        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return y_arr, x_true_arr


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    y_rt = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y_rt")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

x_results=[]

with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'): #using this convention only for tensorboard


        x = tf.concat([tf.zeros_like(x_true)] * memory, axis=-1) #concat for reshaping the tensor
        m = tf.concat([tf.zeros_like(y_rt)] * memory, axis=-1)
        z = m
        #shape of primal will be determined through x_true value
        #z = x_true
        #lagrange = x_true
        sigma = tf.Variable(0.5,dtype="float32",name="sigma",constraint=lambda t: tf.clip_by_value(t, 0, 1))
        tau = tf.Variable(0.5,dtype="float32",name="tau",constraint=lambda t: tf.clip_by_value(t, 0, 1))
        gamma = tf.Variable(0.99,dtype="float32",name="gamma",constraint=lambda t: tf.clip_by_value(t, 0, 1))


    for i in range(n_iter):
            #convolution layer   
        with tf.variable_scope('admm_iterations_{}'.format(i)):
            #reconstruction layer
            update2 = tf.concat([x,odl_op_layer_adjoint(m[..., 0:1]/tau + odl_op_layer(x[..., 0:1]) - z[..., 0:1])],axis=-1)
            update2 = prelu(apply_conv(update2), name='prelu_1') 
            update2 = prelu(apply_conv(update2), name='prelu_2')
            update2 = apply_conv(update2, filters=memory)
            #z=x
            x = x + update2
            #c = apply_conv(x, filters=1)
            #evalop =  odl_op_layer(x)
            update = tf.concat([odl_op_layer(x[...,0:1]),m,y_rt],axis=-1)
            update = prelu(apply_conv(update), name='prelu_3') 
            update = prelu(apply_conv(update), name='prelu_4')
            update = apply_conv(update, filters=memory)
            z = z + update
            # L1 prox layer 

            # lagrange multiplier layer
            m = m + gamma*(odl_op_layer(x[..., 0:1])-z[..., 0:1])
        x_results.append(x[..., 0:1])
    x_result = x 



# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

if 1:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Generate validation data
y_arr_validate, x_true_arr_validate = generate_data(validation=True)


x_values_results,z_values_results,m_values_results = sess.run([x_results],
                    feed_dict={x_true: x_true_arr_validate,
                                y_rt: y_arr_validate,
                                is_training: False})

import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

#Print results in terms of SSIM and PSNR
print(ssim(x_values_results[-1][0, ..., 0], x_true_arr_validate[0, ..., 0]))
print(psnr(x_values_results[-1][0, ..., 0], x_true_arr_validate[0, ..., 0], data_range=1))


path = name
#Save image for each iteration of ADMM
for i in range(n_iter):

    space.element(x_values_results[i][0, ..., 0]).show(clim=[0, 1], saveto='{}/x_{}'.format(path, i))

    plt.close('all')