import sys,os
sys.path.insert(0,'..')
"""Learned ADMM L1 method
Reference : Learned Primal-Dual Reconstruction (Jonas Adler,Ozan Öktem)
https://github.com/adler-j/learned_primal_dual

Run this script learn the neural networks and to generate a folder which saves
the trained parameters.
Run evaluate_learned_admml1.py to evaluate the trained parameters."""

import os
import adler
adler.util.gpu.setup_one_gpu()

from adler.odl.phantom import random_phantom
from adler.tensorflow import prelu, cosine_decay

import tensorflow as tf

import numpy as np
import odl
import odl.contrib.tensorflow
from odl.operator import Operator, OpDomainError, oputils, default_ops
import odl.contrib.solvers.spdhg as spdhg

import scipy

np.random.seed(0)
name = os.path.splitext(os.path.basename(__file__))[0]

sess = tf.InteractiveSession()

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

print("now geometry start")
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=60)
operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

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


with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'): 

        x = tf.zeros_like(x_true) 
        m = tf.zeros_like(y_rt)
        z = m

        sigma = tf.Variable(0.5,dtype="float32",name="sigma",constraint=lambda t: tf.clip_by_value(t, 0, 1))
        tau = tf.Variable(0.5,dtype="float32",name="tau",constraint=lambda t: tf.clip_by_value(t, 0, 1))
        weights = tf.zeros_like(x_true)
        gamma = tf.Variable(0.99,dtype="float32",name="gamma",constraint=lambda t: tf.clip_by_value(t, 0, 1))


    for i in range(n_iter):
            #convolution layer   
        with tf.variable_scope('admm_iterations_{}'.format(i),reuse=tf.AUTO_REUSE):
            #first proximal layer
            update2 = x-odl_op_layer_adjoint(m/tau + odl_op_layer(x) - z)*tau
            update2 = prelu(apply_conv(update2), name='prelu_1')
            update2 = prelu(apply_conv(update2), name='prelu_2')
            update2 = apply_conv(update2, filters=1)
            x = update2

            #second proximal layer

            update = tf.concat([sigma*(odl_op_layer(x)+m/sigma),y_rt],axis=-1)
            update = prelu(apply_conv(update), name='prelu_3')
            update = prelu(apply_conv(update), name='prelu_4')
            update = apply_conv(update, filters=1)
            z = update

            #dual update 

            m = m + gamma*(odl_op_layer(x)-z)
    x_result = x 




with tf.name_scope('loss'):
    loss = tf.linalg.norm(x_result - x_true,1)
    squared_error = x_result - x_true ** 2
    mean_error = tf.reduce_mean(squared_error)
    psnr = -10 * tf.log(mean_error) / tf.log(10.0)

with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 10001
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print("update")
    print(sess.run(update_ops))
    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta2=0.99)

        tvars = tf.trainable_variables()
        print(f"tvars:{tvars}")
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        print(f"grads:{grads}")

        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


# Summary

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.image('x_result', x_result)
    tf.summary.image('x_true', x_true)

    merged_summary = tf.summary.merge_all()
    test_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/test',
                                                sess.graph)
    train_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/train')

# Initialize all TF variables
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./learned_admml1', sess.graph)


# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
y_arr_validate, x_true_arr_validate = generate_data(validation=True)

if 0:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Train the network
for i in range(0, maximum_steps):
    print("STEP:")
    print(i, "started")

    #if i%10 == 0:
    y_arr, x_true_arr = generate_data()


    _, merged_summary_result_train, global_step_result = sess.run([optimizer, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr,
                                         y_rt: y_arr,
                                         is_training: True})

    if i>0 and i%10 == 0: # loss, does not update the variables (it only calculates the loss)
        loss_result, merged_summary_result, global_step_result = sess.run([loss, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr_validate,
                                         y_rt: y_arr_validate,
                                         is_training: False})


        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        test_summary_writer.add_summary(merged_summary_result, global_step_result)

        print('iter={}, loss={}'.format(global_step_result, loss_result))

    if i>0 and i%100 == 0:
        saver.save(sess, adler.tensorflow.util.default_checkpoint_path(name))



