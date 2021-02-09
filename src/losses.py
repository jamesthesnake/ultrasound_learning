# File:       nn_bmode/src/losses.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2019-08-30
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
import math
import numpy as np


# Helper function to convert input to decibels
def _dB(x):
    return 20 / math.log(10) * tf.math.log(x)


def _tf_reshape_a_like_b(a, b):
    # For some reason, tf.keras gives unknown shape to one of the images
    sz = b.get_shape().as_list()
    for i in range(len(sz)):
        if sz[i] is None:
            sz[i] = -1
    return tf.reshape(a, sz)


############################################
# Functions related to regularization losses
############################################
def get_reg_losses():
    # Get all convolution kernel values
    kernel_list = [v for v in tf.trainable_variables() if "kernel" in v.name]
    l1_loss, l2_loss = 0, 0
    for v in kernel_list:
        l1_loss += tf.reduce_mean(tf.abs(v))
        l2_loss += tf.reduce_mean(tf.square(v))
    l2_loss = tf.sqrt(l2_loss)
    return l1_loss, l2_loss


##########################################
# Functions to compute the optimal weights
##########################################
# Helper function to compute dot product of a batch
def _compute_batch_dot(a, b):
    sz = a.get_shape().as_list()
    out = tf.multiply(a, b)
    out = tf.reshape(out, [-1, sz[1] * sz[2] * sz[3]])
    out = tf.reduce_sum(out, axis=1)
    out = tf.reshape(out, [-1, 1, 1, 1])
    return out


# L2 optimal weight
def compute_l2_wopt(y, yhat):
    return tf.divide(_compute_batch_dot(yhat, y), _compute_batch_dot(yhat, yhat))


# L1 optimal weight
def compute_l1_wopt(y, yhat):
    # First get the quotient between y and yhat
    sz = tf.shape(y)
    q = tf.reshape(tf.divide(y, yhat), [-1, sz[1] * sz[2] * sz[3]])
    # Compute the optimal weight for l1 loss (median value)
    midx = sz[1] * sz[2] // 2
    return tf.reshape(tf.nn.top_k(q, midx).values[:, midx - 1], [-1, 1, 1, 1])


# L2 optimal weight (decibel scale)
def compute_l2_wopt_dB(y, yhat):
    return _dB(_compute_batch_dot(yhat, y)) - _dB(_compute_batch_dot(yhat, yhat))


# L1 optimal weight (decibel scale)
def compute_l1_wopt_dB(y, yhat):
    return _dB(compute_l1_wopt(y, yhat))


##############################
# Functions related to L2 loss
##############################
def get_l2loss(y, yhat):
    return tf.sqrt(tf.reduce_mean(tf.square(y - yhat)))


def get_l2loss_wopt(y, yhat):
    wopt = compute_l2_wopt(y, yhat)
    return get_l2loss(y, tf.multiply(yhat, wopt))


def get_l2loss_log(y, yhat):
    return get_l2loss(_dB(y), _dB(yhat))


def get_l2loss_wopt_log(y, yhat):
    wopt_dB = compute_l2_wopt_dB(y, yhat)
    return get_l2loss(_dB(y), _dB(yhat) + wopt_dB)


##############################
# Functions related to L1 loss
##############################
def get_l1loss(y, yhat):
    return tf.reduce_mean(tf.abs(y - yhat))


def get_l1loss_wopt(y, yhat):
    wopt = compute_l1_wopt(y, yhat)
    return get_l1loss(y, tf.multiply(yhat, wopt))


def get_l1loss_log(y, yhat):
    return get_l1loss(_dB(y), _dB(yhat))


def get_l1loss_wopt_log(y, yhat):
    wopt_dB = compute_l1_wopt_dB(y, yhat)
    return get_l1loss(_dB(y), _dB(yhat) + wopt_dB)


###################################
# Functions related to MS-SSIM loss
###################################
_MSSSIM_SCALE_FACTORS = [0.3, 0.4, 0.3]


def get_msssim(y, yhat):
    y = tf.transpose(y, [0, 2, 3, 1])
    yhat = tf.transpose(yhat, [0, 2, 3, 1])
    return tf.image.ssim_multiscale(y, yhat, 4, _MSSSIM_SCALE_FACTORS)


def get_msssim_wopt(y, yhat):
    y = tf.transpose(y, [0, 2, 3, 1])
    yhat = tf.transpose(yhat, [0, 2, 3, 1])
    wopt = compute_l2_wopt(y, yhat)
    return tf.image.ssim_multiscale(y, yhat * wopt, 4, _MSSSIM_SCALE_FACTORS)


def get_msssim_log(y, yhat):
    y = tf.transpose(y, [0, 2, 3, 1])
    yhat = tf.transpose(yhat, [0, 2, 3, 1])
    return tf.image.ssim_multiscale(_dB(y), _dB(yhat), 100, _MSSSIM_SCALE_FACTORS)


def get_msssim_wopt_log(y, yhat):
    return tf_ms_ssim_nolum(_dB(y), _dB(yhat), level=3)


# Gaussian filter for MS-SSIM
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[
        -size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1
    ]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


# Write custom no-luminance ssim for faster computation
def tf_ssim_nolum(img1, img2, mean_metric=True, size=11, sigma=1.5):
    # img1 = _tf_reshape_a_like_b(a=img1, b=img2)
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    K2 = 0.03
    L = 40  # 1  # depth of image (255 in case the image has a differnt scale)
    C2 = (K2 * L) ** 2
    convargs = {"strides": [1, 1, 1, 1], "padding": "VALID", "data_format": "NCHW"}
    mu1 = tf.nn.conv2d(img1, window, **convargs)
    mu2 = tf.nn.conv2d(img2, window, **convargs)
    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 = tf.nn.conv2d(img1 * img1, window, **convargs) - mu11
    sigma22 = tf.nn.conv2d(img2 * img2, window, **convargs) - mu22
    sigma12 = tf.nn.conv2d(img1 * img2, window, **convargs) - mu12

    # Compute contrast times structure (no luminance)
    value = (2 * sigma12 + C2) / (sigma11 + sigma22 + C2)

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


# Write custom no-luminance ms-ssim for faster computation
def tf_ms_ssim_nolum(img1, img2, mean_metric=True, level=3):
    # img1 = _tf_reshape_a_like_b(a=img1, b=img2)
    weight = tf.constant([0.3, 0.4, 0.3], dtype=tf.float32)
    mcs = []
    for _ in range(level):
        cs_map = tf_ssim_nolum(img1, img2, mean_metric=False)
        mcs.append(tf.reduce_mean(cs_map))
        poolargs = {"padding": "same", "data_format": "channels_first"}
        img1 = tf.keras.layers.AveragePooling2D(2, 2, **poolargs)(img1)
        img2 = tf.keras.layers.AveragePooling2D(2, 2, **poolargs)(img2)

    # list to tensor of dim D+1
    mcs = tf.stack(mcs, axis=0)

    # Multiply together
    value = tf.reduce_prod(mcs[:-1] ** weight[:-1])

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

