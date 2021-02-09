# File:       nn_bmode/src/utils.py
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
import numpy as np
import h5py
import tensorflow as tf
import math
import losses


def load_from_mat(dataset_path, nimgs=0):

    print("Loading data from %s..." % dataset_path)

    if dataset_path != "":
        if nimgs == 0:
            x = np.array(h5py.File(dataset_path)["img"])
            y = np.array(h5py.File(dataset_path)["ref"])
        else:
            x = np.array(h5py.File(dataset_path)["img"][:nimgs])
            y = np.array(h5py.File(dataset_path)["ref"][:nimgs])

        # Transpose to put channels in dim 1, R/I in dim 2
        x = np.transpose(x, [0, 3, 4, 1, 2])
        y = np.transpose(y, [0, 3, 4, 1, 2])
        szi = x.shape
        szr = y.shape

        # For img, ref, combine the channels and R/I in dim 1
        x = x.reshape([szi[0], szi[1] * szi[2], szi[3], szi[4]])
        y = y.reshape([szr[0], szr[1] * szr[2], szr[3], szr[4]])
        # Get rid of any accidental non-positive values in ground truth
        # e.g., due to spline interpolation of ImageNet images
        y[y <= 0] = 0
        y += 1e-32  # Add eps to avoid NaN (will be added to network too)

        return x, y

    else:
        return None


def load_img_from_mat(dataset_path, nimgs=0):

    print("Loading data from %s..." % dataset_path)

    if dataset_path != "":
        if nimgs == 0:
            x = np.array(h5py.File(dataset_path)["img"])
        else:
            x = np.array(h5py.File(dataset_path)["img"][:nimgs])

        # Transpose to put channels in dim 1, R/I in dim 2
        x = np.transpose(x, [0, 3, 4, 1, 2])
        sz = x.shape

        # For img, ref, combine the channels and R/I in dim 1
        x = x.reshape([sz[0], sz[1] * sz[2], sz[3], sz[4]])

        return x

    else:
        return None


def make_bmode_tf(x):
    sz = x.get_shape().as_list()
    N = sz[1] // 2
    z = tf.complex(x[:, ::2], x[:, 1::2])
    z = tf.reduce_sum(x, axis=1, keepdims=True)
    z = tf.abs(z)
    z = z * z
    return z


def make_tensorboard_images(dynamic_range, p, b, y):
    # p = p / tf.reduce_max(p, [1, 2, 3], keepdims=True)
    # b = b / tf.reduce_max(b, [1, 2, 3], keepdims=True)
    # y = y / tf.reduce_max(y, [1, 2, 3], keepdims=True)

    # Normalize the B-mode image by its maximum value
    bnorm = b / tf.reduce_max(b, [1, 2, 3], keepdims=True)

    def compShift(img, dr, y):
        # Apply L2 optimal weight to img (w.r.t. y)
        wopt_dB = losses.compute_l2_wopt_dB(y, img)
        img_dB = losses._dB(img) + wopt_dB
        # Clip img_dB by dynamic range
        img_dB -= dr[0]
        img_dB /= dr[1] - dr[0]
        img_dB = tf.clip_by_value(img_dB, 0, 1)
        img_dB = tf.transpose(img_dB, [0, 3, 2, 1])
        return img_dB

    p = compShift(p, dynamic_range, bnorm)
    b = compShift(b, dynamic_range, bnorm)
    y = compShift(y, dynamic_range, bnorm)

    return p, b, y


class TensorBoardBmode(tf.keras.callbacks.TensorBoard):
    """
    TensorBoardBmode extends tf.keras.callbacks.TensorBoard, adding custom processing
    upon setup and after every epoch to store properly processed ultrasound images.
    """

    def __init__(self, val_data, *args, **kwargs):
        # Use base class initialization
        self.val_data = val_data
        super().__init__(*args, **kwargs)

    def set_model(self, *args, **kwargs):
        """ Override set_model function to add image TensorBoard summaries. """
        # Use base class implementation
        super().set_model(*args, **kwargs)

        # Make ground truth, NN B-mode, and DAS B-mode images
        dynamic_range = [-60, 0]
        bimg = make_bmode_tf(self.model.inputs[0])  # DAS B-mode image
        yhat = self.model.outputs[0]  # Predictions
        if tf.__version__ <= "1.13.0":
            ytgt = self.model.targets[0]  # Ground truth
        else:
            ytgt = self.model._targets[0]  # Ground truth

        # For some reason, tf.keras gives ytgt unknown shape, so reshape it to match yhat
        szy = yhat.get_shape().as_list()
        szy[0] = -1
        ytgt = tf.reshape(ytgt, szy)
        yhat, bimg, ytgt = make_tensorboard_images(dynamic_range, yhat, bimg, ytgt)

        self.bsumm = tf.summary.image("Bmode", bimg)
        self.ysumm = tf.summary.image("Target", ytgt)
        self.psumm = tf.summary.image("Output", yhat)

    def on_epoch_end(self, epoch, logs={}):
        """ At the end of each epoch, add prediction images to TensorBoard."""
        # Use base class implementation
        super().on_epoch_end(epoch, logs)
        if tf.__version__ <= "1.13.0":
            feed_dict = {
                self.model.inputs[0]: self.val_data[0],
                self.model.targets[0]: self.val_data[1],
            }
        else:
            feed_dict = {
                self.model.inputs[0]: self.val_data[0],
                self.model._targets[0]: self.val_data[1],
            }

        # Add the input and target summary to TensorBoard only on first epoch
        if epoch == 0:
            bs, ys = tf.keras.backend.get_session().run(
                [self.bsumm, self.ysumm], feed_dict=feed_dict
            )
            self.writer.add_summary(bs, 0)
            self.writer.add_summary(ys, 0)

        # Add the predictions every 10 epochs
        if epoch % 10 == 9:
            # Add the predicted image summary to TensorBoard every epoch
            ps = tf.keras.backend.get_session().run(self.psumm, feed_dict=feed_dict)
            self.writer.add_summary(ps, epoch + 1)

        self.writer.flush()

