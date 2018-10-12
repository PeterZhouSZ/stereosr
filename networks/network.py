"""
=======================================================================
General Information
-------------------
Codename: StereoSR (CVPR 2018)
Writers: Daniel S. Jeon (sjjeon@vclab.kaist.ac.kr), Seung-Hwan Baek (shwbaek@vclab.kaist.ac.kr), Inchang Choi (inchangchoi@vclab.kaist.ac.kr), Min H. Kim (minhkim@vclab.kaist.ac.kr)
Institute: KAIST Visual Computing Laboratory
For information please see the paper:
Enhancing the Spatial Resolution of Stereo Images using a Parallax Prior CVPR 2018, Daniel S. Jeon, Seung-Hwan Baek, Inchang Choi, Min H. Kim
Please cite this paper if you use this code in an academic publication.
Bibtex:     
@InProceedings{Jeonetal:CVPR:2018,
author  = {Daniel S. Jeon and Seung-Hwan Baek and Inchang Choi and Min H. Kim},
title   = {Enhancing the Spatial Resolution of Stereo Images using a Parallax Prior},
booktitle = {Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2018)},
publisher = {IEEE},  
address = {Salt Lake City, Utah, United States},
year = {2018},
pages = {},
volume  = {},
}
==========================================================================
License Information
-------------------
Daniel S. Jeon, Seung-Hwan Baek, Inchang Choi, Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:
Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.
The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].
License: GNU General Public License Usage Alternatively, this file may be used under the terms of the GNU General Public License version 3.0 as published by the Free Software Foundation and appearing in the file LICENSE.GPL included in the packaging of this file. Please review the following information to ensure the GNU General Public License version 3.0 requirements will be met: http://www.gnu.org/copyleft/gpl.html.
Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
=======================================================================
"""

import tensorflow as tf
import re


class Network:
    def __init__(self, weight_parameters, bias_parameters):
        self.epsilon = 1e-3
        self.weight_parameters = weight_parameters
        self.bias_parameters = bias_parameters
        self.layers = {}

    def batch_norm(self, x, n_out):
        return tf.layers.batch_normalization(x)

    def load_conv(self, name, input, stride, weight, bias, use_relu=True):
        var_weight = tf.Variable(weight, trainable=True, name=name + "_weights")
        with tf.name_scope(name) as scope:
            # kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, num_in, num_out], dtype=tf.float32, stddev=1e-3), name='weights')
            # Xavier initializer
            # kernel = tf.get_variable(name + "_weights", shape=[kernel_size, kernel_size, num_in, num_out], initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input, var_weight, strides=[1, stride, stride, 1], padding='SAME')
            biases = tf.Variable(bias, trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            # norm = batch_norm(bias, num_out)
            if (use_relu):
                relu = tf.nn.relu(bias, name=scope)
            else:
                relu = bias
            self.activation_summary(relu)
            # self.weight_parameters.append(kernel)
            # self.bias_parameters.append(biases)

        self.layers[name] = relu

        return relu

    def conv(self, name, input, kernel_size, stride, num_in, num_out, use_relu=True):
        with tf.name_scope(name) as scope:
            # kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, num_in, num_out], dtype=tf.float32, stddev=1e-3), name='weights')
            # Xavier initializer
            kernel = tf.get_variable(name + "_weights", shape=[kernel_size, kernel_size, num_in, num_out],
                                     initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[num_out], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            norm = self.batch_norm(bias, num_out)
            if use_relu:
                relu = tf.nn.relu(norm, name=scope)
            else:
                relu = norm
            self.activation_summary(relu)
            self.weight_parameters.append(kernel)
            self.bias_parameters.append(biases)

        self.layers[name] = relu

        return relu

    def fully(self, name, input, num_in, num_out):
        with tf.name_scope(name) as scope:
            weights = tf.Variable(
                tf.truncated_normal([num_in, num_out], dtype=tf.float32, stddev=1.0 / np.sqrt(float(num_in))),
                name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[num_out], dtype=tf.float32), trainable=True, name='biases')
            fully = tf.matmul(input, weights)
            bias = tf.nn.bias_add(fully, biases)
            relu = tf.nn.relu(bias, name=scope)
            self.activation_summary(relu)
            self.weight_parameters.append(weights)
            self.bias_parameters.append(biases)

        return relu

    def conv_up(self, name, conv_low, conv_high, num_in, num_out):
        shape = conv_high.get_shape()
        conv_r = conv(name, conv_low, 3, 1, num_in, num_out)
        conv_u = tf.image.resize_bilinear(conv_r, (shape[1].value, shape[2].value))
        conv_add = tf.add(conv_high, conv_u)

        self.layers[name] = conv_add

        return conv_add

    def activation_summary(self, x):
        with tf.device('/cpu:0'):
            tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
            # tf.histogram_summary(tensor_name + '/activations', x)
            # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var

    def numel(self, layer):
        shape = layer.get_shape()
        return shape[1].value * shape[2].value * shape[3].value
