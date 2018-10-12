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

from __future__ import division

from datetime import datetime
import time

from config import *
from data_input import *

FLAGS = tf.app.flags.FLAGS


def read_data(file):
    with h5py.File(file, 'r') as hf:
        data_left = hf.get('data_left')
        data_right = hf.get('data_right')
        label = hf.get('label')
        return np.array(data_left), np.array(data_right), np.array(label)


def loss(logits, labels):
    with tf.name_scope('loss') as scope:
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, logits))))
        return loss


def loss_only_valid(logits, labels):
    with tf.name_scope('loss') as scope:
        nonzero_elements = tf.cast(tf.logical_not(tf.equal(logits, tf.zeros_like(logits))), tf.float32)
        count = tf.reduce_sum(nonzero_elements)
        filtered = tf.mul(tf.subtract(labels, logits), nonzero_elements)
        loss = tf.sqrt(tf.div(tf.reduce_sum(tf.square(filtered)), count))
        return loss


def tf_psnr(logits, labels):
    with tf.name_scope('psnr') as scope:
        err = tf.reduce_mean(tf.square(tf.subtract(labels, logits)))
        denominator = tf.log(tf.constant(10, dtype=err.dtype))
        psnr_value = -10 * (tf.log(err) / denominator)

    return psnr_value


def run_benchmark():
    # Read Data
    data_set = DATA_SET

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Construction tensorflow session
    with tf.device('/gpu:0'):
        left_images = tf.placeholder(tf.float32, shape=(
        FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.left_num_channels), name='left_images')
        right_images = tf.placeholder(tf.float32, shape=(
        FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.right_num_channels), name='right_images')
        cbcr_images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 2),
                                     name='cbcr_images')
        labels = tf.placeholder(tf.float32,
                                shape=(FLAGS.batch_size, FLAGS.label_size, FLAGS.label_size, FLAGS.output_num_channels),
                                name='label')

        weight_parameters = []
        bias_parameter = []
        network = build_network(weight_parameters, bias_parameter)
        network.left_num_channels = FLAGS.left_num_channels
        network.right_num_channels = FLAGS.right_num_channels
        outputs = network.inference(left_images, right_images, cbcr_images)

        loss_op = loss(outputs, labels)

        with tf.device('/cpu:0'):
            tf.summary.scalar('loss', loss_op)

        lr = 0.0001

        op = tf.train.AdamOptimizer(lr)
        train_op = op.minimize(loss_op, global_step=global_step)

    with tf.Session() as sess:

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.summary.merge_all()

        init = tf.initialize_all_variables().run()

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # Find Checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Checkpoint found')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found')

        max_iteration = FLAGS.num_iter

        num_steps_burn_in = 100
        step = 0

        start_time = time.time()

        while True:
            batch_data_left, batch_data_right, batch_data_cbcr, batch_label = data_set.next_batch()
            # print (batch_label.shape)

            feed_dict = {
                left_images: batch_data_left,
                right_images: batch_data_right,
                cbcr_images: batch_data_cbcr,
                labels: batch_label}

            _, loss_value, step = sess.run([train_op, loss_op, global_step], feed_dict=feed_dict)

            if step % 100 == 0:
                reference_batch = data_set.reference_batch

                feed_dict = {
                    left_images: reference_batch[0],
                    right_images: reference_batch[1],
                    cbcr_images: reference_batch[2],
                    labels: reference_batch[3],
                }

                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            # if i > num_steps_burn_in:

            if step % num_steps_burn_in == 0:
                duration = time.time() - start_time
                start_time = time.time()
                saver.save(sess, os.path.join(FLAGS.train_dir, 'model.ckpt'), global_step=global_step)
                print('%s: step %d, duration = %.3f, loss = %f' %
                      (datetime.now(), step, duration, loss_value))

            if step > max_iteration:
                return


def main(_):
    run_benchmark()


if __name__ == '__main__':
    tf.app.run()
