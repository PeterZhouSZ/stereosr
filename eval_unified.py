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

import sys
import os

sys.path.append('networks')

import tensorflow as tf
import time

from single import Single
from shift_stereo import ShiftStereo
from shift_stereo_bundle import ShiftStereoBundle
from shift_stereo_bundle_color import ShiftStereoBundleColor
from shift_stereo_bundle_skip import ShiftStereoBundleSkip
from color_recon import ColorRecon

import cv2
from imageop import *
from config import *
import numpy as np
from tensorflow.python.framework import ops
from skimage import measure

FLAGS = tf.app.flags.FLAGS


# data_path = 'data/Test_Set_middlebury'
# scale = 2

def run_network(left_image, right_image, label_image, index, output_path, scale):
    sr_model = 'models/n_shift_stereo_color_layer_64_16_False_False'
    right_num_channels = 64

    if scale % 2 == 0:
        left_image = modcrop(left_image, scale)
        right_image = modcrop(right_image, scale)
        label_image = modcrop(label_image, scale)
    else:
        left_image = modcrop(left_image, scale * 2)
        right_image = modcrop(right_image, scale * 2)
        label_image = modcrop(label_image, scale * 2)

    [height, width, channels] = left_image.shape

    left_ycbcr = cv2.cvtColor(left_image, cv2.COLOR_BGR2YCR_CB)
    left_y = left_ycbcr[:, :, 0:1]
    left_cbcr = left_ycbcr[:, :, 1:3]

    right_image = right_image[:, :, 0:1]
    [height, width, channels] = left_image.shape
    right_input = np.zeros([height, width, right_num_channels])
    for i in range(right_num_channels):
        right_input[:, :, i:i + 1] = np.roll(right_image, i * 2, 1)

    left_y = np.expand_dims(left_y[:, :, :], axis=0)
    left_cbcr = np.expand_dims(left_cbcr[:, :, :], axis=0)
    left_image = np.expand_dims(left_image[:, :, :], axis=0)
    right_input = np.expand_dims(right_input[:, :, :], axis=0)

    with tf.Session() as sess:

        FLAGS.left_num_channels = 1
        FLAGS.right_num_channels = right_num_channels
        FLAGS.number_of_layers = 16
        FLAGS.is_residual = True
        FLAGS.is_last_relu = False
        FLAGS.is_last_sigmoid = False

        left_holder = tf.placeholder(tf.float32, shape=(1, left_image.shape[1], left_image.shape[2], 1))
        right_holder = tf.placeholder(tf.float32,
                                      shape=(1, left_image.shape[1], left_image.shape[2], FLAGS.right_num_channels))
        cbcr_holder = tf.placeholder(tf.float32, shape=(1, left_image.shape[1], left_image.shape[2], 2))
        label_holder = tf.placeholder(tf.float32, shape=(1, left_image.shape[1], left_image.shape[2], 3), name='label')

        network = ShiftStereoBundleColor([], [])
        network.left_num_channels = 1
        network.right_num_channels = FLAGS.right_num_channels
        conv = network.inference(left_holder, right_holder, left_cbcr)

        saver = tf.train.Saver()

        ckpt = tf.train.latest_checkpoint(sr_model)
        saver.restore(sess, ckpt)

        [out] = sess.run([conv], feed_dict={
            left_holder: left_y,
            right_holder: right_input,
            cbcr_holder: left_cbcr
        })

    tf.reset_default_graph()

    out = out[0, :, 64:, :]
    out = cv2.cvtColor(out, cv2.COLOR_YCR_CB2BGR)

    out = np.clip(out, 0, 1)

    label = label_image[:, 64:, :]
    # left_image = left_image[0,:,64:,:]
    # label = left_image

    cv2.imwrite(output_path, out * 255)

    left_image = left_image[0, :, :, :]

    return measure.compare_psnr(out, label), measure.compare_ssim(out, label, multichannel=True)
    # return 0, 0


def eval_data_set(data_path, scale):
    count = 0
    total_psnr = 0
    total_ssim = 0

    text = ''

    start = time.time()

    for fn in sorted(os.listdir('%s/x%d/left' % (data_path, scale))):
        left_image = cv2.imread('%s/x%d/left/' % (data_path, scale) + fn).astype(np.float32) / 255.0
        right_image = cv2.imread('%s/x%d/right/' % (data_path, scale) + fn).astype(np.float32) / 255.0
        label_image = cv2.imread('%s/x%d/label/' % (data_path, scale) + fn).astype(np.float32) / 255.0
        output_path = '%s/x%d/%04d.png' % (data_path.replace('data', 'output'), scale, count)

        psnr_bicubic = measure.compare_psnr(left_image, label_image)
        ssim_bicubic = measure.compare_ssim(left_image, label_image, multichannel=True)
        psnr, ssim = run_network(left_image, right_image, label_image, count, output_path, scale)

        text += '[%s] PSNR: %f,    SSIM: %f | Bicubic PSNR: %f,    SSIM: %f\n' % (
            fn, psnr, ssim, psnr_bicubic, ssim_bicubic)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

    end = time.time()

    log = '======================================= \n'
    log += 'Data: %s \n' % data_path
    log += 'Scale: %d \n' % scale
    log += 'Total PSNR: %f \n' % total_psnr
    log += 'Total SSIM: %f \n' % total_ssim
    log += 'Average PSNR: %f \n' % (total_psnr / count)
    log += 'Average SSIM: %f \n' % (total_ssim / count)
    log += 'Count: %f \n' % count
    log += 'Time: %f \n' % (end - start)
    log += '======================================= \n'
    log += '\n'
    log += text

    output_dir = '%s/x%d' % (data_path.replace('data', 'output'), scale)
    os.makedirs(output_dir, exist_ok=True)

    f = open('%s/data.txt' % (output_dir), 'w')
    f.write(log)
    f.close()

    print(log)


def main(argv):
    scale = 2
    data_set_path = 'data/Test_Set_middlebury/'
    eval_data_set(data_set_path, scale)


if __name__ == "__main__":
    tf.app.run()
