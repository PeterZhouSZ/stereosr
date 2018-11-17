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
sys.path.append('networks')

from networks.shift_stereo_bundle_color import ShiftStereoBundleColor
from imageop import *
from config import *
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt

FLAGS = tf.app.flags.FLAGS

sr_model = 'models/x4'


def run_network(left_image, right_image):
    right_num_channels = 64

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

    return out


def run_stereosr(left_image, right_image, output_path, scale):
    if scale % 2 == 0:
        left_image = modcrop(left_image, scale)
        right_image = modcrop(right_image, scale)
    else:
        left_image = modcrop(left_image, scale * 2)
        right_image = modcrop(right_image, scale * 2)

    [height, width, channels] = left_image.shape

    left_image_low = down_res(left_image, scale)
    right_image_low = down_res(right_image, scale)

    left_image = left_image[:, 64:, :]
    left_image_bicubic = left_image_low[:, 64:, :]
    left_image_sr = run_network(left_image_low, right_image_low)

    psnr_bicubic = measure.compare_psnr(left_image_bicubic, left_image)

    psnr_sr = measure.compare_psnr(left_image_sr, left_image)

    print("Bicubic: PSNR: %.2f" % psnr_bicubic)
    print("StereoSR: PSNR: %.2f" % psnr_sr)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    cv2.imwrite("%s/left_image.png" % output_path, left_image * 255.0)
    cv2.imwrite("%s/bicubic.png" % output_path, left_image_bicubic * 255.0)
    cv2.imwrite("%s/sr.png" % output_path, left_image_sr * 255.0)

    # show plot
    plt.subplot(131)
    plt.gca().set_title('Ground truth')
    plt.gca().set_xlabel('PSNR')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))

    plt.subplot(132)
    plt.gca().set_title('Bicubic')
    plt.gca().set_xlabel('%.2f dB' % psnr_bicubic)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(left_image_bicubic, cv2.COLOR_BGR2RGB))

    plt.subplot(133)
    plt.gca().set_title('Ours')
    plt.gca().set_xlabel('%.2f dB' % psnr_sr)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(left_image_sr, cv2.COLOR_BGR2RGB))

    plt.show()


def main():
    scale = 4

    # input paths
    path_im1 = 'data/cloth2/im1.png'
    path_im2 = 'data/cloth2/im2.png'
    output_dir = 'data/cloth2/output/'

    left_image = cv2.imread(path_im1).astype(np.float32) / 255.0
    right_image = cv2.imread(path_im2).astype(np.float32) / 255.0

    run_stereosr(left_image, right_image, output_dir, scale)


if __name__ == "__main__":
    main()
