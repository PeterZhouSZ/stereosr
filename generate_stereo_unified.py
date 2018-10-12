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

import os

import h5py

from imageop import *

folder = 'data/Train_stereo'
savename = 'data/color_stereo_x2/train_stereo'

size_input = 33
size_label = 33
scale = 2
stride = 24
num_shift = 64
num_max_data = 3800

left_num_channels = 1
right_num_channels = num_shift
label_num_channels = 3

padding = int(abs(size_input - size_label) / 2)
data_left = np.zeros((num_max_data, size_input, size_input, left_num_channels))
data_right = np.zeros((num_max_data, size_input, size_input, right_num_channels))
data_cbcr = np.zeros((num_max_data, size_input, size_input, 2))
label = np.zeros((num_max_data, size_label, size_label, label_num_channels))

filepaths = os.listdir(folder)


def save(data_left, data_right, data_cbcr, label, count, data_number):
    data_left = data_left[:count, :, :, :]
    data_right = data_right[:count, :, :, :]
    data_cbcr = data_cbcr[:count, :, :, :]
    label = label[:count, :, :, :, ]

    save_path = savename + str(data_number) + '.h5'
    with h5py.File(save_path, 'w') as f:
        dset_data_left = f.create_dataset('data_left', data=data_left, dtype=np.float32)
        dset_data_right = f.create_dataset('data_right', data=data_right, dtype=np.float32)
        dset_data_cbcr = f.create_dataset('data_cbcr', data=data_cbcr, dtype=np.float32)
        dset_label = f.create_dataset('label', data=label, dtype=np.float32)

        print('Save: %s, Count: %d' % (save_path, count))


data_number = 0
total_count = 0
count = 0
total_mse = 0
for i in range(len(filepaths)):
    dataname = filepaths[i]
    directory = folder + '/' + dataname

    left_image = modcrop(cv2.imread(directory + '/im0.png').astype(np.float32) * (1 / 255), scale)
    right_image = modcrop(cv2.imread(directory + '/im1.png').astype(np.float32) * (1 / 255), scale)

    [height, width, channels] = left_image.shape

    left_low_rgb = down_res(left_image, scale)
    right_low_rgb = down_res(right_image, scale)

    left_ycbcr = cv2.cvtColor(left_image, cv2.COLOR_BGR2YCR_CB)
    left_low_ycbcr = cv2.cvtColor(left_low_rgb, cv2.COLOR_BGR2YCR_CB)
    right_low_ycbcr = cv2.cvtColor(right_low_rgb, cv2.COLOR_BGR2YCR_CB)

    left_y = left_ycbcr[:, :, 0:1]
    left_low_y = left_low_ycbcr[:, :, 0:1]
    right_low_y = right_low_ycbcr[:, :, 0:1]

    im_label = left_ycbcr

    # left_input = scale_to(left_demosaic_ycbcr, (width, height))[:, :, 0:1]
    left_input = left_low_y

    right_input = np.zeros((height, width, num_shift))

    for i in range(num_shift):
        right_input[:, :, i:i + 1] = np.roll(right_low_y, i, 1)

    cbcr_input = left_low_ycbcr[:, :, 1:3]

    for y in range(128, height - size_input, stride):
        for x in range(128, width - size_input, stride):

            subim_input_left = left_input[y: y + size_input, x: x + size_input, :]
            subim_input_cbcr = cbcr_input[y: y + size_input, x: x + size_input, :]

            subim_input_right = np.zeros((size_input, size_input, num_shift))
            for i in range(num_shift):
                subim_input_right[:, :, i] = right_input[y: y + size_input, x: x + size_input, i]

            subim_label = im_label[y + padding: y + padding + size_label,
                          x + padding: x + padding + size_label]
            err = np.mean((subim_input_right[:, :, 0] - subim_label[:, :, 0]) ** 2)

            if count < 1000 and count % 100 == 0:
                cv2.imwrite('output/sample/%d_left.png' % count, subim_input_left[:, :, 0] * 255)
                cv2.imwrite('output/sample/%d_right.png' % count, subim_input_right[:, :, 16] * 255)
                cv2.imwrite('output/sample/%d_label.png' % count, subim_label[:, :, 0] * 255)

            # if err > 0.05:
            total_mse += err

            data_left[count, :, :, :] = subim_input_left
            data_right[count, :, :, :] = subim_input_right
            data_cbcr[count, :, :, :] = subim_input_cbcr
            label[count, :, :, :] = subim_label

            count += 1
            total_count += 1

            if count == num_max_data:
                save(data_left, data_right, data_cbcr, label, count, data_number)

                data_left = np.zeros((num_max_data, size_input, size_input, left_num_channels))
                data_right = np.zeros((num_max_data, size_input, size_input, right_num_channels))
                data_cbcr = np.zeros((num_max_data, size_input, size_input, 2))
                label = np.zeros((num_max_data, size_label, size_label, label_num_channels))
                count = 0
                data_number += 1

save(data_left, data_right, data_cbcr, label, count, data_number)

avg_mse = total_mse / total_count
print(avg_mse)
print(total_count)
