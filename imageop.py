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

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import re
import numpy as np
from skimage import measure


def mse(imageA, imageB):
    err = np.mean((imageA - imageB) ** 2)
    return err


def psnr(imageA, imageB):
    return measure.compare_psnr(imageA, imageB)


def modcrop(img, modulo):
    sz = img.shape
    if len(sz) == 3:
        height = sz[0] - np.mod(sz[0], modulo)
        width = sz[1] - np.mod(sz[1], modulo)
        img = img[:height, :width, :]
    else:
        height = sz[0] - np.mod(sz[0], modulo)
        width = sz[1] - np.mod(sz[1], modulo)
        img = img[:height, :width]

    return img


def pad_crop(img, padding):
    return img[padding:-padding, padding:-padding]


def tv(img):
    shift_y = np.roll(img, 1, 0)
    shift_x = np.roll(img, 1, 1)
    diff_y = np.abs(shift_y - img)
    diff_x = np.abs(shift_x - img)
    return np.sum(diff_x + diff_y)


def down_res(img, scale):
    [height, width, channels] = img.shape
    t = down(img, scale)

    result = cv2.resize(t, (width, height))

    result[result < 0] = 0
    result[result > 1] = 1
    return result


def down(img, scale):
    [height, width, channels] = img.shape
    small = down_scale(img, scale)

    return small


def down_scale(img, scale):
    [height, width, channels] = img.shape
    return cv2.resize(img, (int(width / scale), int(height / scale)), interpolation=cv2.INTER_AREA)


def up_scale(img, scale):
    [height, width, channels] = img.shape
    return cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)


def scale_to(img, size):
    return cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)


def save_layer(name, layer, prefix):
    layer_depth = layer.shape[3]

    count = 0

    for i in range(layer_depth):
        path_normalized = "%s/normalized_%d.png" % (prefix, i)
        path = "%s/%d.png" % (prefix, i)
        img = (layer[0, :, :, i] / (np.max(layer[0, :, :, i]) + 0.0000001)) * 255
        tv_value = tv(img)
        tv_avg = tv_value / img.size

        if tv_avg != 0:
            count += 1

        cv2.imwrite(path_normalized, img)
        cv2.imwrite(path, layer[0, :, :, i] * 255)

    print("Name: %s, Total layer: %d,  Valid layer: %d" % (name, layer_depth, count))


def rgb2bayer(rgb):
    [h, w, c] = rgb.shape

    raw = np.zeros((h, w))
    raw[::2, ::2] = rgb[::2, ::2, 2]
    raw[1::2, ::2] = rgb[1::2, ::2, 1]
    raw[::2, 1::2] = rgb[::2, 1::2, 1]
    raw[1::2, 1::2] = rgb[1::2, 1::2, 0]

    return raw


def bayerVisualizer(raw):
    [h, w] = raw.shape
    img = np.zeros((h, w, 3))
    raw[::2, ::2] = raw[::2, ::2]
    raw[1::2, ::2] = raw[1::2, ::2]
    raw[::2, 1::2] = raw[::2, 1::2]
    raw[1::2, 1::2] = raw[1::2, 1::2]


def rgb2ycbcr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale


'''
Save a Numpy array to a PFM file.
'''


def save_pfm(file, image, scale=1):
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write(('%d %d\n' % (image.shape[1], image.shape[0])).encode())

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode())

    image.tofile(file)
