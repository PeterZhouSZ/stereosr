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
import numpy as np
import h5py
import cv2
import random
from skimage import measure

FLAGS = tf.app.flags.FLAGS


class DataSet(object):
    def __init__(self, files):
        self.files = files
        self.file_index = 0
        self.i = 0
        self._num_examples = 0

        self.read_data(self.files[self.file_index])

        packed = self.pack_batch(0)
        t1 = np.concatenate([packed[0][::32, :, :, :] for i in range(32)], axis=0)
        t2 = np.concatenate([packed[1][::32, :, :, :] for i in range(32)], axis=0)
        t3 = np.concatenate([packed[2][::32, :, :, :] for i in range(32)], axis=0)
        t4 = np.concatenate([packed[3][::32, :, :, :] for i in range(32)], axis=0)
        batch = (t1, t2, t3, t4)

        self.reference_batch = batch

    def read_data(self, file):
        with h5py.File(file, 'r') as hf:
            return self.parse_h5(hf)

    def parse_h5(self, hf):
        raise NotImplementedError('subclasses must override parse_h5()')

    def permutate_data(self, perm):
        raise NotImplementedError('subclasses must override permutate_data()')

    def pack_batch(self, i):
        raise NotImplementedError('subclasses must override pack_batch()')

    def next_batch(self):
        if (self.i + 1) * FLAGS.batch_size > self._num_examples:
            self.read_data(self.files[self.file_index])
            self.file_index += 1

            if self.file_index == len(self.files):
                self.file_index = 0

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.i = 0

            self.permutate_data(perm)

        batch = self.pack_batch(self.i)
        batch = self._batch_augmentation(batch)
        self.i += 1
        return batch

    def _batch_augmentation(self, batchs):
        num_batch = len(batchs)
        new_batchs = []

        for batch in batchs:
            new_batchs.append(np.zeros(batch.shape))

        for i in range(batch.shape[0]):
            r = random.randrange(0, 6)

            if r == 0:
                # rotate 0
                for j in range(num_batch):
                    new_batchs[j][i, :, :, :] = batchs[j][i, :, :, :]
            elif r == 1:
                # rotate 90
                for j in range(num_batch):
                    new_batchs[j][i, :, :, :] = np.rot90(batchs[j][i, :, :, :])
            elif r == 2:
                # rotate 180
                for j in range(num_batch):
                    new_batchs[j][i, :, :, :] = batchs[j][i, ::-1, ::-1, :]
            elif r == 3:
                # rotate 270
                for j in range(num_batch):
                    new_batchs[j][i, :, :, :] = np.rot90(batchs[j][i, :, :, :], 3)
            elif r == 4:
                # flip horizontal
                for j in range(num_batch):
                    new_batchs[j][i, :, :, :] = batchs[j][i, :, ::-1, :]
            elif r == 5:
                # flip vertical
                for j in range(num_batch):
                    new_batchs[j][i, :, :, :] = batchs[j][i, ::-1, :, :]

        return new_batchs


class SingleDataSet(DataSet):
    def __init__(self, files):
        DataSet.__init__(self, files)

    def parse_h5(self, hf):
        self._train_data = np.array(hf.get('data'))
        self._train_label = np.array(hf.get('label'))
        self._num_examples = len(self._train_data)

    def permutate_data(self, perm):
        self._train_data = self._train_data[perm]
        self._train_label = self._train_label[perm]

    def pack_batch(self, i):
        batch_data = self._train_data[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :, :, :]
        batch_label = self._train_label[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :, :, :]
        return batch_data, batch_label


class StereoDataSet(DataSet):
    def __init__(self, files):
        DataSet.__init__(self, files)

    def parse_h5(self, hf):
        self._train_data_left = np.array(hf.get('data_left'))
        self._train_data_right = np.array(hf.get('data_right'))
        self._train_data_cbcr = np.array(hf.get('data_cbcr'))
        self._train_label = np.array(hf.get('label'))

        self._num_examples = len(self._train_data_left)

        indicies = list(range(self._train_data_right.shape[3]))
        np.random.shuffle(indicies)

        self._train_data_right = self._train_data_right[:, :, :, indicies]

    def permutate_data(self, perm):
        self._train_data_left = self._train_data_left[perm]
        self._train_data_right = self._train_data_right[perm]
        self._train_data_cbcr = self._train_data_cbcr[perm]
        self._train_label = self._train_label[perm]

    def pack_batch(self, i):
        batch_data_left = self._train_data_left[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :, :, :]
        batch_data_right = self._train_data_right[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :, :,
                           ::int(64 / FLAGS.right_num_channels)]
        batch_data_cbcr = self._train_data_cbcr[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :, :, :]
        batch_label = self._train_label[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :, :, :]
        return batch_data_left, batch_data_right, batch_data_cbcr, batch_label


class PairDataSet(DataSet):
    def __init__(self, train_data_left, train_data_right, train_label):
        DataSet.__init__(self)
        self._train_data_left = train_data_left
        self._train_data_right = train_data_right
        self._train_label = train_label
        self._train_data_pair = []

        for i in range(len(train_data_left)):
            for j in range(train_data_right[i].shape[2]):
                self._train_data_pair.append((i, j))

        self._train_data_pair = np.array(self._train_data_pair)
        self._num_examples = len(self._train_data_pair)
        self.i = self._num_examples

    def next_batch(self):
        if (self.i + 1) * FLAGS.batch_size > self._num_examples:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._train_data_pair = self._train_data_pair[perm]
            self.i = 0

        batch_data_left = []
        batch_data_right = []
        batch_label = []

        for pair in self._train_data_pair[i * FLAGS.batch_size: (self.i + 1) * FLAGS.batch_size]:
            batch_data_left.append(self._train_data_left[pair[0]])
            batch_data_right.append(self._train_data_right[pair[0], :, :, pair[1]:pair[1] + 1])
            batch_label.append(self._train_label[pair[0]])

        self.i += 1

        return batch_data_left, batch_data_right, batch_label
