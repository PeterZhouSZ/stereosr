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
import getopt

sys.path.append('networks')

import tensorflow as tf
import os

from shift_stereo_bundle_color import ShiftStereoBundleColor
from data_input import *

right_num_channels = 64
number_of_layers = 16
is_last_sigmoid = False
is_last_relu = False

if (len(sys.argv) > 3):
    right_num_channels = int(sys.argv[1])
    number_of_layers = int(sys.argv[2])
    is_last_sigmoid = sys.argv[3] in ["True"]
    is_last_relu = sys.argv[4] in ["True"]

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('image_size', 33,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 33,
                            """Size of the labels.""")

tf.app.flags.DEFINE_string('data_name', 'Motorcycle',
                           """Directory where to write event logs """)

tf.app.flags.DEFINE_string('train_dir', 'models/n_shift_stereo_color_layer_%d_%d_%s_%s' % (right_num_channels, number_of_layers, is_last_sigmoid, is_last_relu),
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('scale', 2,
                          """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('num_iter', 100000,
                         """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('left_num_channels', 1,
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('right_num_channels', right_num_channels,
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('output_num_channels', 3,
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('number_of_layers', number_of_layers,
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_boolean('is_right_picking', False,
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_boolean('is_residual', True,
                           """Directory where to write event logs """)

tf.app.flags.DEFINE_boolean('is_last_sigmoid', is_last_sigmoid,
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_boolean('is_last_relu', is_last_relu,
                           """Directory where to write event logs """)


DATA_PATH = 'data/color_stereo_x2/'
if os.path.isdir(DATA_PATH):
    DATA_SET = StereoDataSet(list(map(lambda fn: DATA_PATH + fn, os.listdir(DATA_PATH))))

_network = ShiftStereoBundleColor


def build_network(weight_parameters, bias_parameters):
    return _network(weight_parameters, bias_parameters)
