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

import numpy as np
import tensorflow as tf
from network import Network

FLAGS = tf.app.flags.FLAGS

class ShiftStereoBundleColor(Network):

    def __init__(self, weight_parameters, bias_parameters):
        Network.__init__(self, weight_parameters, bias_parameters)

    def inference(self, left_images, right_images, cbcr_images):
        images = tf.concat([left_images, right_images], 3)

        conv1 = self.conv('conv1', images, 3, 1, FLAGS.right_num_channels + FLAGS.left_num_channels, 64)

        conv_node = conv1
        for i in range(FLAGS.number_of_layers - 2):
            conv_node = self.conv('conv%d' % (i + 2), conv_node, 3, 1, 64, 64)

        matching = self.conv('conv_last', conv_node, 3, 1, 64, 1, use_relu=False)
        self.layers['matching'] = matching

        if FLAGS.is_last_sigmoid:
            matching = tf.sigmoid(matching)
        elif FLAGS.is_last_relu:
            matching = tf.nn.relu(matching)

        if FLAGS.is_residual:
            if FLAGS.is_last_sigmoid:
                logits = (matching - 0.5) + left_images
            elif FLAGS.is_last_relu:
                logits = (matching - 1) + left_images
            else:
                logits = matching + left_images
        else:
            logits = matching

        luminance_image = logits

        # Color Recon
        images = tf.concat([luminance_image, cbcr_images], 3)

        conv1 = self.conv('color_conv1', images, 3, 1, 3, 64)
        conv_node = conv1
        for i in range(14):
            conv_node = self.conv('color_conv%d' % (i + 2), conv_node, 3, 1, 64, 64)

        residual = self.conv('color_conv16', conv_node, 3, 1, 64, 3)
        self.layers['merged'] = residual
        merged = tf.concat([residual, images], 3)
        logits = self.conv('color_residual', merged, 3, 1, 6, 3)

        return logits
