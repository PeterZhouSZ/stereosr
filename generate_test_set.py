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

def generate_test_data(test_set, name, scale):
    left_image = modcrop(cv2.imread('data/%s/left/%s.png' % (test_set, name)).astype(np.float32) * (1 / 255), scale)
    right_image = modcrop(cv2.imread('data/%s/right/%s.png' % (test_set, name)).astype(np.float32) * (1 / 255), scale)

    [height, width, channels] = left_image.shape

    left_low_rgb = down_res(left_image, scale)
    right_low_rgb = down_res(right_image, scale)

    os.makedirs('data/%s/x%d/label' % (test_set, scale), exist_ok=True)
    os.makedirs('data/%s/x%d/left' % (test_set, scale), exist_ok=True)
    os.makedirs('data/%s/x%d/right' % (test_set, scale), exist_ok=True)

    cv2.imwrite('data/%s/x%d/label/%s.png' % (test_set, scale, name), left_image * 255)
    cv2.imwrite('data/%s/x%d/left/%s.png' % (test_set, scale, name), left_low_rgb * 255)
    cv2.imwrite('data/%s/x%d/right/%s.png' % (test_set, scale, name), right_low_rgb * 255)


if __name__ == '__main__':
    for name in ['piano', 'motorcycle', 'pipes', 'cloth2', 'sword2']:
        generate_test_data('Test_Set_middlebury', name, 2)
        generate_test_data('Test_Set_middlebury', name, 3)
        generate_test_data('Test_Set_middlebury', name, 4)