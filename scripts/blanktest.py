from pybug.io import auto_import
from pybug.landmark.labels import (labeller, ibug_68_points,
                                    ibug_68_contour, ibug_68_trimesh)
from pybug.shape import PointCloud

# load the training images of the LFPW database as landmarked images using the autoimporter
images = auto_import('/Users/joan/PhD/DataBases/lfpw/testset/' + '*.png', max_images=10)
shapes = [img.landmarks['PTS'].lms for img in images]

from pybug.image import MaskedNDImage
from pybug.activeappearancemodel.base import gaussian_pyramid
img = images[0]
gaussian_pyramid(img)