import numpy as np
from pybug.io import auto_import
from pybug.landmark.labels import labeller, ibug_68_points, ibug_68_contour, ibug_68_trimesh
from pybug.transform.affine import Translation
from pybug.image import RGBImage

# load the training images of the LFPW database as landmarked images using the autoimporter
images2 = auto_import('/vol/atlas/databases/lfpw/trainset/*.png', max_images=10)

img = images2[3]
img = images2[0].rescale(1.33)
a = img.extract_local_patches()