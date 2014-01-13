from pybug.io import auto_import
from pybug.image import RGBImage, MaskedNDImage

images = auto_import('/Users/joan/PhD/DataBases/lfpw/trainset/*.png')
images = [i.as_greyscale() if type(i) is RGBImage else i for i in images]

new_images = []
for i in images:
    img = MaskedNDImage(i.pixels)
    img.landmarks = i.landmarks
    new_images.append(img)
images = new_images

del new_images

# <codecell>

from pybug.activeappearancemodel import aam_builder, AAM
from pybug.transform.tps import TPS
from pybug.landmark import ibug_68_trimesh

options = {'triangulation': {'function': ibug_68_trimesh, 'label': 'ibug_68_trimesh'},
           'features': {'type': None, 'options': {'kwargs': None}},
           'max_shape_components': 25,
           'max_appearance_components': 250}

aam = aam_builder(images, **options)

import numpy as np

# <codecell>

aam.instance(np.random.randn(25), np.random.randn(10), level=2).view()

# <codecell>

from pybug.io import auto_import
from pybug.landmark.labels import (labeller, ibug_68_points,
                                    ibug_68_contour, ibug_68_trimesh)
from pybug.shape import PointCloud

# load the training images of the LFPW database as landmarked images using the autoimporter
images = auto_import('/Users/joan/PhD/DataBases/lfpw/testset/' + '*.png')
shapes = [img.landmarks['PTS'].lms for img in images]

# <codecell>

aam.lk_fit_annotated_database(images)