from __future__ import division
from pybug.io import auto_import
from pybug.image import RGBImage
import numpy as np

images = auto_import('/Users/joan/PhD/DataBases/lfpw/trainset/*.png',
                     max_images=811)
images = [i.as_greyscale() if type(i) is RGBImage else i for i in images]

from pybug.activeappearancemodel.builder import aam_builder
from pybug.landmark import ibug_68_trimesh

options = {'triangulation': {'function': ibug_68_trimesh, 'label': 'ibug_68_trimesh'},
           'features': {'type': 'igo', 'options': {'double_angles': True}},
           'max_shape_components': 25,
           'max_appearance_components': 250}


aam = aam_builder(images, **options)


images = auto_import('/Users/joan/PhD/DataBases/lfpw/testset/*.png')
images = [i.as_greyscale() if type(i) is RGBImage else i for i in images]

# crop images around their landmarks
for i in images:
    i.crop_to_landmarks_proportion(0.2)


from pybug.lucaskanade.appearance.projectout import \
    ProjectOutInverseCompositional

from pybug.lucaskanade.residual import Huber, GemanMcClure, L1L2

aam.initialize_lk(n_shape=[3, 6, 12], n_appearance=[250, 250, 250],
                  lk_algorithm=ProjectOutInverseCompositional)

fitted_transforms = aam.lk_fit_landmarked_database(images, runs=1, noise_std=0.0, verbose=True, view=True, max_iters=20)


# #### 6. Evaluate Performance

from pybug.activeappearancemodel.functions import plot_ced
from pybug.transform.affine import UniformScale


original_landmarks = [i.landmarks['PTS'].lms for i in images]

fitted_landmarks = []
for ft in fitted_transforms:
    for f in ft:
        fitted_landmarks.append(UniformScale(1,2).apply(f[-1].target))

errors = plot_ced(fitted_landmarks, original_landmarks, label='AIC')[0]

fitted_landmarks = []
for ft in fitted_transforms:
    for f in ft:
        fitted_landmarks.append(UniformScale(2,2).apply(f[-2].target))

plot_ced(fitted_landmarks, original_landmarks, label='AIC')

fitted_landmarks = []
for ft in fitted_transforms:
    for f in ft:
        fitted_landmarks.append(UniformScale(4,2).apply(f[-3].target))

plot_ced(fitted_landmarks, original_landmarks, label='AIC');

np.count_nonzero(np.array(errors) < 0.03) / len(images)
