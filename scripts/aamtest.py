from __future__ import division
from copy import deepcopy
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pylab as plt

from pybug.io import auto_import
from pybug.transform.affine import Translation
from pybug.transform.modeldriven import GlobalMDTransform
from pybug.transform.affine import SimilarityTransform
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.landmark.labels import labeller, ibug_68_points
from pybug.image import MaskedNDImage
from pybug.lucaskanade.residual import LSIntensity
from pybug.lucaskanade.appearance import (AlternatingForwardAdditive,
                                          AlternatingForwardCompositional,
                                          AlternatingInverseCompositional)

#------------------------------------------------------------------------------


def compute_p2pe(fitted, ground_truth):
    return np.mean(np.sqrt(np.sum((fitted - ground_truth) ** 2, axis=-1)))


def compute_rame(fitted, ground_truth):
    face_size = np.mean(np.max(ground_truth, axis=0) - np.min(ground_truth, axis=0))
    return compute_p2pe(fitted, ground_truth) / face_size


#------------------------------------------------------------------------------


# load the training images of the LFPW database as landmarked images using
# the auto-importer
images = auto_import('/vol/atlas/databases/lfpw/testset/image_0001.png')
# extract shape data. -1 because the annotation are 1 based
translation = Translation(np.array([-1, -1]))
for img in images:
    translation.apply_inplace(img.landmarks['PTS'].lms)
# label the landmarks using the ibug's "standard" 68 points mark-up
labeller(images, 'PTS', ibug_68_points)


# load a previously buid AAM
aam = pickle.load(open('/vol/atlas/aams/lfpw_pwa_multi', "rb"))
ref_frame_list = aam["ref_frame_list"]
appearance_model_list = aam["appearance_model_list"]
shape_model_list = aam["shape_model_list"]


# define sources
source_1 = ref_frame_list[0].landmarks['ibug_68_trimesh'].lms
source_2 = ref_frame_list[1].landmarks['ibug_68_trimesh'].lms
source_3 = ref_frame_list[2].landmarks['ibug_68_trimesh'].lms
# define dummy global transform
dummy_similarity_transform = SimilarityTransform.from_vector(np.array([0, 0, 0, 0]))
global_transform = dummy_similarity_transform
# define dummy global model transforms
md_transform_list = []
for sm, rf in zip(shape_model_list, ref_frame_list):
    md_transform_list.append(
        GlobalMDTransform(sm, PiecewiseAffineTransform,
                          source=rf.landmarks['ibug_68_trimesh'].lms,
                          global_transform=global_transform))
# standard Least Squares Residual
residual = LSIntensity()
# forward additive
fa_multi = AlternatingForwardAdditive(appearance_model_list, 
                                      residual, 
                                      deepcopy(md_transform_list),
                                      interpolator='cinterp')
# forward compositional
fc_multi = AlternatingForwardCompositional(appearance_model_list, 
                                           residual, 
                                           deepcopy(md_transform_list),
                                           interpolator='cinterp')
# inverse compositional
ic_multi = AlternatingInverseCompositional(appearance_model_list, 
                                           residual, 
                                           deepcopy(md_transform_list),
                                           interpolator='cinterp')


# initialization
level = 0
noise_scale = 0.01
global_transform_list = []
for img in images:
    global_transform = SimilarityTransform.align(shape_model_list[level].mean, 
                                                 img.landmarks['PTS'].lms)
    global_trans_params = global_transform.as_vector()
    # kill rotation
    global_trans_params[1] = 0
    # add Gaussian noise
    # global_trans_params += noise_scale * np.random.randn(4)
    global_transform_list.append(global_transform.from_vector(global_trans_params))
# define initial global model transforms
ini_md_trans = [GlobalMDTransform(shape_model_list[level], 
                                  PiecewiseAffineTransform, 
                                  source=ref_frame_list[level].landmarks['PTS'].lms, 
                                  global_transform=gt)
                for gt in global_transform_list]


# define NDImages and shapes
gs_images = [img.as_greyscale() for img in images]
nd_images = [MaskedNDImage(img.pixels, mask=img.mask) for img in gs_images]
shapes = [img.landmarks['PTS'].lms for img in images]



# load matlab data
data = sio.loadmat('/data/matlab/aams/aamtoolbox/data.mat')
img_data = data['img']
img = MaskedNDImage(img_data[..., None])



# fit
#fa_multi_result = [deepcopy(fa_multi.align(img, deepcopy(t), max_iters=50))
#                   for t, img in zip(ini_md_trans, nd_images)]
#fc_multi_result = [deepcopy(fc_multi.align(img, deepcopy(t), max_iters=50))
#                   for t, img in zip(ini_md_trans, nd_images)]
ic_multi_result = deepcopy(ic_multi.align(img,
                                          deepcopy(ini_md_trans[0]),
                                          max_iters=50))


# compute initial error
initial_shapes = [t.target for t in ini_md_trans]
initial_error = np.array([compute_rame(f.points, s.points)
                          for f, s in zip(initial_shapes, shapes)])
# compute fitting error
fitted_shapes = [t.target for t in ic_multi_result]
fc_error = np.array([compute_rame(f.points, s.points)
                     for f, s in zip(fitted_shapes, shapes)])

# define step
stop = 0.05
step = 0.001
xs = np.arange(0, stop, step)
n_fittings = len(ic_multi_result)
# compute fitting cumulative graph
fc_cumulative_error = np.zeros_like(xs)
for i, limit in enumerate(xs):
    fc_cumulative_error[i] = (np.count_nonzero(fc_error < limit) / 
                              n_fittings)
# compute initial cumulative graph
initial_cumulative_error = np.zeros_like(xs)
for i, limit in enumerate(xs):
    initial_cumulative_error[i] = (np.count_nonzero(initial_error < limit) / 
                                   n_fittings)
# plot graphs
plt.plot(xs, fc_cumulative_error, 'ro-')
plt.plot(xs, initial_cumulative_error, 'b^-')
# print mean, median and % of fitted images under 0.05
print 'mean: {}'.format(np.mean(fc_error))
print 'median: {}'.format(np.median(fc_error))
print fc_cumulative_error[-1]



1
