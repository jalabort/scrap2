from pybug.io import auto_import
from pybug.image import RGBImage, MaskedNDImage

images = auto_import('/Users/joan/PhD/DataBases/lfpw/trainset/*.png',
                     max_images=811)
images = [i.as_greyscale() if type(i) is RGBImage else i for i in images]

new_images = []
for i in images:
    img = MaskedNDImage(i.pixels)
    img.landmarks = i.landmarks
    new_images.append(img)
images = new_images

del new_images


import pickle

# load a previously buid AAM
obj = pickle.load(open('/Users/joan/PhD/Models/aam_lfpw', "rb"))
aam = obj["aam"]


from pybug.activeappearancemodel.builder import mean_pointcloud, \
    rescale_to_reference_landmarks

group = 'PTS'
label = 'all'
interpolator = 'scipy'
reference_landmarks = None
scale = 0.5
crop_boundary = 0.2

print '- Cropping images'
# crop images around their landmarks
for i in images:
    i.crop_to_landmarks_proportion(crop_boundary, group=group, label=label)

# TODO:
if reference_landmarks is None:
    print '- Compute reference shape'
    # extract original shapes
    shapes = [i.landmarks[group][label].lms for i in images]
    # define reference shape
    reference_landmarks = mean_pointcloud(shapes)

# TODO:
print '- Rescaling images to reference shape'
# rescale images so that the scale of their corresponding shapes matches
# the scale of the reference shape
images = [rescale_to_reference_landmarks(i, reference_landmarks, scale=scale,
                                         group=group, label=label,
                                         interpolator=interpolator)
          for i in images]


from pybug.activeappearancemodel.regression import ParametricRegressor

aam.initialize_lk(n_shape=[12, 3, 3], n_appearance=[150, 800, 250])

pr = ParametricRegressor(aam._lk_pyramid[-1].appearance_model,
                         aam._lk_pyramid[-1].transform, order=1,
                         features='probabilistic', noise_std=0.01,
                         n_perturbations=50)

landmarks = [i.landmarks['PTS'].lms for i in images]
pr.train(images, landmarks)
