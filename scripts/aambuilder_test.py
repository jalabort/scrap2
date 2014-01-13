from __future__ import division
import numpy as np
from skimage.transform import pyramid_gaussian
from pybug.io import auto_import
from pybug.shape import PointCloud
from pybug.landmark.labels import labeller, ibug_68_points, ibug_68_trimesh
from pybug.transform import Scale, Translation
from pybug.transform.affine import UniformScale
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.groupalign import GeneralizedProcrustesAnalysis
from pybug.image import MaskedNDImage, RGBImage
from pybug.model import PCAModel


def aam_builder(path, max_images=None, group='PTS', label='all',
                crop_boundary=0.2, interpolator='scipy',
                scale=1, max_shape_components=25, reference_frame_boundary=3,
                labels=[], triangulation_func=ibug_68_trimesh,
                triangulation_label='ibug_68_trimesh',
                n_multiresolution_levels=3,
                transform_cls=PiecewiseAffineTransform,
                max_appearance_components=250):

    # TODO:
    print '- Loading images from:' + path
    # load images
    images = auto_import(path, max_images=max_images)
    # convert to greyscale
    images = [i.as_greyscale() if type(i) is RGBImage else i for i in images]
    # crop images around their landmarks
    for i in images:
        i.crop_to_landmarks_proportion(crop_boundary, group=group, label=label)

    # TODO:
    print '- Normalizing object scales'
    # extract shapes
    shapes = [i.landmarks[group][label].lms for i in images]
    # define reference shape
    reference_shape = PointCloud(np.mean([s.points for s in shapes], axis=0))
    # compute scale difference between all shapes and reference shape
    scales = [UniformScale.align(s, reference_shape).as_vector()
              for s in shapes]
    # rescale all images using previous scales
    images = [i.rescale(s, interpolator=interpolator)
              for i, s in zip(images, scales)]
    # extract rescaled shapes
    shapes = [i.landmarks[group][label].lms for i in images]

    # TODO:
    pyramid_iterator = [pyramid_gaussian(i.pixels) for i in images]

    # TODO:
    print '- Building shape model'
    # centralize shapes
    centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source for s in gpa.transforms]

    # TODO:
    # scale shape if necessary
    if scale is not 1:
        aligned_shapes = [Scale(scale, n_dims=reference_shape.n_dims).apply(s)
                          for s in aligned_shapes]
    # build shape model
    shape_model = PCAModel(aligned_shapes)
    # trim shape model if required
    if max_shape_components is not None:
        shape_model.trim_components(max_shape_components)

    # TODO:
    print '- Building reference frame'
    # scale reference shape if necessary
    if scale is not 1:
        reference_shape = Scale(
            scale, n_dims=reference_shape.n_dims).apply(reference_shape)
    # compute lower bound
    lower_bound = reference_shape.bounds(
        boundary=reference_frame_boundary)[0]
    # translate reference shape using lower bound
    reference_landmarks = Translation(-lower_bound).apply(
        reference_shape)
    # compute reference frame resolution
    reference_resolution = reference_landmarks.range(
        boundary=reference_frame_boundary)
    # build reference frame
    reference_frame = MaskedNDImage.blank(reference_resolution)
    # assign landmarks using the default group
    reference_frame.landmarks[group] = reference_landmarks
    # label reference frame
    for l in labels:
        labeller([reference_frame], group, l)
    # check for precomputed triangulation
    if triangulation_func is not None:
        labeller([reference_frame], group, triangulation_func)
        trilist = reference_frame.landmarks[triangulation_label].lms.trilist
    else:
        trilist = None
    # mask reference frame
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    # TODO:
    print '- Building Apperance Models'
    # extract landmarks
    landmarks = [i.landmarks[group][label].lms for i in images]
    # initialize list of appearance models
    appearance_model_list = []

    # for each level
    for j in range(0, n_multiresolution_levels):
        print ' - Level {}'.format(j)
        # obtain images
        level_images = [MaskedNDImage(p.next()) for p in
                        pyramid_iterator]
        # rescale and reassign landmarks if necessary
        for li, i in zip(level_images, images):
            li.landmarks[group] = i.landmarks[group]
            li.landmarks[group].lms.points = (i.landmarks[group].lms.points /
                                              (2 ** j))
        # mask level_images
        for i in level_images:
            i.constrain_mask_to_landmarks(group=group, trilist=trilist)

        # compute features
        for i in level_images:
            i.normalize_inplace()
        feature_images = level_images

        # compute transforms
        transforms = [transform_cls(reference_frame.landmarks[group].lms,



                                    i.landmarks[group].lms)
                      for i in feature_images]
        # warp images
        warped_images = [i.warp_to(reference_frame.mask, t,
                                   warp_landmarks=False,
                                   interpolator=interpolator)
                         for i, t in zip(feature_images, transforms)]
        # assign reference landmarks using the default group
        for i in warped_images:
            i.landmarks[group] = reference_landmarks
        # label reference frame
        for l in labels:
            labeller([reference_frame], group, l)
        # check for precomputed triangulation
        if triangulation_func is not None:
            labeller([reference_frame], group, triangulation_func)
        # mask warped images
        for i in warped_images:
            i.constrain_mask_to_landmarks(group=group, trilist=trilist)

        # build appearance model
        appearance_model = PCAModel(warped_images)
        # trim apperance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)
        # add appearance model to the list
        appearance_model_list.append(appearance_model)

    return shape_model, reference_frame, appearance_model_list


options = {'max_images': 500,
           'max_shape_components': 5,
           'max_appearance_components': 5}

aam = aam_builder('/Users/joan/PhD/DataBases/lfpw/trainset/*.png', **options)