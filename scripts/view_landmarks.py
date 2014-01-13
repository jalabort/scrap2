from pybug.io import auto_import
from pybug.landmark.labels import labeller, ibug_68_points

images = auto_import('/Users/joan/PhD/DataBases/lfpw/trainset/*.png',
                     max_images=1)

# crop images around their landmarks
for i in images:
    i.crop_to_landmarks_proportion(0.2)

labeller(images, 'PTS', ibug_68_points)

images[0].landmarks['ibug_68_points'].view()