import numpy as np
from pybug.transform import SimilarityTransform, AffineTransform, Translation

a = SimilarityTransform.from_vector(np.array([1,1,1,1]))
b = AffineTransform.from_vector(np.array([1,1,1,1,1,1]))
c = Translation(np.array([1,1]))

d = c.compose(c)