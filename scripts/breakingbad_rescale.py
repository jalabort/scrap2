from pybug import data_path_to
from pybug.io import auto_import

breakingbad = auto_import(data_path_to('breakingbad.jpg'))[0]

breakingbad.rescale(0.1)