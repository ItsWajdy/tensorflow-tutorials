import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys

import matplotlib.pyplot as plt
from IPython.display import clear_output

tf.enable_eager_execution()

models_path = os.path.join(os.getcwd(), 'models')
sys.path.append(models_path)

from official.wide_deep import census_dataset
from official.wide_deep import census_main

census_dataset.download("/tmp/census_data")
