"""
This version of AltModelCheckpoint is linked against the Keras bundled with Tensorflow. If you are using a
non-Tensorflow backend with Keras, you should use the AltModelCheckpoint provided in `alt_model_checkpoint.keras`.
"""

import tensorflow as tf

from alt_model_checkpoint import _define_alt_model_checkpoint

AltModelCheckpoint = _define_alt_model_checkpoint(tf.keras.callbacks.ModelCheckpoint)
