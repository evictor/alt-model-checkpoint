"""
This version of AltModelCheckpoint is linked against Keras directly. If you are using Tensorflow backend with Keras, you
will most likely want the version of AltModelCheckpoint in `alt_model_checkpoint.tensorflow`.
"""

from keras.callbacks import ModelCheckpoint

from alt_model_checkpoint import _define_alt_model_checkpoint

AltModelCheckpoint = _define_alt_model_checkpoint(ModelCheckpoint)
