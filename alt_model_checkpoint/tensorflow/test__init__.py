import tensorflow as tf

from alt_model_checkpoint.tensorflow import AltModelCheckpoint
from alt_model_checkpoint.test__init__ import CommonAltModelCheckpointTests


class TfKerasAltModelCheckpointTest(CommonAltModelCheckpointTests):
    def setUp(self):
        self.cls = AltModelCheckpoint
        self.model_cls = tf.keras.Model

    def test_base_cls(self):
        self.assertIsInstance(AltModelCheckpoint('foobar', None), tf.keras.callbacks.ModelCheckpoint)
