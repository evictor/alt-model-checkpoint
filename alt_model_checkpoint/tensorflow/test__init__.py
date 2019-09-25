import tensorflow
from tensorflow.python.keras import Model

from alt_model_checkpoint.tensorflow import AltModelCheckpoint
from alt_model_checkpoint.test__init__ import CommonAltModelCheckpointTests


class TfKerasAltModelCheckpointTest(CommonAltModelCheckpointTests):
    def setUp(self):
        self.cls = AltModelCheckpoint
        self.model_cls = Model

    def test_base_cls(self):
        self.assertIsInstance(AltModelCheckpoint('foobar', None), tensorflow.keras.callbacks.ModelCheckpoint)
