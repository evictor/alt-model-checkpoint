import keras
from keras import Model

from alt_model_checkpoint.keras import AltModelCheckpoint
from alt_model_checkpoint.test__init__ import CommonAltModelCheckpointTests


class KerasAltModelCheckpointTest(CommonAltModelCheckpointTests):
    def setUp(self):
        self.cls = AltModelCheckpoint
        self.model_cls = Model

    def test_base_cls(self):
        self.assertIsInstance(AltModelCheckpoint('foobar', None), keras.callbacks.ModelCheckpoint)
