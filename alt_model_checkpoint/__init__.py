from typing import Union

# Keep for type hints/IDE
# noinspection PyUnreachableCode
if False:
    import tensorflow
    import keras


def _define_alt_model_checkpoint(parent_cls: Union['tensorflow.keras.callbacks.ModelCheckpoint',
                                                   'keras.callbacks.ModelCheckpoint']):
    class AltModelCheckpoint(parent_cls):
        def __init__(self, filepath, alternate_model, **kwargs):
            """
            Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.

            :param filepath:
            :param alternate_model: Keras model to save instead of the default. This is used especially when training multi-
                                    gpu models built with Keras multi_gpu_model(). In that case, you would pass the original
                                    "template model" to be saved each checkpoint.
            :param kwargs:          Passed to ModelCheckpoint.
            """

            self.alternate_model = alternate_model
            super().__init__(filepath, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            model_before = self.model
            self.model = self.alternate_model
            super().on_epoch_end(epoch, logs)
            # noinspection PyAttributeOutsideInit
            self.model = model_before

    return AltModelCheckpoint
