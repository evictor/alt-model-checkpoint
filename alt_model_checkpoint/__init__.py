from typing import Union, Type

# Keep for type hints/IDE
# noinspection PyUnreachableCode
if False:
    import tensorflow
    import keras


def _define_alt_model_checkpoint(parent_cls: Union[Type['tensorflow.keras.callbacks.ModelCheckpoint'],
                                                   Type['keras.callbacks.ModelCheckpoint']]):
    class AltModelCheckpoint(parent_cls):
        def __init__(self, filepath, alternate_model, inherit_optimizer: bool = True, **kwargs):
            """
            Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are
            accepted.

            :param filepath:
            :param alternate_model: Keras model to save instead of the default. This is used especially when training
                                    multi-gpu models built with Keras multi_gpu_model(). In that case, you would pass
                                    the original "template model" to be saved each checkpoint.
            :param inherit_optimizer: If TRUE (default), saves the optimizer of the base model (e.g. a multi-gpu
                                      model) with the alternate model. This is necessary if you want to be able to
                                      resume training on a saved alternate model. If FALSE, the alternate model's
                                      optimizer will be saved as-is.
            :param kwargs:          Passed to ModelCheckpoint.
            """

            self.alternate_model = alternate_model
            self.inherit_optimizer = inherit_optimizer
            super().__init__(filepath, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            alt_model_optimizer_before = None
            if self.inherit_optimizer:
                # Swap the trained optimizer into the model to be saved
                alt_model_optimizer_before = self.alternate_model.optimizer
                self.alternate_model.optimizer = self.model.optimizer

            # Swap the model to be saved into position
            model_before = self.model
            self.model = self.alternate_model

            super().on_epoch_end(epoch, logs)

            # Restore state
            # noinspection PyAttributeOutsideInit
            self.model = model_before

            if self.inherit_optimizer:
                self.alternate_model.optimizer = alt_model_optimizer_before

    return AltModelCheckpoint
