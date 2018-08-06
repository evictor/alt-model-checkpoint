from unittest import TestCase
from unittest.mock import Mock
from submodel_checkpoint import SubmodelCheckpoint


class SubmodelCheckpointTest(TestCase):
    def test_kwargs_pass_through(self):
        callback = SubmodelCheckpoint('path/to/model.hdf5', None, monitor='foobar')
        self.assertEqual(callback.filepath, 'path/to/model.hdf5')
        self.assertEqual(callback.monitor, 'foobar')

    def test_on_epoch_end(self):
        model1 = Mock()

        model2 = Mock()
        model2.save = Mock()

        callback = SubmodelCheckpoint('path/to/model.hdf5', model2)
        callback.model = model1

        callback.on_epoch_end(42)
        self.assertIs(callback.model, model1, 'original model is restored')

        # model2 saved
        model2.save.assert_called_once_with('path/to/model.hdf5', overwrite=True)
