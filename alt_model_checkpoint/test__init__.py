import unittest
from unittest import TestCase
from unittest.mock import Mock


class CommonAltModelCheckpointTests(TestCase):
    cls = None

    def test_kwargs_pass_through(self):
        if type(self) is CommonAltModelCheckpointTests:
            return unittest.skip('base class test skip')

        callback = self.cls('path/to/model.hdf5', None, monitor='foobar')
        self.assertEqual(callback.filepath, 'path/to/model.hdf5')
        self.assertEqual(callback.monitor, 'foobar')

    def test_on_epoch_end(self):
        if type(self) is CommonAltModelCheckpointTests:
            return unittest.skip('base class test skip')

        model1 = Mock()

        model2 = Mock()
        model2.save = Mock()

        callback = self.cls('path/to/model.hdf5', model2)
        callback.model = model1

        callback.on_epoch_end(42)
        self.assertIs(callback.model, model1, 'original model is restored')

        # model2 saved
        model2.save.assert_called_once_with('path/to/model.hdf5', overwrite=True)
