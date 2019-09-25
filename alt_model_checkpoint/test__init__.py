import unittest
from typing import Callable, Union
from unittest import TestCase
from unittest.mock import Mock


class CommonAltModelCheckpointTests(TestCase):
    cls = None

    def run(self, *args, **kwargs) -> Union[TestCase, Callable]:
        # Don't run tests on naked base class
        if type(self) is CommonAltModelCheckpointTests:
            return unittest.skip('base class test skip')

        return super().run(*args, **kwargs)

    def test_kwargs_pass_through(self):
        callback = self.cls('path/to/model.hdf5', None, monitor='foobar')
        self.assertEqual(callback.filepath, 'path/to/model.hdf5')
        self.assertEqual(callback.monitor, 'foobar')

    def test_on_epoch_end(self):
        model1 = Mock()

        model2 = Mock()
        model2.save = Mock()

        callback = self.cls('path/to/model.hdf5', model2)
        callback.model = model1

        callback.on_epoch_end(42)
        self.assertIs(callback.model, model1, 'original model is restored')

        # model2 saved
        model2.save.assert_called_once_with('path/to/model.hdf5', overwrite=True)
