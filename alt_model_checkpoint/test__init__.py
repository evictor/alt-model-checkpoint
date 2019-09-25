import unittest
from typing import Callable, Union
from unittest import TestCase
from unittest.mock import Mock


class CommonAltModelCheckpointTests(TestCase):
    cls = None
    model_cls = None

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
        multigpu_model = Mock()
        multigpu_model_optimizer = Mock()
        multigpu_model.optimizer = multigpu_model_optimizer

        template_model = Mock()

        # noinspection PyUnusedLocal
        def save_impl_mock(*args, **kwargs):
            self.assertIs(template_model.optimizer, multigpu_model_optimizer,
                          'original optimizer is saved with template model')
        template_model.save = Mock(side_effect=save_impl_mock)

        template_model_optimizer = Mock()
        template_model.optimizer = template_model_optimizer

        callback = self.cls('path/to/model.hdf5', template_model)
        callback.model = multigpu_model

        callback.on_epoch_end(42)
        self.assertIs(callback.model, multigpu_model, 'original model is restored')
        self.assertIs(template_model.optimizer, template_model_optimizer, 'template model optimizer state restored')

        # template model is saved
        template_model.save.assert_called_once_with('path/to/model.hdf5', overwrite=True)

    def test_can_opt_out_of_inherited_optimizer(self):
        multigpu_model = Mock()
        multigpu_model_optimizer = Mock()
        multigpu_model.optimizer = multigpu_model_optimizer

        template_model = Mock()
        template_model_optimizer = Mock()
        template_model.optimizer = template_model_optimizer

        # noinspection PyUnusedLocal
        def save_impl_mock(*args, **kwargs):
            self.assertIs(template_model.optimizer, template_model_optimizer,
                          'template model optimizer is saved with template model')

        template_model.save = Mock(side_effect=save_impl_mock)

        callback = self.cls('path/to/model.hdf5', template_model, inherit_optimizer=False)
        callback.model = multigpu_model

        callback.on_epoch_end(42)
        self.assertIs(callback.model, multigpu_model, 'original model is restored')
        self.assertIs(template_model.optimizer, template_model_optimizer, 'template model optimizer is preserved')

        # template model is saved
        template_model.save.assert_called_once_with('path/to/model.hdf5', overwrite=True)
