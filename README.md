# alt-model-checkpoint

An adapter callback for Keras [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) that allows checkpointing
an alternate model (often submodel of a multi-GPU model).

## Installation

```bash
pip install alt-model-checkpoint
```

## Usage

*You must provide your own Keras or Tensorflow installation.* See `Pipfile` for preferred versions.


If using the Keras bundled in Tensorflow:

```python
from alt_model_checkpoint.tensorflow import AltModelCheckpoint
```

If using Keras standalone:

```python
from alt_model_checkpoint.keras import AltModelCheckpoint
```

Common usage involving multi-GPU models built with Keras `multi_gpu_model()`:

```python
from alt_model_checkpoint.keras import AltModelCheckpoint
from keras.models import Model
from keras.utils import multi_gpu_model

def compile_model(m):
    """Implement with your model compile logic; both base and GPU models should be compiled identically"""
    m.compile(...)

base_model = Model(...)
gpu_model = multi_gpu_model(base_model)
compile_model(base_model)
compile_model(gpu_model)

gpu_model.fit(..., callbacks=[
    AltModelCheckpoint('save/path/for/model.hdf5', base_model)
])
```

## Constructor args

### filepath

Model save file path; see [underlying ModelCheckpoint docs](https://keras.io/callbacks/#modelcheckpoint) for details.

### alternate_model

Keras model to save instead of the default. This is used especially when training multi-gpu models built with Keras
multi_gpu_model(). In that case, you would pass the original "template model" to be saved each checkpoint.

### inherit_optimizer

If TRUE (default), saves the optimizer of the base model (e.g. a multi-gpu model) with the alternate model. This is
necessary if you want to be able to resume training on a saved alternate model. If FALSE, the alternate model's
optimizer will be saved as-is.

### *args, **kwargs

These are passed as-is to the underlying [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) constructor.

## Dev environment setup

1. Install [pipenv](https://docs.pipenv.org/install/).
2. Run `make test` (runs `make test-build` automatically to ensure deps)
