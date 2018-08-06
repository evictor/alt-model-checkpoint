# alt-model-checkpoint

An adapter callback for Keras [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) that allows checkpointing
an alternate model (often submodel of a multi-GPU model).

## Installation

```bash
pip install alt-model-checkpoint
```

## Usage

The import you need is:

```python
from alt_model_checkpoint import AltModelCheckpoint
```

Common usage involving multi-GPU models built with Keras `multi_gpu_model()`:

```python
from alt_model_checkpoint import AltModelCheckpoint
from keras.models import Model
from keras.utils import multi_gpu_model

base_model = Model(...)
gpu_model = multi_gpu_model(base_model)
gpu_model.compile(...)

gpu_model.fit(..., callbacks=[
    AltModelCheckpoint('save/path/for/model.hdf5', base_model)
])
```

## Environment setup

1. Install [pipenv](https://docs.pipenv.org/install/).
2. Run `make`.

## Running tests

Run `make test`.