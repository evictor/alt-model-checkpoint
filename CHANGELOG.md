# 2.0.0

Semantic versioning can't be maintained while trying to track version numbers to TF; that didn't last long. :)

This is a major version bump because the default behavior changes with introduction of `inherit_optimizer` arg to
`AltModelCheckpoint(...)`. When `inherit_optimizer` is `True` (the default), the optimizer of the base model (e.g. the
multi-gpu model) will be saved with the alternate model so training can be resumed later from the saved file.

Pass `inherit_optimizer=False` to preserve old behavior (save alternate model optimizer as-is, which is not typically
useful for the common multi-gpu model use case of this library).

# 1.13.0

* **BREAKING:** Tensorflow/Keras dependencies not enforced since user can use either
    * How to migrate: specify your own version in requirements.txt or Pipfile and install accordingly
* **BREAKING:** Provide versions of AltModelCheckpoint linked against Keras standalone and TF packaged Keras
    * Update your imports from `alt_model_checkpoint.AltModelCheckpoint` to
        alt_model_checkpoint.[tensorflow|keras].AltModelCheckpoint`
* Fix TF version to ~1.13 and associated Keras version (~2.2.4) according to breaking changes in upstream
  ModelCheckpoint
* Match package minor version to TF minor version

# 1.0.2

Upgrade deps

# 1.0.1

Upgrade dependencies, esp. for requests==2.20.0 security patch

# 1.0.0

Initial release