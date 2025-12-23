"""Callback configuration for gin.

This module registers Lightning's ModelCheckpoint as a gin configurable,
allowing direct configuration via gin files.
"""

import gin
from lightning.pytorch.callbacks import ModelCheckpoint

# Register Lightning's ModelCheckpoint with gin so it can be configured directly
gin.external_configurable(ModelCheckpoint)
