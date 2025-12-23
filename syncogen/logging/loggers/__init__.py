"""Logger configuration for gin.

This module registers Lightning's WandbLogger as a gin configurable,
allowing direct configuration via gin files.
"""

import gin
from lightning.pytorch.loggers import WandbLogger

# Register Lightning's WandbLogger with gin so it can be configured directly
gin.external_configurable(WandbLogger)
