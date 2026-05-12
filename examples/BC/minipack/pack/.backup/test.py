import os
# ensure that the environment is at current folder
os.system("export PYTHONPATH=.")

# we only use one gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''This is a empty template to be added codes for user to test the solver and BCs.'''
from jax import config
import jax
print("GPU: ",jax.local_devices())
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import *
from src.lattice import LatticeD2Q9
from src.utils import *

from jax.experimental.multihost_utils import process_allgather


# Use 8 CPU devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
