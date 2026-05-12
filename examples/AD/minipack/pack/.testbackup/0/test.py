import os
# ensure that the environment is at current folder
os.system("export PYTHONPATH=.")

'''This is a empty template to be added codes for user to test the solver and BCs.'''
from jax import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import *
from src.lattice import LatticeD2Q9
from src.utils import *

from jax.experimental.multihost_utils import process_allgather


# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'


#<BEG> LLM added <\BEG>

import os
os.system("export PYTHONPATH=.")
from jax import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import ADMSim
from src.lattice import LatticeD2Q9
from src.utils import *

from jax.experimental.multihost_utils import process_allgather

# Use 8 CPU devices (optional)
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

class ADMTestSim(ADMSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        """Initialize the scalar field phi as a Gaussian distribution."""
        x_center = self.nx // 2
        y_center = self.ny // 2
        sigma = 10.0
        x = jnp.arange(self.nx)
        y = jnp.arange(self.ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        phi = jnp.exp(-((X - x_center)**2 + (Y - y_center)**2) / (2 * sigma**2))
        # Return phi and the constant velocity field (self.u)
        return phi[..., jnp.newaxis], self.u

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    nx = 200
    ny = 200

    # Parameters
    alpha = 0.1  # Diffusion coefficient
    prescribed_vel = 0.1  # x-component of velocity
    u = jnp.zeros((nx, ny, 2), dtype=lattice.precisionPolicy.compute_dtype)
    u = u.at[:, :, 0].set(prescribed_vel)  # Set x-component to 0.1

    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'alpha': alpha,
        'u': u,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100,
        'timesteps': 1000,
        'restore_checkpoint': False,
    }

    sim = ADMTestSim(**kwargs)
    nsf = sim.assign_fields_sharded()  # Initialize distribution functions
    timestep = 0

    for i in range(kwargs.get('timesteps')):
        nsf, _ = sim.step(nsf, timestep, False)

        if i % kwargs.get('print_info_rate') == 0:
            print(f"Timestep {i}")

        if i % kwargs.get("io_rate") == 0:
            # Compute macroscopic variables
            phi, _ = sim.update_macroscopic(nsf)
            phi = downsample_field(phi, sim.downsamplingFactor)
            
            # Gather data from all processes
            phi = process_allgather(phi)
            phi = np.array(phi[1:-1, 1:-1, 0])  # Remove halo and convert to numpy

            # Save image
            save_image(timestep, phi, prefix='phi_')

            # Save VTK file
            fields = {"phi": phi}
            save_fields_vtk(timestep, fields, prefix='phi')

        timestep += 1

#<END> LLM added <\END>

