from jax import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9
from src.utils import *

from jax.experimental.multihost_utils import process_allgather
# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

class Cavity(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel # velocity would be defined below
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    nx = 200
    ny = 200

    Re = 200.0
    prescribed_vel = 0.1
    clength = nx - 1

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 10,
        'timesteps': 1000,
        'restore_checkpoint': False,
    }

    sim = Cavity(**kwargs)
    nsf = sim.assign_fields_sharded() # create the initialized field
    timestep = 0
    for i in range(kwargs.get('timesteps')):
        nsf, nsfstar = sim.step(nsf, timestep, False)
        
        if i%kwargs.get('print_info_rate') == 0:
            print(i)

        if i%kwargs.get("io_rate")==0:
            rho_prev, u_prev = sim.update_macroscopic(nsf)
            rho_prev = downsample_field(rho_prev, sim.downsamplingFactor)
            u_prev = downsample_field(u_prev, sim.downsamplingFactor)
            
            # Gather the data from all processes and convert it to numpy arrays (move to host memory)
            rho_prev = process_allgather(rho_prev)
            u_prev = process_allgather(u_prev)
            rho = np.array(rho_prev[1:-1, 1:-1])

            # save the image 
            u = np.array(u_prev[1:-1, 1:-1, :])
            save_image(timestep, u)

            # save the fields in .vtk file
            # the .vtk file could be processed by the installed pyvista lib
            fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
            save_fields_vtk(timestep, fields)
        timestep = timestep + 1