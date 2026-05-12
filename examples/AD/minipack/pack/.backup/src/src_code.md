## src/base.py 
 ```python
# Standard Libraries
import os
import time

# Third-Party Libraries
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from termcolor import colored

# JAX-related imports
from jax import jit, lax, vmap
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec, PositionalSharding, Mesh
import orbax.checkpoint as orb

# functools imports
from functools import partial

# Local/Custom Libraries
from src.utils import downsample_field

jax.config.update("jax_spmd_mode", 'allow_all')
# Disables annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class LBMBase(object):
    """
    LBMBase: A class that represents a base for Lattice Boltzmann Method simulation.
    
    Parameters
    ----------
        lattice (object): The lattice object that contains the lattice structure and weights.
        omega (float): The relaxation parameter for the LBM simulation.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        nz (int, optional): Number of grid points in the z-direction. Defaults to 0.
        precision (str, optional): A string specifying the precision used for the simulation. Defaults to "f32/f32".
    """
    
    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz")

        self.precision = kwargs.get("precision")
        computedType, storedType = self.set_precisions(self.precision)
        self.precisionPolicy = jmp.Policy(compute_dtype=computedType,
                                            param_dtype=computedType, output_dtype=storedType)
        
        self.lattice = kwargs.get("lattice")
        self.checkpointRate = kwargs.get("checkpoint_rate", 0)
        self.checkpointDir = kwargs.get("checkpoint_dir", './checkpoints')
        self.downsamplingFactor = kwargs.get("downsampling_factor", 1)
        self.printInfoRate = kwargs.get("print_info_rate", 100)
        self.ioRate = kwargs.get("io_rate", 0)
        self.returnFpost = kwargs.get("return_fpost", False)
        self.computeMLUPS = kwargs.get("compute_MLUPS", False)
        self.restore_checkpoint = kwargs.get("restore_checkpoint", False)
        self.nDevices = jax.device_count()
        self.backend = jax.default_backend()

        if self.computeMLUPS:
            self.restore_checkpoint = False
            self.ioRate = 0
            self.checkpointRate = 0
            self.printInfoRate = 0

        # Check for distributed mode
        if self.nDevices > jax.local_device_count():
            print("WARNING: Running in distributed mode. Make sure that jax.distributed.initialize is called before performing any JAX computations.")
                    
        self.c = self.lattice.c
        self.q = self.lattice.q
        self.w = self.lattice.w
        self.dim = self.lattice.d

        # Set the checkpoint manager
        if self.checkpointRate > 0:
            mngr_options = orb.CheckpointManagerOptions(save_interval_steps=self.checkpointRate, max_to_keep=1)
            self.mngr = orb.CheckpointManager(self.checkpointDir, orb.PyTreeCheckpointer(), options=mngr_options)
        else:
            self.mngr = None
        
        # Adjust the number of grid points in the x direction, if necessary.
        # If the number of grid points is not divisible by the number of devices
        # it increases the number of grid points to the next multiple of the number of devices.
        # This is done in order to accommodate the domain sharding per XLA device
        nx, ny, nz = kwargs.get("nx"), kwargs.get("ny"), kwargs.get("nz")
        if None in {nx, ny, nz}:
            raise ValueError("nx, ny, and nz must be provided. For 2D examples, nz must be set to 0.")
        self.nx = nx
        if nx % self.nDevices:
            self.nx = nx + (self.nDevices - nx % self.nDevices)
            print("WARNING: nx increased from {} to {} in order to accommodate domain sharding per XLA device.".format(nx, self.nx))
        self.ny = ny
        self.nz = nz

        self.show_simulation_parameters()
    
        # Store grid information
        self.gridInfo = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dim": self.lattice.d,
            "lattice": self.lattice
        }

        P = PartitionSpec

        # Define the right permutation
        self.rightPerm = [(i, (i + 1) % self.nDevices) for i in range(self.nDevices)]
        # Define the left permutation
        self.leftPerm = [((i + 1) % self.nDevices, i) for i in range(self.nDevices)]

        # Set up the sharding and streaming for 2D and 3D simulations
        if self.dim == 2:
            self.devices = mesh_utils.create_device_mesh((self.nDevices, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "value"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "value"))

            self.streaming = jit(shard_map(self.streaming_m, mesh=self.mesh,
                                                      in_specs=P("x", None, None), out_specs=P("x", None, None), check_rep=False))

        # Set up the sharding and streaming for 2D and 3D simulations
        elif self.dim == 3:
            self.devices = mesh_utils.create_device_mesh((self.nDevices, 1, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "z", "value"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "z", "value"))

            self.streaming = jit(shard_map(self.streaming_m, mesh=self.mesh,
                                                      in_specs=P("x", None, None, None), out_specs=P("x", None, None, None), check_rep=False))

        else:
            raise ValueError(f"dim = {self.dim} not supported")
        
        # Compute the bounding box indices for boundary conditions
        self.boundingBoxIndices= self.bounding_box_indices()
        # Create boundary data for the simulation
        self._create_boundary_data()
        self.force = self.get_force()

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        if value is None:
            raise ValueError("Lattice type must be provided.")
        if self.nz == 0 and value.name not in ['D2Q9']:
            raise ValueError("For 2D simulations, lattice type must be LatticeD2Q9.")
        if self.nz != 0 and value.name not in ['D3Q19', 'D3Q27']:
            raise ValueError("For 3D simulations, lattice type must be LatticeD3Q19, or LatticeD3Q27.")
                            
        self._lattice = value

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if value is None:
            raise ValueError("omega must be provided")
        if not isinstance(value, float):
            raise TypeError("omega must be a float")
        self._omega = value

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is None:
            raise ValueError("nx must be provided")
        if not isinstance(value, int):
            raise TypeError("nx must be an integer")
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is None:
            raise ValueError("ny must be provided")
        if not isinstance(value, int):
            raise TypeError("ny must be an integer")
        self._ny = value

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value is None:
            raise ValueError("nz must be provided")
        if not isinstance(value, int):
            raise TypeError("nz must be an integer")
        self._nz = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if not isinstance(value, str):
            raise TypeError("precision must be a string")
        self._precision = value

    @property
    def checkpointRate(self):
        return self._checkpointRate

    @checkpointRate.setter
    def checkpointRate(self, value):
        if not isinstance(value, int):
            raise TypeError("checkpointRate must be an integer")
        self._checkpointRate = value

    @property
    def checkpointDir(self):
        return self._checkpointDir

    @checkpointDir.setter
    def checkpointDir(self, value):
        if not isinstance(value, str):
            raise TypeError("checkpointDir must be a string")
        self._checkpointDir = value

    @property
    def downsamplingFactor(self):
        return self._downsamplingFactor

    @downsamplingFactor.setter
    def downsamplingFactor(self, value):
        if not isinstance(value, int):
            raise TypeError("downsamplingFactor must be an integer")
        self._downsamplingFactor = value

    @property
    def printInfoRate(self):
        return self._printInfoRate

    @printInfoRate.setter
    def printInfoRate(self, value):
        if not isinstance(value, int):
            raise TypeError("printInfoRate must be an integer")
        self._printInfoRate = value

    @property
    def ioRate(self):
        return self._ioRate

    @ioRate.setter
    def ioRate(self, value):
        if not isinstance(value, int):
            raise TypeError("ioRate must be an integer")
        self._ioRate = value

    @property
    def returnFpost(self):
        return self._returnFpost

    @returnFpost.setter
    def returnFpost(self, value):
        if not isinstance(value, bool):
            raise TypeError("returnFpost must be a boolean")
        self._returnFpost = value

    @property
    def computeMLUPS(self):
        return self._computeMLUPS

    @computeMLUPS.setter
    def computeMLUPS(self, value):
        if not isinstance(value, bool):
            raise TypeError("computeMLUPS must be a boolean")
        self._computeMLUPS = value

    @property
    def restore_checkpoint(self):
        return self._restore_checkpoint

    @restore_checkpoint.setter
    def restore_checkpoint(self, value):
        if not isinstance(value, bool):
            raise TypeError("restore_checkpoint must be a boolean")
        self._restore_checkpoint = value

    @property
    def nDevices(self):
        return self._nDevices

    @nDevices.setter
    def nDevices(self, value):
        if not isinstance(value, int):
            raise TypeError("nDevices must be an integer")
        self._nDevices = value

    def show_simulation_parameters(self):
        attributes_to_show = [
            'omega', 'nx', 'ny', 'nz', 'dim', 'precision', 'lattice', 
            'checkpointRate', 'checkpointDir', 'downsamplingFactor', 
            'printInfoRate', 'ioRate', 'computeMLUPS', 
            'restore_checkpoint', 'backend', 'nDevices'
        ]

        descriptive_names = {
            'omega': 'Omega',
            'nx': 'Grid Points in X',
            'ny': 'Grid Points in Y',
            'nz': 'Grid Points in Z',
            'dim': 'Dimensionality',
            'precision': 'Precision Policy',
            'lattice': 'Lattice Type',
            'checkpointRate': 'Checkpoint Rate',
            'checkpointDir': 'Checkpoint Directory',
            'downsamplingFactor': 'Downsampling Factor',
            'printInfoRate': 'Print Info Rate',
            'ioRate': 'I/O Rate',
            'computeMLUPS': 'Compute MLUPS',
            'restore_checkpoint': 'Restore Checkpoint',
            'backend': 'Backend',
            'nDevices': 'Number of Devices'
        }
        simulation_name = self.__class__.__name__
        
        print(colored(f'**** Simulation Parameters for {simulation_name} ****', 'green'))
                
        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        print(header)
        print('-' * 50)
        
        for attr in attributes_to_show:
            value = getattr(self, attr, 'Attribute not set')
            descriptive_name = descriptive_names.get(attr, attr)  # Use the attribute name as a fallback
            row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
            print(row)

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.BCs = []
        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_halo_list = [np.array(bc.indices).T for bc in self.BCs if bc.isSolid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

        # Create the grid mask on each process
        start = time.time()
        grid_mask = self.create_grid_mask(solid_halo_voxels)
        print("Time to create the grid mask:", time.time() - start)

        start = time.time()
        for bc in self.BCs:
            assert bc.implementationStep in ['PostStreaming', 'PostCollision']
            bc.create_local_mask_and_normal_arrays(grid_mask)
        print("Time to create the local masks and normal arrays:", time.time() - start)

    # This is another non-JITed way of creating the distributed arrays. It is not used at the moment.
    # def distributed_array_init(self, shape, type, init_val=None):
    #     sharding_dim = shape[0] // self.nDevices
    #     sharded_shape = (self.nDevices, sharding_dim,  *shape[1:])
    #     device_shape = sharded_shape[1:]
    #     arrays = []

    #     for d, index in self.sharding.addressable_devices_indices_map(sharded_shape).items():
    #         jax.default_device = d
    #         if init_val is None:
    #             x = jnp.zeros(shape=device_shape, dtype=type)
    #         else:
    #             x = jnp.full(shape=device_shape, fill_value=init_val, dtype=type)  
    #         arrays += [jax.device_put(x, d)] 
    #     jax.default_device = jax.devices()[0]
    #     return jax.make_array_from_single_device_arrays(shape, self.sharding, arrays)

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def distributed_array_init(self, shape, type, init_val=0, sharding=None):
        """
        Initialize a distributed array using JAX, with a specified shape, data type, and initial value.
        Optionally, provide a custom sharding strategy.

        Parameters
        ----------
            shape (tuple): The shape of the array to be created.
            type (dtype): The data type of the array to be created.
            init_val (scalar, optional): The initial value to fill the array with. Defaults to 0.
            sharding (Sharding, optional): The sharding strategy to use. Defaults to `self.sharding`.

        Returns
        -------
            jax.numpy.ndarray: A JAX array with the specified shape, data type, initial value, and sharding strategy.
        """
        if sharding is None:
            sharding = self.sharding
        x = jnp.full(shape=shape, fill_value=init_val, dtype=type)        
        return jax.lax.with_sharding_constraint(x, sharding)
    
    @partial(jit, static_argnums=(0,))
    def create_grid_mask(self, solid_halo_voxels):
        """
        This function creates a mask for the background grid that accounts for the location of the boundaries.
        
        Parameters
        ----------
            solid_halo_voxels: A numpy array representing the voxels in the halo of the solid object.
            
        Returns
        -------
            A JAX array representing the grid mask of the grid.
        """
        # Halo width (hw_x is different to accommodate the domain sharding per XLA device)
        hw_x = self.nDevices
        hw_y = hw_z = 1
        if self.dim == 2:
            grid_mask = self.distributed_array_init((self.nx + 2 * hw_x, self.ny + 2 * hw_y, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)  

            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

        elif self.dim == 3:
            grid_mask = self.distributed_array_init((self.nx + 2 * hw_x, self.ny + 2 * hw_y, self.nz + 2 * hw_z, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(hw_z, -hw_z), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                solid_halo_voxels = solid_halo_voxels.at[:, 2].add(hw_z)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)
            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

    def bounding_box_indices(self):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Returns
        -------
            boundingBox (dict): A dictionary where keys are the names of the bounding box faces
            ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
            are numpy arrays of indices corresponding to each face.
        """
        if self.dim == 2:
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(self.nx)].
            bounding_box = {"bottom": np.array([[i, 0] for i in range(self.nx)], dtype=int),
                           "top": np.array([[i, self.ny - 1] for i in range(self.nx)], dtype=int),
                           "left": np.array([[0, i] for i in range(self.ny)], dtype=int),
                           "right": np.array([[self.nx - 1, i] for i in range(self.ny)], dtype=int)}
                            
            return bounding_box

        elif self.dim == 3:
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(self.nx) for j in range(self.ny)].
            bounding_box = {
                "bottom": np.array([[i, j, 0] for i in range(self.nx) for j in range(self.ny)], dtype=int),
                "top": np.array([[i, j, self.nz - 1] for i in range(self.nx) for j in range(self.ny)],dtype=int),
                "left": np.array([[0, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "right": np.array([[self.nx - 1, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "front": np.array([[i, 0, k] for i in range(self.nx) for k in range(self.nz)], dtype=int),
                "back": np.array([[i, self.ny - 1, k] for i in range(self.nx) for k in range(self.nz)], dtype=int)}

            return bounding_box

    def set_precisions(self, precision):
        """
        This function sets the precision of the computations. The precision is defined by a pair of values,
        representing the precision of the computation and the precision of the storage, respectively.

        Parameters
        ----------
            precision (str): A string representing the desired precision. The string should be in the format
            "computation/storage", where "computation" and "storage" are either "f64", "f32", or "f16",
            representing 64-bit, 32-bit, or 16-bit floating point numbers, respectively.

        Returns
        -------
            tuple: A pair of jax.numpy data types representing the computation and storage precisions, respectively.
            If the input string does not match any of the predefined options, the function defaults to (jnp.float32, jnp.float32).
        """
        return {
            "f64/f64": (jnp.float64, jnp.float64),
            "f32/f32": (jnp.float32, jnp.float32),
            "f32/f16": (jnp.float32, jnp.float16),
            "f16/f16": (jnp.float16, jnp.float16),
            "f64/f32": (jnp.float64, jnp.float32),
            "f64/f16": (jnp.float64, jnp.float16),
        }.get(precision, (jnp.float32, jnp.float32))

    def initialize_macroscopic_fields(self):
        """
        This function initializes the macroscopic fields (density and velocity) to their default values.
        The default density is 1 and the default velocity is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
            None, None: The default density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        print("WARNING: Default initial conditions assumed: density = 1, velocity = 0")
        print("         To set explicit initial density and velocity, use self.initialize_macroscopic_fields.")
        return None, None

    def assign_fields_sharded(self):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the density (rho0) and velocity (u0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the 
        distribution functions (f).

        If the density or velocity are not provided, the function initializes the distribution functions with a default 
        value (self.w), representing density=1 and velocity=0. Otherwise, it uses the provided density and velocity to initialize the populations.

        Parameters
        ----------
        None

        Returns
        -------
        f: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the distribution functions for the simulation.
        """
        rho0, u0 = self.initialize_macroscopic_fields()

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)
    
        if rho0 is None or u0 is None:
            f = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=self.w)
        else:
            f = self.initialize_populations(rho0, u0)

        return f
    
    def initialize_populations(self, rho0, u0):
        """
        This function initializes the populations (distribution functions) for the simulation.
        It uses the equilibrium distribution function, which is a function of the macroscopic 
        density and velocity.

        Parameters
        ----------
        rho0: jax.numpy.ndarray
            The initial density field.
        u0: jax.numpy.ndarray
            The initial velocity field.

        Returns
        -------
        f: jax.numpy.ndarray
            The array holding the initialized distribution functions for the simulation.
        """
        return self.equilibrium(rho0, u0)

    def send_right(self, x, axis_name):
        """
        This function sends the data to the right neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
        jax.numpy.ndarray
            The data after being sent to the right neighboring process.
        """
        return lax.ppermute(x, perm=self.rightPerm, axis_name=axis_name)
   
    def send_left(self, x, axis_name):
        """
        This function sends the data to the left neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
            The data after being sent to the left neighboring process.
        """
        return lax.ppermute(x, perm=self.leftPerm, axis_name=axis_name)
    
    def streaming_m(self, f):
        """
        This function performs the streaming step in the Lattice Boltzmann Method, which is 
        the propagation of the distribution functions in the lattice.

        To enable multi-GPU/TPU functionality, it extracts the left and right boundary slices of the
        distribution functions that need to be communicated to the neighboring processes.

        The function then sends the left boundary slice to the right neighboring process and the right 
        boundary slice to the left neighboring process. The received data is then set to the 
        corresponding indices in the receiving domain.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The array holding the distribution functions for the simulation.

        Returns
        -------
        jax.numpy.ndarray
            The distribution functions after the streaming operation.
        """
        f = self.streaming_p(f)
        left_comm, right_comm = f[:1, ..., self.lattice.right_indices], f[-1:, ..., self.lattice.left_indices]

        left_comm, right_comm = self.send_right(left_comm, 'x'), self.send_left(right_comm, 'x')
        f = f.at[:1, ..., self.lattice.right_indices].set(left_comm)
        f = f.at[-1:, ..., self.lattice.left_indices].set(right_comm)
        return f

    @partial(jit, static_argnums=(0,))
    def streaming_p(self, f):
        """
        Perform streaming operation on a partitioned (in the x-direction) distribution function.
        
        The function uses the vmap operation provided by the JAX library to vectorize the computation 
        over all lattice directions.

        Parameters
        ----------
            f: The distribution function.

        Returns
        -------
            The updated distribution function after streaming.
        """
        def streaming_i(f, c):
            """
            Perform individual streaming operation in a direction.

            Parameters
            ----------
                f: The distribution function.
                c: The streaming direction vector.

            Returns
            -------
                jax.numpy.ndarray
                The updated distribution function after streaming.
            """
            if self.dim == 2:
                return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
            elif self.dim == 3:
                return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))

        return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(f, self.c.T)

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho, u, cast_output=True):
        """
        This function computes the equilibrium distribution function in the Lattice Boltzmann Method.
        The equilibrium distribution function is a function of the macroscopic density and velocity.

        The function first casts the density and velocity to the compute precision if the cast_output flag is True.
        The function finally casts the equilibrium distribution function to the output precision if the cast_output 
        flag is True.

        Parameters
        ----------
        rho: jax.numpy.ndarray
            The macroscopic density.
        u: jax.numpy.ndarray
            The macroscopic velocity.
        cast_output: bool, optional
            A flag indicating whether to cast the density, velocity, and equilibrium distribution function to the 
            compute and output precisions. Default is True.

        Returns
        -------
        feq: ja.numpy.ndarray
            The equilibrium distribution function.
        """
        # Cast the density and velocity to the compute precision if the cast_output flag is True
        if cast_output:
            rho, u = self.precisionPolicy.cast_to_compute((rho, u))

        # Cast c to compute precision so that XLA call FXX matmul, 
        # which is faster (it is faster in some older versions of JAX, newer versions are smart enough to do this automatically)
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype)
        cu = self.lattice.inv_cs2 * jnp.dot(u, c)
        usqr = 0.5*self.lattice.inv_cs2 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        feq = rho * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

        # cu = 3.0 * jnp.dot(u, c)
        # usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        # feq = rho * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

        if cast_output:
            return self.precisionPolicy.cast_to_output(feq)
        else:
            return feq

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium 
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann 
        Method (LBM).

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The non-equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed momentum flux.
        """
        return jnp.dot(fneq, self.lattice.cc)

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f):
        """
        This function computes the macroscopic variables (density and velocity) based on the 
        distribution functions (f).

        The density is computed as the sum of the distribution functions over all lattice directions. 
        The velocity is computed as the dot product of the distribution functions and the lattice 
        velocities, divided by the density.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution functions.

        Returns
        -------
        rho: jax.numpy.ndarray
            Computed density.
        u: jax.numpy.ndarray
            Computed velocity.
        """
        rho =jnp.sum(f, axis=-1, keepdims=True)
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
        u = jnp.dot(f, c) / rho

        return rho, u
    
    @partial(jit, static_argnums=(0, 4), inline=True)
    def apply_bc(self, fout, fin, timestep, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the 
        boundary condition matches the provided implementation step. If it does, it applies the 
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        fout: jax.numpy.ndarray
            The post-collision distribution functions.
        fin: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        ja.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))
                    
        return fout

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming, timestep, return_fpost=False):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions 
        towards their equilibrium values. It then applies the respective boundary conditions to the 
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution 
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming 
        distribution functions.

        Parameters
        ----------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions.
        timestep: int
            The current timestep of the simulation.
        return_fpost: bool, optional
            If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if 
            return_fpost is False.
        """
        f_postcollision = self.collision(f_poststreaming)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    def run(self, t_max):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the 
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and 
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max: int
            The total number of time steps to run the simulation.
        Returns
        -------
        f: jax.numpy.ndarray
            The distribution functions after the simulation.
        """
        f = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                state = {'f': f}
                shardings = jax.tree_map(lambda x: x.sharding, state)
                restore_args = orb.checkpoint_utils.construct_restore_args(state, shardings)
                try:
                    f = self.mngr.restore(latest_step, restore_kwargs={'restore_args': restore_args})['f']
                    print(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")
                
                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")
        if self.computeMLUPS:
            start = time.time()
        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate> 0 and timestep % self.printInfoRate== 0
            checkpoint_flag = self.checkpointRate > 0 and timestep % self.checkpointRate == 0

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev, u_prev = self.update_macroscopic(f)
                rho_prev = downsample_field(rho_prev, self.downsamplingFactor)
                u_prev = downsample_field(u_prev, self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)


            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.step(f, timestep, return_fpost=self.returnFpost)
            # Print the progress of the simulation
            if print_iter_flag:
                print(colored("Timestep ", 'blue') + colored(f"{timestep}", 'green') + colored(" of ", 'blue') + colored(f"{t_max}", 'green') + colored(" completed", 'blue'))

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                rho, u = self.update_macroscopic(f)
                rho = downsample_field(rho, self.downsamplingFactor)
                u = downsample_field(u, self.downsamplingFactor)
                
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, rho, u, rho_prev, u_prev)
            
            if checkpoint_flag:
                # Save the checkpoint
                print(f"Saving checkpoint at timestep {timestep}/{t_max}")
                state = {'f': f}
                self.mngr.save(timestep, state)
            
            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f)
            end = time.time()
            if self.dim == 2:
                print(colored("Domain: ", 'blue') + colored(f"{self.nx} x {self.ny}", 'green') if self.dim == 2 else colored(f"{self.nx} x {self.ny} x {self.nz}", 'green'))
                print(colored("Number of voxels: ", 'blue') + colored(f"{self.nx * self.ny}", 'green') if self.dim == 2 else colored(f"{self.nx * self.ny * self.nz}", 'green'))
                print(colored("MLUPS: ", 'blue') + colored(f"{self.nx * self.ny * t_max / (end - start) / 1e6}", 'red'))

            elif self.dim == 3:
                print(colored("Domain: ", 'blue') + colored(f"{self.nx} x {self.ny} x {self.nz}", 'green'))
                print(colored("Number of voxels: ", 'blue') + colored(f"{self.nx * self.ny * self.nz}", 'green'))
                print(colored("MLUPS: ", 'blue') + colored(f"{self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}", 'red'))

        return f

    def handle_io_timestep(self, timestep, f, fstar, rho, u, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten 
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f: jax.numpy.ndarray
            The post-streaming distribution functions at the current time step.
        fstar: jax.numpy.ndarray
            The post-collision distribution functions at the current time step.
        rho: jax.numpy.ndarray
            The density field at the current time step.
        u: jax.numpy.ndarray
            The velocity field at the current time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar
        }
        self.output_data(**kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize the input/output (I/O) 
        operations of the simulation.

        By default, it does nothing. When overwritten, it could save the simulation data to files, 
        display the simulation results in real time, send the data to another process for analysis, etc.

        Parameters
        ----------
        **kwargs: dict
            A dictionary containing the simulation data to be outputted. The keys are the names of the 
            data fields, and the values are the data fields themselves.
        """
        pass

    def set_boundary_conditions(self):
        """
        This function sets the boundary conditions for the simulation.

        It is intended to be overwritten by the user to specify the boundary conditions according to 
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip 
        boundaries, inflow/outflow boundaries, etc.
        """
        pass

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin):
        """
        This function performs the collision step in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the collision operator according to 
        the specific LBM model being used.

        By default, it does nothing. When overwritten, it could implement the BGK collision operator,
        the MRT collision operator, etc.

        Parameters
        ----------
        fin: jax.numpy.ndarray
            The pre-collision distribution functions.

        Returns
        -------
        fin: jax.numpy.ndarray
            The post-collision distribution functions.
        """
        pass

    def get_force(self):
        """
        This function computes the force to be applied to the fluid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific 
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a constant 
        force term.

        Returns
        -------
        force: jax.numpy.ndarray
            The force to be applied to the fluid.
        """
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision, feq, rho, u):
        """
        add force based on exact-difference method due to Kupershtokh

        Parameters
        ----------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions.
        feq: jax.numpy.ndarray
            The equilibrium distribution functions.
        rho: jax.numpy.ndarray
            The density field.

        u: jax.numpy.ndarray
            The velocity field.
        
        Returns
        -------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions with the force applied.
        
        References
        ----------
        Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann equation. In
        Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers, Poitiers, France.
        Chikatamarla, S. S., & Karlin, I. V. (2013). Entropic lattice Boltzmann method for turbulent flow simulations:
        Boundary conditions. Physica A, 392, 1925-1930.
        Krüger, T., et al. (2017). The lattice Boltzmann method. Springer International Publishing, 10.978-3, 4-15.
        """
        delta_u = self.get_force()
        feq_force = self.equilibrium(rho, u + delta_u, cast_output=False)
        f_postcollision = f_postcollision + feq_force - feq
        return f_postcollision
    
    @partial(jit, static_argnums=(0, 2), inline=True)
    def diffusion_equilibrium(self, phi, cast_output=True):
        """
        Compute the equilibrium distribution function for the diffusion equation.
        
        Parameters
        ----------
        phi: jax.numpy.ndarray
            The scalar field (analogous to density rho in NS equations).
        cast_output: bool, optional
            A flag indicating whether to cast the equilibrium distribution function to the output precision. Default is True.

        Returns
        -------
        feq: jax.numpy.ndarray
            The equilibrium distribution function for diffusion.
        """
        # Cast the scalar field to the compute precision if the cast_output flag is True
        if cast_output:
            phi = self.precisionPolicy.cast_to_compute(phi)

        feq = phi * self.w

        if cast_output:
            return self.precisionPolicy.cast_to_output(feq)
        else:
            return feq
        
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def diffusion_collision(self, f):
        """
        BGK collision step for the diffusion equation.
        
        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution function.

        Returns
        -------
        fout: jax.numpy.ndarray
            The post-collision distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        phi = jnp.sum(f, axis=-1, keepdims=True)  # Compute scalar field analogous to rho
        feq = self.diffusion_equilibrium(phi, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precisionPolicy.cast_to_output(fout)
    

    

```
## src/boundary_conditions.py 
 ```python
import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np
class BoundaryCondition(object):
    """
    Base class for boundary conditions in a LBM simulation.

    This class provides a general structure for implementing boundary conditions. It includes methods for preparing the
    boundary attributes and for applying the boundary condition. Specific boundary conditions should be implemented as
    subclasses of this class, with the `apply` method overridden as necessary.

    Attributes
    ----------
    lattice : Lattice
        The lattice used in the simulation.
    nx:
        The number of nodes in the x direction.
    ny:
        The number of nodes in the y direction.
    nz:
        The number of nodes in the z direction.
    dim : int
        The number of dimensions in the simulation (2 or 3).
    precision_policy : PrecisionPolicy
        The precision policy used in the simulation.
    indices : array-like
        The indices of the boundary nodes.
    name : str or None
        The name of the boundary condition. This should be set in subclasses.
    isSolid : bool
        Whether the boundary condition is for a solid boundary. This should be set in subclasses.
    isDynamic : bool
        Whether the boundary condition is dynamic (changes over time). This should be set in subclasses.
    needsExtraConfiguration : bool
        Whether the boundary condition requires extra configuration. This should be set in subclasses.
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. This should be set in subclasses.
    """

    def __init__(self, indices, gridInfo, precision_policy):
        self.lattice = gridInfo["lattice"]
        self.nx = gridInfo["nx"]
        self.ny = gridInfo["ny"]
        self.nz = gridInfo["nz"]
        self.dim = gridInfo["dim"]
        self.precisionPolicy = precision_policy
        self.indices = indices
        self.name = None
        self.isSolid = False
        self.isDynamic = False
        self.needsExtraConfiguration = False
        self.implementationStep = "PostStreaming"

    def create_local_mask_and_normal_arrays(self, grid_mask):

        """
        Creates local mask and normal arrays for the boundary condition.

        Parameters
        ----------
        grid_mask : array-like
            The grid mask for the lattice.

        Returns
        -------
        None

        Notes
        -----
        This method creates local mask and normal arrays for the boundary condition based on the grid mask.
        If the boundary condition requires extra configuration, the `configure` method is called.
        """

        if self.needsExtraConfiguration:
            boundaryMask = self.get_boundary_mask(grid_mask)
            self.configure(boundaryMask)
            self.needsExtraConfiguration = False

        boundaryMask = self.get_boundary_mask(grid_mask)
        self.normals = self.get_normals(boundaryMask)
        self.imissing, self.iknown = self.get_missing_indices(boundaryMask)
        self.imissingMask, self.iknownMask, self.imiddleMask = self.get_missing_mask(boundaryMask)

        return

    def get_boundary_mask(self, grid_mask):  
        """
        Add jax.device_count() to the self.indices in x-direction, and 1 to the self.indices other directions
        This is to make sure the boundary condition is applied to the correct nodes as grid_mask is
        expanded by (jax.device_count(), 1, 1)

        Parameters
        ----------
        grid_mask : array-like
            The grid mask for the lattice.
        
        Returns
        -------
        boundaryMask : array-like
        """   
        shifted_indices = np.array(self.indices)
        shifted_indices[0] += device_count()
        shifted_indices[1:] += 1
        # Convert back to tuple
        shifted_indices = tuple(shifted_indices)
        boundaryMask = np.array(grid_mask[shifted_indices])

        return boundaryMask

    def configure(self, boundaryMask):
        """
        Configures the boundary condition.

        Parameters
        ----------
        boundaryMask : array-like
            The grid mask for the boundary voxels.

        Returns
        -------
        None

        Notes
        -----
        This method should be overridden in subclasses if the boundary condition requires extra configuration.
        """
        return

    @partial(jit, static_argnums=(0, 3), inline=True)
    def prepare_populations(self, fout, fin, implementation_step):
        """
        Prepares the distribution functions for the boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The incoming distribution functions.
        fin : jax.numpy.ndarray
            The outgoing distribution functions.
        implementation_step : str
            The step in the lattice Boltzmann method algorithm at which the preparation is applied.

        Returns
        -------
        jax.numpy.ndarray
            The prepared distribution functions.

        Notes
        -----
        This method should be overridden in subclasses if the boundary condition requires preparation of the distribution functions during post-collision or post-streaming. See ExtrapolationBoundaryCondition for an example.
        """   
        return fout

    def get_normals(self, boundaryMask):
        """
        Calculates the normal vectors at the boundary nodes.

        Parameters
        ----------
        boundaryMask : array-like
            The boundary mask for the lattice.

        Returns
        -------
        array-like
            The normal vectors at the boundary nodes.

        Notes
        -----
        This method calculates the normal vectors by dotting the boundary mask with the main lattice directions.
        """
        main_c = self.lattice.c.T[self.lattice.main_indices]
        m = boundaryMask[..., self.lattice.main_indices]
        normals = -np.dot(m, main_c)
        return normals

    def get_missing_indices(self, boundaryMask):
        """
        Returns two int8 arrays the same shape as boundaryMask. The non-zero entries of these arrays indicate missing
        directions that require BCs (imissing) as well as their corresponding opposite directions (iknown).

        Parameters
        ----------
        boundaryMask : array-like
            The boundary mask for the lattice.

        Returns
        -------
        tuple of array-like
            The missing and known indices for the boundary condition.

        Notes
        -----
        This method calculates the missing and known indices based on the boundary mask. The missing indices are the
        non-zero entries of the boundary mask, and the known indices are their corresponding opposite directions.
        """

        # Find imissing, iknown 1-to-1 corresponding indices
        # Note: the "zero" index is used as default value here and won't affect BC computations
        nbd = len(self.indices[0])
        imissing = np.vstack([np.arange(self.lattice.q, dtype='uint8')] * nbd)
        iknown = np.vstack([self.lattice.opp_indices] * nbd)
        imissing[~boundaryMask] = 0
        iknown[~boundaryMask] = 0
        return imissing, iknown

    def get_missing_mask(self, boundaryMask):
        """
        Returns three boolean arrays the same shape as boundaryMask.
        Note: these boundary masks are useful for reduction (eg. summation) operators of selected q-directions.

        Parameters
        ----------
        boundaryMask : array-like
            The boundary mask for the lattice.

        Returns
        -------
        tuple of array-like
            The missing, known, and middle masks for the boundary condition.

        Notes
        -----
        This method calculates the missing, known, and middle masks based on the boundary mask. The missing mask
        is the boundary mask, the known mask is the opposite directions of the missing mask, and the middle mask
        is the directions that are neither missing nor known.
        """
        # Find masks for imissing, iknown and imiddle
        imissingMask = boundaryMask
        iknownMask = imissingMask[:, self.lattice.opp_indices]
        imiddleMask = ~(imissingMask | iknownMask)
        return imissingMask, iknownMask, imiddleMask

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        None

        Notes
        -----
        This method should be overridden in subclasses to implement the specific boundary condition. The method should
        modify the output distribution functions in place to apply the boundary condition.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def equilibrium(self, rho, u):
        """
        Compute equilibrium distribution function.

        Parameters
        ----------
        rho : jax.numpy.ndarray
            The density at each node in the lattice.
        u : jax.numpy.ndarray
            The velocity at each node in the lattice.

        Returns
        -------
        jax.numpy.ndarray
            The equilibrium distribution function at each node in the lattice.

        Notes
        -----
        This method computes the equilibrium distribution function based on the density and velocity. The computation is
        performed in the compute precision specified by the precision policy. The result is not cast to the output precision as
        this is function is used inside other functions that require the compute precision.
        """
        rho, u = self.precisionPolicy.cast_to_compute((rho, u))
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        usqr = 1.5 * jnp.sum(u**2, axis=-1, keepdims=True)
        feq = rho * self.lattice.w * (1.0 + 1.0 * cu + 0.5 * cu**2 - usqr)

        return feq

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        Compute the momentum flux.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            The non-equilibrium distribution function at each node in the lattice.

        Returns
        -------
        jax.numpy.ndarray
            The momentum flux at each node in the lattice.

        Notes
        -----
        This method computes the momentum flux by dotting the non-equilibrium distribution function with the lattice
        direction vectors.
        """
        return jnp.dot(fneq, self.lattice.cc)

    @partial(jit, static_argnums=(0,))
    def momentum_exchange_force(self, f_poststreaming, f_postcollision):
        """
        Using the momentum exchange method to compute the boundary force vector exerted on the solid geometry
        based on [1] as described in [3]. Ref [2] shows how [1] is applicable to curved geometries only by using a
        bounce-back method (e.g. Bouzidi) that accounts for curved boundaries.
        NOTE: this function should be called after BC's are imposed.
        [1] A.J.C. Ladd, Numerical simulations of particular suspensions via a discretized Boltzmann equation.
            Part 2 (numerical results), J. Fluid Mech. 271 (1994) 311-339.
        [2] R. Mei, D. Yu, W. Shyy, L.-S. Luo, Force evaluation in the lattice Boltzmann method involving
            curved geometry, Phys. Rev. E 65 (2002) 041203.
        [3] Caiazzo, A., & Junk, M. (2008). Boundary forces in lattice Boltzmann: Analysis of momentum exchange
            algorithm. Computers & Mathematics with Applications, 55(7), 1415-1423.

        Parameters
        ----------
        f_poststreaming : jax.numpy.ndarray
            The post-streaming distribution function at each node in the lattice.
        f_postcollision : jax.numpy.ndarray
            The post-collision distribution function at each node in the lattice.

        Returns
        -------
        jax.numpy.ndarray
            The force exerted on the solid geometry at each boundary node.

        Notes
        -----
        This method computes the force exerted on the solid geometry at each boundary node using the momentum exchange method. 
        The force is computed based on the post-streaming and post-collision distribution functions. This method
        should be called after the boundary conditions are imposed.
        """
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        phi = f_postcollision[self.indices][bindex, self.iknown] + \
              f_poststreaming[self.indices][bindex, self.imissing]
        force = jnp.sum(c[:, self.iknown] * phi, axis=-1).T
        return force

class BounceBack(BoundaryCondition):
    """
    Bounce-back boundary condition for a lattice Boltzmann method simulation.

    This class implements a full-way bounce-back boundary condition, where particles hitting the boundary are reflected
    back in the direction they came from. The boundary condition is applied after the collision step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "BounceBackFullway".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostCollision".
    """
    def __init__(self, indices, gridInfo, precision_policy):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "BounceBackFullway"
        self.implementationStep = "PostCollision"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the bounce-back boundary condition by reflecting the input distribution functions at the
        boundary nodes in the opposite direction.
        """
        
        return fin[self.indices][..., self.lattice.opp_indices]

class BounceBackMoving(BoundaryCondition):
    """
    Moving bounce-back boundary condition for a lattice Boltzmann method simulation.

    This class implements a moving bounce-back boundary condition, where particles hitting the boundary are reflected
    back in the direction they came from, with an additional velocity due to the movement of the boundary. The boundary
    condition is applied after the collision step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "BounceBackFullwayMoving".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostCollision".
    isDynamic : bool
        Whether the boundary condition is dynamic (changes over time). For this class, it is True.
    update_function : function
        A function that updates the boundary condition. For this class, it is a function that updates the boundary
        condition based on the current time step. The signature of the function is `update_function(time) -> (indices, vel)`,

    """
    def __init__(self, gridInfo, precision_policy, update_function=None):
        # We get the indices at time zero to pass to the parent class for initialization
        indices, _ = update_function(0)
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "BounceBackFullwayMoving"
        self.implementationStep = "PostCollision"
        self.isDynamic = True
        self.update_function = jit(update_function)

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin, time):
        """
        Applies the moving bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.
        time : int
            The current time step.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        indices, vel = self.update_function(time)
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        cu = 6.0 * self.lattice.w * jnp.dot(vel, c)
        return fout.at[indices].set(fin[indices][..., self.lattice.opp_indices] - cu)

class BounceBackHalfway(BoundaryCondition):
    """
    Halfway bounce-back boundary condition for a lattice Boltzmann method simulation.

    This class implements a halfway bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "BounceBackHalfway".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    needsExtraConfiguration : bool
        Whether the boundary condition needs extra configuration before it can be applied. For this class, it is True.
    isSolid : bool
        Whether the boundary condition represents a solid boundary. For this class, it is True.
    vel : array-like
        The prescribed value of velocity vector for the boundary condition. No-slip BC is assumed if vel=None (default).
    """
    def __init__(self, indices, gridInfo, precision_policy, vel=None):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "BounceBackHalfway"
        self.implementationStep = "PostStreaming"
        self.needsExtraConfiguration = True
        self.isSolid = True
        self.vel = vel

    def configure(self, boundaryMask):
        """
        Configures the boundary condition.

        Parameters
        ----------
        boundaryMask : array-like
            The grid mask for the boundary voxels.

        Returns
        -------
        None

        Notes
        -----
        This method performs an index shift for the halfway bounce-back boundary condition. It updates the indices of
        the boundary nodes to be the indices of fluid nodes adjacent of the solid nodes.
        """
        # Perform index shift for halfway BB.
        hasFluidNeighbour = ~boundaryMask[:, self.lattice.opp_indices]
        nbd_orig = len(self.indices[0])
        idx = np.array(self.indices).T
        idx_trg = []
        for i in range(self.lattice.q):
            idx_trg.append(idx[hasFluidNeighbour[:, i], :] + self.lattice.c[:, i])
        indices_new = np.unique(np.vstack(idx_trg), axis=0)
        self.indices = tuple(indices_new.T)
        nbd_modified = len(self.indices[0])
        if (nbd_orig != nbd_modified) and self.vel is not None:
            vel_avg = np.mean(self.vel, axis=0)
            self.vel = jnp.zeros(indices_new.shape, dtype=self.precisionPolicy.compute_dtype) + vel_avg
            print("WARNING: assuming a constant averaged velocity vector is imposed at all BC cells!")

        return

    @partial(jit, static_argnums=(0,))
    def impose_boundary_vel(self, fbd, bindex):
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        cu = 6.0 * self.lattice.w * jnp.dot(self.vel, c)
        fbd = fbd.at[bindex, self.imissing].add(-cu[bindex, self.iknown])
        return fbd

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the halfway bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fout[self.indices]

        fbd = fbd.at[bindex, self.imissing].set(fin[self.indices][bindex, self.iknown])
        if self.vel is not None:
            fbd = self.impose_boundary_vel(fbd, bindex)
        return fbd
    
class EquilibriumBC(BoundaryCondition):
    """
    Equilibrium boundary condition for a lattice Boltzmann method simulation.

    This class implements an equilibrium boundary condition, where the distribution function at the boundary nodes is
    set to the equilibrium distribution function. The boundary condition is applied after the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "EquilibriumBC".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    out : jax.numpy.ndarray
        The equilibrium distribution function at the boundary nodes.
    """

    def __init__(self, indices, gridInfo, precision_policy, rho, u):
        super().__init__(indices, gridInfo, precision_policy)
        self.out = self.precisionPolicy.cast_to_output(self.equilibrium(rho, u))
        self.name = "EquilibriumBC"
        self.implementationStep = "PostStreaming"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the equilibrium boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the equilibrium boundary condition by setting the output distribution functions at the
        boundary nodes to the equilibrium distribution function.
        """
        return self.out

class DoNothing(BoundaryCondition):
    def __init__(self, indices, gridInfo, precision_policy):
        """
        Do-nothing boundary condition for a lattice Boltzmann method simulation.

        This class implements a do-nothing boundary condition, where no action is taken at the boundary nodes. The boundary
        condition is applied after the streaming step.

        Attributes
        ----------
        name : str
            The name of the boundary condition. For this class, it is "DoNothing".
        implementationStep : str
            The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
            it is "PostStreaming".

        Notes
        -----
        This boundary condition enforces skipping of streaming altogether as it sets post-streaming equal to post-collision
        populations (so no streaming at this BC voxels). The problem with returning post-streaming values or "fout[self.indices]
        is that the information that exit the domain on the opposite side of this boundary, would "re-enter". This is because
        we roll the entire array and so the boundary condition acts like a one-way periodic BC. If EquilibriumBC is used as
        the BC for that opposite boundary, then the rolled-in values are taken from the initial condition at equilibrium.
        Otherwise if ZouHe is used for example the simulation looks like a run-down simulation at low-Re. The opposite boundary
        may be even a wall (consider pipebend example). If we correct imissing directions and assign "fin", this method becomes
        much less stable and also one needs to correctly take care of corner cases.
        """
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "DoNothing"
        self.implementationStep = "PostStreaming"


    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the do-nothing boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the do-nothing boundary condition by simply returning the input distribution functions at the
        boundary nodes.
        """
        return fin[self.indices]

class ZouHe(BoundaryCondition):
    """
    Zou-He boundary condition for a lattice Boltzmann method simulation.

    This class implements the Zou-He boundary condition, which is a non-equilibrium bounce-back boundary condition.
    It can be used to set inflow and outflow boundary conditions with prescribed pressure or velocity.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ZouHe".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    type : str
        The type of the boundary condition. It can be either 'velocity' for a prescribed velocity boundary condition,
        or 'pressure' for a prescribed pressure boundary condition.
    prescribed : float or array-like
        The prescribed values for the boundary condition. It can be either the prescribed velocities for a 'velocity'
        boundary condition, or the prescribed pressures for a 'pressure' boundary condition.

    References
    ----------
    Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model.
    Physics of Fluids, 9(6), 1591-1598. doi:10.1063/1.869307
    """
    def __init__(self, indices, gridInfo, precision_policy, type, prescribed):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ZouHe"
        self.implementationStep = "PostStreaming"
        self.type = type
        self.prescribed = prescribed
        self.needsExtraConfiguration = True

    def configure(self, boundaryMask):
        """
        Correct boundary indices to ensure that only voxelized surfaces with normal vectors along main cartesian axes
        are assigned this type of BC.
        """
        nv = np.dot(self.lattice.c, ~boundaryMask.T)
        corner_voxels = np.count_nonzero(nv, axis=0) > 1
        # removed_voxels = np.array(self.indices)[:, corner_voxels]
        self.indices = tuple(np.array(self.indices)[:, ~corner_voxels])
        self.prescribed = self.prescribed[~corner_voxels]
        return

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_vel(self, fpop, rho):
        """
        Calculate velocity based on the prescribed pressure/density (Zou/He BC)
        """
        unormal = -1. + 1. / rho * (jnp.sum(fpop[self.indices] * self.imiddleMask, axis=1, keepdims=True) +
                               2. * jnp.sum(fpop[self.indices] * self.iknownMask, axis=1, keepdims=True))

        # Return the above unormal as a normal vector which sets the tangential velocities to zero
        vel = unormal * self.normals
        return vel

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_rho(self, fpop, vel):
        """
        Calculate density based on the prescribed velocity (Zou/He BC)
        """
        unormal = np.sum(self.normals*vel, axis=1)

        rho = (1.0/(1.0 + unormal))[..., None] * (jnp.sum(fpop[self.indices] * self.imiddleMask, axis=1, keepdims=True) +
                                  2.*jnp.sum(fpop[self.indices] * self.iknownMask, axis=1, keepdims=True))
        return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_equilibrium(self, fpop):
        """
        This is the ZouHe method of calculating the missing macroscopic variables at the boundary.
        """
        if self.type == 'velocity':
            vel = self.prescribed
            rho = self.calculate_rho(fpop, vel)
        elif self.type == 'pressure':
            rho = self.prescribed
            vel = self.calculate_vel(fpop, rho)
        else:
            raise ValueError(f"type = {self.type} not supported! Use \'pressure\' or \'velocity\'.")

        # compute feq at the boundary
        feq = self.equilibrium(rho, vel)
        return feq

    @partial(jit, static_argnums=(0,), inline=True)
    def bounceback_nonequilibrium(self, fpop, feq):
        """
        Calculate unknown populations using bounce-back of non-equilibrium populations
        a la original Zou & He formulation
        """
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fpop[self.indices]
        fknown = fpop[self.indices][bindex, self.iknown] + feq[bindex, self.imissing] - feq[bindex, self.iknown]
        fbd = fbd.at[bindex, self.imissing].set(fknown)
        return fbd

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, _):
        """
        Applies the Zou-He boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        _ : jax.numpy.ndarray
            The input distribution functions. This is not used in this method.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the Zou-He boundary condition by first computing the equilibrium distribution functions based
        on the prescribed values and the type of boundary condition, and then setting the unknown distribution functions
        based on the non-equilibrium bounce-back method. 
        Tangential velocity is not ensured to be zero by adding transverse contributions based on
        Hecth & Harting (2010) (doi:10.1088/1742-5468/2010/01/P01018) as it caused numerical instabilities at higher
        Reynolds numbers. One needs to use "Regularized" BC at higher Reynolds.
        """
        # compute the equilibrium based on prescribed values and the type of BC
        feq = self.calculate_equilibrium(fout)

        # set the unknown f populations based on the non-equilibrium bounce-back method
        fbd = self.bounceback_nonequilibrium(fout, feq)


        return fbd

class Regularized(ZouHe):
    """
    Regularized boundary condition for a lattice Boltzmann method simulation.

    This class implements the regularized boundary condition, which is a non-equilibrium bounce-back boundary condition
    with additional regularization. It can be used to set inflow and outflow boundary conditions with prescribed pressure
    or velocity.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "Regularized".
    Qi : numpy.ndarray
        The Qi tensor, which is used in the regularization of the distribution functions.

    References
    ----------
    Latt, J. (2007). Hydrodynamic limit of lattice Boltzmann equations. PhD thesis, University of Geneva.
    Latt, J., Chopard, B., Malaspinas, O., Deville, M., & Michler, A. (2008). Straight velocity boundaries in the
    lattice Boltzmann method. Physical Review E, 77(5), 056703. doi:10.1103/PhysRevE.77.056703
    """

    def __init__(self, indices, gridInfo, precision_policy, type, prescribed):
        super().__init__(indices, gridInfo, precision_policy, type, prescribed)
        self.name = "Regularized"
        #TODO for Hesam: check to understand why corner cases cause instability here.
        # self.needsExtraConfiguration = False
        self.construct_symmetric_lattice_moment()

    def construct_symmetric_lattice_moment(self):
        """
        Construct the symmetric lattice moment Qi.

        The Qi tensor is used in the regularization of the distribution functions. It is defined as Qi = cc - cs^2*I,
        where cc is the tensor of lattice velocities, cs is the speed of sound, and I is the identity tensor.
        """
        Qi = self.lattice.cc
        if self.dim == 3:
            diagonal = (0, 3, 5)
            offdiagonal = (1, 2, 4)
        elif self.dim == 2:
            diagonal = (0, 2)
            offdiagonal = (1,)
        else:
            raise ValueError(f"dim = {self.dim} not supported")

        # Qi = cc - cs^2*I
        Qi = Qi.at[:, diagonal].set(self.lattice.cc[:, diagonal] - 1./3.)

        # multiply off-diagonal elements by 2 because the Q tensor is symmetric
        Qi = Qi.at[:, offdiagonal].set(self.lattice.cc[:, offdiagonal] * 2.0)

        self.Qi = Qi.T
        return

    @partial(jit, static_argnums=(0,), inline=True)
    def regularize_fpop(self, fpop, feq):
        """
        Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.

        Parameters
        ----------
        fpop : jax.numpy.ndarray
            The distribution functions.
        feq : jax.numpy.ndarray
            The equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The regularized distribution functions.
        """

        # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
        f_neq = fpop - feq
        PiNeq = self.momentum_flux(f_neq)
        # PiNeq = self.momentum_flux(fpop) - self.momentum_flux(feq)

        # Compute double dot product Qi:Pi1
        # QiPi1 = np.zeros_like(fpop)
        # Pi1 = PiNeq
        # QiPi1 = jnp.dot(Qi, Pi1)
        QiPi1 = jnp.dot(PiNeq, self.Qi)

        # assign all populations based on eq 45 of Latt et al (2008)
        # fneq ~ f^1
        fpop1 = 9. / 2. * self.lattice.w[None, :] * QiPi1
        fpop_regularized = feq + fpop1

        return fpop_regularized

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, _):
        """
        Applies the regularized boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        _ : jax.numpy.ndarray
            The input distribution functions. This is not used in this method.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the regularized boundary condition by first computing the equilibrium distribution functions based
        on the prescribed values and the type of boundary condition, then setting the unknown distribution functions
        based on the non-equilibrium bounce-back method, and finally regularizing the distribution functions.
        """

        # compute the equilibrium based on prescribed values and the type of BC
        feq = self.calculate_equilibrium(fout)

        # set the unknown f populations based on the non-equilibrium bounce-back method
        fbd = self.bounceback_nonequilibrium(fout, feq)

        # Regularize the boundary fpop
        fbd = self.regularize_fpop(fbd, feq)
        return fbd

class ExtrapolationOutflow(BoundaryCondition):
    """
    Extrapolation outflow boundary condition for a lattice Boltzmann method simulation.

    This class implements the extrapolation outflow boundary condition, which is a type of outflow boundary condition
    that uses extrapolation to avoid strong wave reflections.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ExtrapolationOutflow".
    sound_speed : float
        The speed of sound in the simulation.

    References
    ----------
    Geier, M., Schönherr, M., Pasquali, A., & Krafczyk, M. (2015). The cumulant lattice Boltzmann equation in three
    dimensions: Theory and validation. Computers & Mathematics with Applications, 70(4), 507–547.
    doi:10.1016/j.camwa.2015.05.001.
    """

    def __init__(self, indices, gridInfo, precision_policy):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ExtrapolationOutflow"
        self.needsExtraConfiguration = True
        self.sound_speed = 1./jnp.sqrt(3.)

    def configure(self, boundaryMask):
        """
        Configure the boundary condition by finding neighbouring voxel indices.

        Parameters
        ----------
        boundaryMask : np.ndarray
            The grid mask for the boundary voxels.
        """        
        hasFluidNeighbour = ~boundaryMask[:, self.lattice.opp_indices]
        idx = np.array(self.indices).T
        idx_trg = []
        for i in range(self.lattice.q):
            idx_trg.append(idx[hasFluidNeighbour[:, i], :] + self.lattice.c[:, i])
        indices_nbr = np.unique(np.vstack(idx_trg), axis=0)
        self.indices_nbr = tuple(indices_nbr.T)

        return

    @partial(jit, static_argnums=(0, 3), inline=True)
    def prepare_populations(self, fout, fin, implementation_step):
        """
        Prepares the distribution functions for the boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The incoming distribution functions.
        fin : jax.numpy.ndarray
            The outgoing distribution functions.
        implementation_step : str
            The step in the lattice Boltzmann method algorithm at which the preparation is applied.

        Returns
        -------
        jax.numpy.ndarray
            The prepared distribution functions.

        Notes
        -----
        Because this function is called "PostCollision", f_poststreaming refers to previous time step or t-1
        """
        f_postcollision = fout
        f_poststreaming = fin
        if implementation_step == 'PostStreaming':
            return f_postcollision
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fps_bdr = f_poststreaming[self.indices]
        fps_nbr = f_poststreaming[self.indices_nbr]
        fpc_bdr = f_postcollision[self.indices]
        fpop = fps_bdr[bindex, self.imissing]
        fpop_neighbour = fps_nbr[bindex, self.imissing]
        fpop_extrapolated = self.sound_speed * fpop_neighbour + (1. - self.sound_speed) * fpop

        # Use the iknown directions of f_postcollision that leave the domain during streaming to store the BC data
        fpc_bdr = fpc_bdr.at[bindex, self.iknown].set(fpop_extrapolated)
        f_postcollision = f_postcollision.at[self.indices].set(fpc_bdr)
        return f_postcollision

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the extrapolation outflow boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fout[self.indices]
        fbd = fbd.at[bindex, self.imissing].set(fin[self.indices][bindex, self.iknown])
        return fbd

class InterpolatedBounceBackBouzidi(BounceBackHalfway):
    """
    A local single-node version of the interpolated bounce-back boundary condition due to Bouzidi for a lattice
    Boltzmann method simulation.

    This class implements a interpolated bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "InterpolatedBounceBackBouzidi".
    implicit_distances : array-like
        An array of shape (nx,ny,nz) indicating the signed-distance field from the solid walls
    weights : array-like
        An array of shape (number_of_bc_cells, q) initialized as None and constructed using implicit_distances array
        during runtime. These "weights" are associated with the fractional distance of fluid cell to the boundary 
        position defined as: weights(dir_i) = |x_fluid - x_boundary(dir_i)| / |x_fluid - x_solid(dir_i)|.
    """

    def __init__(self, indices, implicit_distances, grid_info, precision_policy, vel=None):

        super().__init__(indices, grid_info, precision_policy, vel=vel)
        self.name = "InterpolatedBounceBackBouzidi"
        self.implicit_distances = implicit_distances
        self.weights = None

    def set_proximity_ratio(self):
        """
        Creates the interpolation data needed for the boundary condition.

        Returns
        -------
        None. The function updates the object's weights attribute in place.
        """
        epsilon = 1e-12
        nbd = len(self.indices[0])
        idx = np.array(self.indices).T
        bindex = np.arange(nbd)[:, None]
        weights = np.full((idx.shape[0], self.lattice.q), 0.5)
        c = np.array(self.lattice.c)
        sdf_f = self.implicit_distances[self.indices]
        for q in range(1, self.lattice.q):
            solid_indices = idx + c[:, q]
            solid_indices_tuple = tuple(map(tuple, solid_indices.T))
            sdf_s = self.implicit_distances[solid_indices_tuple]
            weights[:, q] = sdf_f / (sdf_f - sdf_s + epsilon)
        self.weights = weights[bindex, self.iknown]
        return

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the halfway bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        if self.weights is None:
            self.set_proximity_ratio()
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fout[self.indices]
        f_postcollision_iknown = fin[self.indices][bindex, self.iknown]
        f_postcollision_imissing = fin[self.indices][bindex, self.imissing]
        f_poststreaming_iknown = fout[self.indices][bindex, self.iknown]

        # if weights<0.5
        fs_near = 2. * self.weights * f_postcollision_iknown + \
                  (1.0 - 2.0 * self.weights) * f_poststreaming_iknown

        # if weights>=0.5
        fs_far = 1.0 / (2. * self.weights) * f_postcollision_iknown + \
                 (2.0 * self.weights - 1.0) / (2. * self.weights) * f_postcollision_imissing

        # combine near and far contributions
        fmissing = jnp.where(self.weights < 0.5, fs_near, fs_far)
        fbd = fbd.at[bindex, self.imissing].set(fmissing)

        if self.vel is not None:
            fbd = self.impose_boundary_vel(fbd, bindex)
        return fbd

class InterpolatedBounceBackDifferentiable(InterpolatedBounceBackBouzidi):
    """
    A differentiable variant of the "InterpolatedBounceBackBouzidi" BC scheme. This BC is now differentiable at
    self.weight = 0.5 unlike the original Bouzidi scheme which switches between 2 equations at weight=0.5. Refer to
    [1] (their Appendix E) for more information.

    References
    ----------
    [1] Geier, M., Schönherr, M., Pasquali, A., & Krafczyk, M. (2015). The cumulant lattice Boltzmann equation in three
    dimensions: Theory and validation. Computers & Mathematics with Applications, 70(4), 507–547.
    doi:10.1016/j.camwa.2015.05.001.


    This class implements a interpolated bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "InterpolatedBounceBackDifferentiable".
    """

    def __init__(self, indices, implicit_distances, grid_info, precision_policy, vel=None):

        super().__init__(indices, implicit_distances, grid_info, precision_policy, vel=vel)
        self.name = "InterpolatedBounceBackDifferentiable"


    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the halfway bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        if self.weights is None:
            self.set_proximity_ratio()
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fout[self.indices]
        f_postcollision_iknown = fin[self.indices][bindex, self.iknown]
        f_postcollision_imissing = fin[self.indices][bindex, self.imissing]
        f_poststreaming_iknown = fout[self.indices][bindex, self.iknown]
        fmissing = ((1. - self.weights) * f_poststreaming_iknown +
                    self.weights * (f_postcollision_imissing + f_postcollision_iknown)) / (1.0 + self.weights)
        fbd = fbd.at[bindex, self.imissing].set(fmissing)

        if self.vel is not None:
            fbd = self.impose_boundary_vel(fbd, bindex)
        return fbd

class HeatConst(BoundaryCondition):
    """
    Bounce-back boundary condition for a lattice Boltzmann method simulation.

    This class implements a full-way bounce-back boundary condition, where particles hitting the boundary are reflected
    back in the direction they came from. The boundary condition is applied after the collision step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "BounceBackFullway".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostCollision".
    """
    def __init__(self, indices, gridInfo, precision_policy, constT):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "HeatConst"
        self.implementationStep = "PostCollision"
        self.constT = constT
    
    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the bounce-back boundary condition by reflecting the input distribution functions at the
        boundary nodes in the opposite direction.
        """
        outputf = self.constT*(self.lattice.w[...,self.lattice.opp_indices] + self.lattice.w) - fin[self.indices][..., self.lattice.opp_indices]
        # fin[self.indices][..., self.lattice.opp_indices] self.lattice.w[...,self.lattice.opp_indices]
        return outputf
    
class ConstantHeatBoundaryCondition(BoundaryCondition):
    """
    Constant heat boundary condition for a lattice Boltzmann method simulation.

    This class implements a constant heat boundary condition, inspired by the bounce-back boundary condition.
    The boundary condition is applied after the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ConstantHeat".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied.
    C : float
        The constant scalar (temperature or others) on the boundary.
    """

    def __init__(self, indices, gridInfo, precision_policy, C):
        super().__init__(indices, gridInfo, precision_policy)
        self.C = C
        self.name = "ConstantHeat"
        self.implementationStep = "PostStreaming"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the constant heat boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        
        Notes
        -----
        This method applies the constant heat boundary condition by modifying the distribution functions
        based on the specified scalar constant and weights.
        """
        # Apply the constant heat boundary condition equations here
        # f[opp_idx] = C*(w[curr_idx] + w[opp_idx]) - f[curr_idx] for each direction

        w = self.lattice.w
        opp_indices = self.lattice.opp_indices

        # Update fout based on fin and the constant heat (scalar) boundary values
        for i in range(1, self.lattice.q):
            opp_idx = opp_indices[i]
            fout = fout.at[self.indices, i].set(self.C*(w[i] + w[opp_idx]) - fin[self.indices, opp_idx])

        return fout
```
## src/lattice.py 
 ```python
import re
import numpy as np
import jax.numpy as jnp


class Lattice(object):
    """
    This class represents a lattice in the Lattice Boltzmann Method.

    It stores the properties of the lattice, including the dimensions, the number of 
    velocities, the velocity vectors, the weights, the moments, and the indices of the 
    opposite, main, right, and left velocities.

    The class also provides methods to construct these properties based on the name of the 
    lattice.

    Parameters
    ----------
    name: str
        The name of the lattice, which specifies the dimensions and the number of velocities.
        For example, "D2Q9" represents a 2D lattice with 9 velocities.
    precision: str, optional
        The precision of the computations. It can be "f32/f32", "f32/f16", "f64/f64", 
        "f64/f32", or "f64/f16". The first part before the slash is the precision of the 
        computations, and the second part after the slash is the precision of the outputs.
    """
    def __init__(self, name, precision="f32/f32") -> None:
        print("test 3-7 here")
        self.name = name
        dq = re.findall(r"\d+", name)
        self.precision = precision
        self.d = int(dq[0])
        self.q = int(dq[1])
        if precision == "f32/f32" or precision == "f32/f16":
            self.precisionPolicy = jnp.float32
        elif precision == "f64/f64" or precision == "f64/f32" or precision == "f64/f16":
            self.precisionPolicy = jnp.float64
        elif precision == "f16/f16":
            self.precisionPolicy = jnp.float16
        else:
            raise ValueError("precision not supported")

        # Construct the properties of the lattice
        self.c = jnp.array(self.construct_lattice_velocity(), dtype=jnp.int8)
        self.w = jnp.array(self.construct_lattice_weight(), dtype=self.precisionPolicy)
        self.cc = jnp.array(self.construct_lattice_moment(), dtype=self.precisionPolicy)
        self.opp_indices = jnp.array(self.construct_opposite_indices(), dtype=jnp.int8)
        self.main_indices = jnp.array(self.construct_main_indices(), dtype=jnp.int8)
        self.right_indices = np.array(self.construct_right_indices(), dtype=jnp.int8)
        self.left_indices = np.array(self.construct_left_indices(), dtype=jnp.int8)

    def construct_opposite_indices(self):
        """
        This function constructs the indices of the opposite velocities for each velocity.

        The opposite velocity of a velocity is the velocity that has the same magnitude but the 
        opposite direction.

        Returns
        -------
        opposite: numpy.ndarray
            The indices of the opposite velocities.
        """
        c = self.c.T
        opposite = np.array([c.tolist().index((-c[i]).tolist()) for i in range(self.q)])
        return opposite
    
    def construct_right_indices(self):
        """
        This function constructs the indices of the velocities that point in the positive 
        x-direction.

        Returns
        -------
        numpy.ndarray
            The indices of the right velocities.
        """
        c = self.c.T
        return np.nonzero(c[:, 0] == 1)[0]
    
    def construct_left_indices(self):
        """
        This function constructs the indices of the velocities that point in the negative 
        x-direction.

        Returns
        -------
        numpy.ndarray
            The indices of the left velocities.
        """
        c = self.c.T
        return np.nonzero(c[:, 0] == -1)[0]
    
    def construct_main_indices(self):
        """
        This function constructs the indices of the main velocities.

        The main velocities are the velocities that have a magnitude of 1 in lattice units.

        Returns
        -------
        numpy.ndarray
            The indices of the main velocities.
        """
        c = self.c.T
        if self.d == 2:
            return np.nonzero((np.abs(c[:, 0]) + np.abs(c[:, 1]) == 1))[0]

        elif self.d == 3:
            return np.nonzero((np.abs(c[:, 0]) + np.abs(c[:, 1]) + np.abs(c[:, 2]) == 1))[0]

    def construct_lattice_velocity(self):
        """
        This function constructs the velocity vectors of the lattice.

        The velocity vectors are defined based on the name of the lattice. For example, for a D2Q9 
        lattice, there are 9 velocities: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), 
        (1,-1), and (-1,1).

        Returns
        -------
        c.T: numpy.ndarray
            The velocity vectors of the lattice.
        """
        if self.name == "D2Q9":  # D2Q9
            cx = [0, 0, 0, 1, -1, 1, -1, 1, -1]
            cy = [0, 1, -1, 0, 1, -1, 0, 1, -1]
            c = np.array(tuple(zip(cx, cy)))
        elif self.name == "D3Q19":  # D3Q19
            c = [(x, y, z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]]
            c = np.array([ci for ci in c if np.linalg.norm(ci) < 1.5])
        elif self.name == "D3Q27":  # D3Q27
            c = [(x, y, z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]]
            # c = np.array([ci for ci in c if np.linalg.norm(ci) < 1.5])
            c = np.array(c)
        else:
            raise ValueError("Supported Lattice types are D2Q9, D3Q19 and D3Q27")

        return c.T

    def construct_lattice_weight(self):
        """
        This function constructs the weights of the lattice.

        The weights are defined based on the name of the lattice. For example, for a D2Q9 lattice, 
        the weights are 4/9 for the rest velocity, 1/9 for the main velocities, and 1/36 for the 
        diagonal velocities.

        Returns
        -------
        w: numpy.ndarray
            The weights of the lattice.
        """
        # Get the transpose of the lattice vector
        c = self.c.T

        # Initialize the weights to be 1/36
        w = 1.0 / 36.0 * np.ones(self.q)

        # Update the weights for 2D and 3D lattices
        if self.name == "D2Q9":
            w[np.linalg.norm(c, axis=1) < 1.1] = 1.0 / 9.0
            w[0] = 4.0 / 9.0
        elif self.name == "D3Q19":
            w[np.linalg.norm(c, axis=1) < 1.1] = 2.0 / 36.0
            w[0] = 1.0 / 3.0
        elif self.name == "D3Q27":
            cl = np.linalg.norm(c, axis=1)
            w[np.isclose(cl, 1.0, atol=1e-8)] = 2.0 / 27.0
            w[(cl > 1) & (cl <= np.sqrt(2))] = 1.0 / 54.0
            w[(cl > np.sqrt(2)) & (cl <= np.sqrt(3))] = 1.0 / 216.0
            w[0] = 8.0 / 27.0
        else:
            raise ValueError("Supported Lattice types are D2Q9, D3Q19 and D3Q27")

        # Return the weights
        return w

    def construct_lattice_moment(self):
        """
        This function constructs the moments of the lattice.

        The moments are the products of the velocity vectors, which are used in the computation of 
        the equilibrium distribution functions and the collision operator in the Lattice Boltzmann 
        Method (LBM).

        Returns
        -------
        cc: numpy.ndarray
            The moments of the lattice.
        """
        c = self.c.T
        # Counter for the loop
        cntr = 0

        # nt: number of independent elements of a symmetric tensor
        nt = self.d * (self.d + 1) // 2

        cc = np.zeros((self.q, nt))
        for a in range(0, self.d):
            for b in range(a, self.d):
                cc[:, cntr] = c[:, a] * c[:, b]
                cntr += 1

        return cc
    
    def __str__(self):
        return self.name

class LatticeD2Q9(Lattice):
    """
    Lattice class for 2D D2Q9 lattice.

    D2Q9 stands for two-dimensional nine-velocity model. It is a common model used in the 
    Lat tice Boltzmann Method for simulating fluid flows in two dimensions.

    Parameters
    ----------
    precision: str, optional
        The precision of the lattice. The default is "f32/f32"
    """
    def __init__(self, precision="f32/f32"):
        super().__init__("D2Q9", precision)
        self._set_constants(1.0)

    # def _set_constants(self):
    #     self.cs = jnp.sqrt(3) / 3.0
    #     self.cs2 = 1.0 / 3.0
    #     self.inv_cs2 = 3.0
    #     self.i_s = jnp.asarray(list(range(9)))
    #     self.im = 3  # Number of imiddles (includes center)
    #     self.ik = 3  # Number of iknowns or iunknowns
    
    def _set_constants(self, iCs):
        self.cs = jnp.sqrt(3) / 3.0 * iCs
        self.cs2 = self.cs * self.cs
        self.inv_cs2 = 1/self.cs2
        self.i_s = jnp.asarray(list(range(9)))
        self.im = 3  # Number of imiddles (includes center)
        self.ik = 3  # Number of iknowns or iunknowns

        

class LatticeD3Q19(Lattice):
    """
    Lattice class for 3D D3Q19 lattice.

    D3Q19 stands for three-dimensional nineteen-velocity model. It is a common model used in the 
    Lattice Boltzmann Method for simulating fluid flows in three dimensions.

    Parameters
    ----------
    precision: str, optional
        The precision of the lattice. The default is "f32/f32"
    """
    def __init__(self, precision="f32/f32"):
        super().__init__("D3Q19", precision)
        self._set_constants()

    def _set_constants(self):
        self.cs = jnp.sqrt(3) / 3.0
        self.cs2 = 1.0 / 3.0
        self.inv_cs2 = 3.0
        self.i_s = jnp.asarray(list(range(19)), dtype=jnp.int8)

        self.im = 9  # Number of imiddles (includes center)
        self.ik = 5  # Number of iknowns or iunknowns

class LatticeD3Q27(Lattice):
    """
    Lattice class for 3D D3Q27 lattice.

    D3Q27 stands for three-dimensional twenty-seven-velocity model. It is a common model used in the 
    Lattice Boltzmann Method for simulating fluid flows in three dimensions.

    Parameters
    ----------
    precision: str, optional
        The precision of the lattice. The default is "f32/f32"
    """

    def __init__(self, precision="f32/f32"):
        super().__init__("D3Q27", precision)
        self._set_constants()

    def _set_constants(self):
        self.cs = jnp.sqrt(3) / 3.0
        self.cs2 = 1.0 / 3.0
        self.inv_cs2 = 3.0
        self.i_s = jnp.asarray(list(range(27)), dtype=jnp.int8)
```
## src/models.py 
 ```python
import jax.numpy as jnp
from jax import jit
from functools import partial
from src.base import LBMBase
"""
Collision operators are defined in this file for different models.
"""

class BGKSim(LBMBase):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation, 
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        print("shape of distribution function is: {}".format(f.shape))
        print("type of f is: {}".format(type(f)))
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)

class KBCSim(LBMBase):
    """
    KBC simulation class.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """
    def __init__(self, **kwargs):
        if kwargs.get('lattice').name != 'D3Q27' and kwargs.get('nz') > 0:
            raise ValueError("KBC collision operator in 3D must only be used with D3Q27 lattice.")
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        KBC collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        tiny = 1e-32
        beta = self.omega * 0.5
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        if self.dim == 2:
            deltaS = self.fdecompose_shear_d2q9(fneq) * rho / 4.0
        else:
            deltaS = self.fdecompose_shear_d3q27(fneq) * rho
        deltaH = fneq - deltaS
        invBeta = 1.0 / beta
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (tiny + self.entropic_scalar_product(deltaH, deltaH, feq))

        fout = f - beta * (2.0 * deltaS + gamma[..., None] * deltaH)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)
    
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision_modified(self, f):
        """
        Alternative KBC collision step for lattice.
        Note: 
        At low Reynolds number the orignal KBC collision above produces inaccurate results because
        it does not check for the entropy increase/decrease. The KBC stabalizations should only be 
        applied in principle to cells whose entropy decrease after a regular BGK collision. This is 
        the case in most cells at higher Reynolds numbers and hence a check may not be needed. 
        Overall the following alternative collision is more reliable and may replace the original 
        implementation. The issue at the moment is that it is about 60-80% slower than the above method.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        tiny = 1e-32
        beta = self.omega * 0.5
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, castOutput=False)

        # Alternative KBC: only stabalizes for voxels whose entropy decreases after BGK collision.
        f_bgk = f - self.omega * (f - feq)
        H_fin = jnp.sum(f * jnp.log(f / self.w), axis=-1, keepdims=True)
        H_fout = jnp.sum(f_bgk * jnp.log(f_bgk / self.w), axis=-1, keepdims=True)

        # the rest is identical to collision_deprecated
        fneq = f - feq
        if self.dim == 2:
            deltaS = self.fdecompose_shear_d2q9(fneq) * rho / 4.0
        else:
            deltaS = self.fdecompose_shear_d3q27(fneq) * rho
        deltaH = fneq - deltaS
        invBeta = 1.0 / beta
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (tiny + self.entropic_scalar_product(deltaH, deltaH, feq))

        f_kbc = f - beta * (2.0 * deltaS + gamma[..., None] * deltaH)
        fout = jnp.where(H_fout > H_fin, f_kbc, f_bgk)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)

    @partial(jit, static_argnums=(0,), inline=True)
    def entropic_scalar_product(self, x, y, feq):
        """
        Compute the entropic scalar product of x and y to approximate gamma in KBC.

        Returns
        -------
        jax.numpy.array
            Entropic scalar product of x, y, and feq.
        """
        return jnp.sum(x * y / feq, axis=-1)

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d2q9(self, fneq):
        """
        Decompose fneq into shear components for D2Q9 lattice.

        Parameters
        ----------
        fneq : jax.numpy.array
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        Pi = self.momentum_flux(fneq)
        N = Pi[..., 0] - Pi[..., 2]
        s = jnp.zeros_like(fneq)
        s = s.at[..., 6].set(N)
        s = s.at[..., 3].set(N)
        s = s.at[..., 2].set(-N)
        s = s.at[..., 1].set(-N)
        s = s.at[..., 8].set(Pi[..., 1])
        s = s.at[..., 4].set(-Pi[..., 1])
        s = s.at[..., 5].set(-Pi[..., 1])
        s = s.at[..., 7].set(Pi[..., 1])

        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d3q27(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """
        # if self.grid.dim == 3:
        #     diagonal    = (0, 3, 5)
        #     offdiagonal = (1, 2, 4)
        # elif self.grid.dim == 2:
        #     diagonal    = (0, 2)
        #     offdiagonal = (1,)

        # c=
        # array([[0, 0, 0],-----0
        #        [0, 0, -1],----1
        #        [0, 0, 1],-----2
        #        [0, -1, 0],----3
        #        [0, -1, -1],---4
        #        [0, -1, 1],----5
        #        [0, 1, 0],-----6
        #        [0, 1, -1],----7
        #        [0, 1, 1],-----8
        #        [-1, 0, 0],----9
        #        [-1, 0, -1],--10
        #        [-1, 0, 1],---11
        #        [-1, -1, 0],--12
        #        [-1, -1, -1],-13
        #        [-1, -1, 1],--14
        #        [-1, 1, 0],---15
        #        [-1, 1, -1],--16
        #        [-1, 1, 1],---17
        #        [1, 0, 0],----18
        #        [1, 0, -1],---19
        #        [1, 0, 1],----20
        #        [1, -1, 0],---21
        #        [1, -1, -1],--22
        #        [1, -1, 1],---23
        #        [1, 1, 0],----24
        #        [1, 1, -1],---25
        #        [1, 1, 1]])---26
        Pi = self.momentum_flux(fneq)
        Nxz = Pi[..., 0] - Pi[..., 5]
        Nyz = Pi[..., 3] - Pi[..., 5]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = jnp.zeros_like(fneq)
        s = s.at[..., 9].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 18].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 3].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 6].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 1].set((-Nxz - Nyz) / 6.0)
        s = s.at[..., 2].set((-Nxz - Nyz) / 6.0)

        # For c = (i, j, 0)
        s = s.at[..., 12].set(Pi[..., 1] / 4.0)
        s = s.at[..., 24].set(Pi[..., 1] / 4.0)
        s = s.at[..., 21].set(-Pi[..., 1] / 4.0)
        s = s.at[..., 15].set(-Pi[..., 1] / 4.0)

        # For c = (i, 0, k)
        s = s.at[..., 10].set(Pi[..., 2] / 4.0)
        s = s.at[..., 20].set(Pi[..., 2] / 4.0)
        s = s.at[..., 19].set(-Pi[..., 2] / 4.0)
        s = s.at[..., 11].set(-Pi[..., 2] / 4.0)

        # For c = (0, j, k)
        s = s.at[..., 8].set(Pi[..., 4] / 4.0)
        s = s.at[..., 4].set(Pi[..., 4] / 4.0)
        s = s.at[..., 7].set(-Pi[..., 4] / 4.0)
        s = s.at[..., 5].set(-Pi[..., 4] / 4.0)

        return s

class NonNewtonianBGK(LBMBase):
    def __init__(self, **kwargs):
        super().__init__
        self.NNPower = kwargs.get("NNPower",1)
        self.NNnu = kwargs.get("NNnu")

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming, timestep, return_fpost=False):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions 
        towards their equilibrium values. It then applies the respective boundary conditions to the 
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution 
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming 
        distribution functions.

        Parameters
        ----------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions.
        timestep: int
            The current timestep of the simulation.
        return_fpost: bool, optional
            If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if 
            return_fpost is False.
        """
        f_postcollision = self.collision(f_poststreaming)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

class AdvectionDiffusionBGK(LBMBase):
    """
    Advection Diffusion Model based on the BGK model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vel = kwargs.get("vel", None)
        if self.vel is None:
            raise ValueError("Velocity must be specified for AdvectionDiffusionBGK.")

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho =jnp.sum(f, axis=-1, keepdims=True)
        feq = self.equilibrium(rho, self.vel, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precisionPolicy.cast_to_output(fout)
    
class HeatBGK(LBMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @partial(jit, static_argnums=(0,))
    def collision(self, f, vel):
        """
        BGK collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        vel = self.precisionPolicy.cast_to_compute(vel)
        rho =jnp.sum(f, axis=-1, keepdims=True)
        feq = self.equilibrium(rho, vel, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precisionPolicy.cast_to_output(fout)
    
    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def step(self, f_poststreaming, timestep, vel, return_fpost=False):
        f_postcollision = self.collision(f_poststreaming, vel)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

class DiffusionBGK(LBMBase):
    """
    DiffusionBGK simulation class.

    This class implements the BGK approximation for the collision step in the Lattice Boltzmann Method 
    for simulating the diffusion equation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice using the diffusion equilibrium function.
        """
        return self.diffusion_collision(f)
```
## src/utils.py 
 ```python
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import numpy as np
from time import time
import pyvista as pv
from jax.image import resize
from jax import jit
import jax.numpy as jnp
from functools import partial
import trimesh

import os
import __main__


@partial(jit, static_argnums=(1, 2))
def downsample_field(field, factor, method='bicubic'):
    """
    Downsample a JAX array by a factor of `factor` along each axis.

    Parameters
    ----------
    field : jax.numpy.ndarray
        The input vector field to be downsampled. This should be a 3D or 4D JAX array where the last dimension is 2 or 3 (vector components).
    factor : int
        The factor by which to downsample the field. The dimensions of the field will be divided by this factor.
    method : str, optional
        The method to use for downsampling. Default is 'bicubic'.

    Returns
    -------
    jax.numpy.ndarray
        The downsampled field.
    """
    if factor == 1:
        return field
    else:
        new_shape = tuple(dim // factor for dim in field.shape[:-1])
        downsampled_components = []
        for i in range(field.shape[-1]):  # Iterate over the last dimension (vector components)
            resized = resize(field[..., i], new_shape, method=method)
            downsampled_components.append(resized)

        return jnp.stack(downsampled_components, axis=-1)

def save_image(timestep, fld, prefix=None):
    """
    Save an image of a field at a given timestep.

    Parameters
    ----------
    timestep : int
        The timestep at which the field is being saved.
    fld : jax.numpy.ndarray
        The field to be saved. This should be a 2D or 3D JAX array. If the field is 3D, the magnitude of the field will be calculated and saved.
    prefix : str, optional
        A prefix to be added to the filename. The filename will be the name of the main script file by default.

    Returns
    -------
    None

    Notes
    -----
    This function saves the field as an image in the PNG format. The filename is based on the name of the main script file, the provided prefix, and the timestep number.
    If the field is 3D, the magnitude of the field is calculated and saved. The image is saved with the 'nipy_spectral' colormap and the origin set to 'lower'.
    """
    fname = os.path.basename(__main__.__file__)
    fname = os.path.splitext(fname)[0]
    if prefix is not None:
        fname = prefix + fname
    fname = fname + "_" + str(timestep).zfill(4)

    if len(fld.shape) > 3:
        raise ValueError("The input field should be 2D!")
    elif len(fld.shape) == 3:
        fld = np.sqrt(fld[..., 0] ** 2 + fld[..., 1] ** 2)

    plt.clf()
    plt.imsave(fname + '.png', fld.T, cmap=cm.nipy_spectral, origin='lower')

def save_fields_vtk(timestep, fields, output_dir='.', prefix='fields'):
    """
    Save VTK fields to the specified directory.

    Parameters
    ----------
    timestep (int): The timestep number to be associated with the saved fields.
    fields (Dict[str, np.ndarray]): A dictionary of fields to be saved. Each field must be an array-like object 
        with dimensions (nx, ny) for 2D fields or (nx, ny, nz) for 3D fields, where:
            - nx : int, number of grid points along the x-axis
            - ny : int, number of grid points along the y-axis
            - nz : int, number of grid points along the z-axis (for 3D fields only)
        The key value for each field in the dictionary must be a string containing the name of the field.
    output_dir (str, optional, default: '.'): The directory in which to save the VTK files. Defaults to the current directory.
    prefix (str, optional, default: 'fields'): A prefix to be added to the filename. Defaults to 'fields'.

    Returns
    -------
    None

    Notes
    -----
    This function saves the VTK fields in the specified directory, with filenames based on the provided timestep number
    and the filename. For example, if the timestep number is 10 and the file name is fields, the VTK file
    will be saved as 'fields_0000010.vtk'in the specified directory.

    """
    # Assert that all fields have the same dimensions except for the last dimension assuming fields is a dictionary
    for key, value in fields.items():
        if key == list(fields.keys())[0]:
            dimensions = value.shape
        else:
            assert value.shape == dimensions, "All fields must have the same dimensions!"

    output_filename = os.path.join(output_dir, prefix +  "_" + f"{timestep:07d}.vtk")

    # Add 1 to the dimensions tuple as we store cell values
    dimensions = tuple([dim + 1 for dim in dimensions])

    # Create a uniform grid
    if value.ndim == 2:
        dimensions = dimensions + (1,)

    grid = pv.ImageData(dimensions=dimensions)

    # Add the fields to the grid
    for key, value in fields.items():
        grid[key] = value.flatten(order='F')

    # Save the grid to a VTK file
    start = time()
    grid.save(output_filename, binary=True)
    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")

def live_volume_randering(timestep, field):
    # WORK IN PROGRESS
    """
    Live rendering of a 3D volume using pyvista.

    Parameters
    ----------
    field (np.ndarray): A 3D array containing the field to be rendered.

    Returns
    -------
    None

    Notes
    -----
    This function uses pyvista to render a 3D volume. The volume is rendered with a colormap based on the field values.
    The colormap is updated every 0.1 seconds to reflect changes to the field.

    """
    # Create a uniform grid (Note that the field must be 3D) otherwise raise error
    if field.ndim != 3:
        raise ValueError("The input field must be 3D!")
    dimensions = field.shape
    grid = pv.ImageData(dimensions=dimensions)

    # Add the field to the grid
    grid['field'] = field.flatten(order='F')

    # Create the rendering scene
    if timestep == 0:
        plt.ion()
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.title("Live rendering of the field")
        pl = pv.Plotter(off_screen=True)
        pl.add_volume(grid, cmap='nipy_spectral', opacity='sigmoid_10', shade=False)
        plt.imshow(pl.screenshot())

    else:
        pl = pv.Plotter(off_screen=True)
        pl.add_volume(grid, cmap='nipy_spectral', opacity='sigmoid_10', shade=False)
        # Update the rendering scene every 0.1 seconds
        plt.imshow(pl.screenshot())
        plt.pause(0.1)

def save_BCs_vtk(timestep, BCs, gridInfo,  output_dir='.'):
    """
    Save boundary conditions as VTK format to the specified directory.

    Parameters
    ----------
    timestep (int): The timestep number to be associated with the saved fields.
    BCs (List[BC]): A list of boundary conditions to be saved. Each boundary condition must be an object of type BC.

    Returns
    -------
    None

    Notes
    -----
    This function saves the boundary conditions in the specified directory, with filenames based on the provided timestep number
    and the filename. For example, if the timestep number is 10, the VTK file
    will be saved as 'BCs_0000010.vtk'in the specified directory.
    """

    # Create a uniform grid
    if gridInfo['nz'] == 0:
        gridDimensions = (gridInfo['nx'] + 1, gridInfo['ny'] + 1, 1)
        fieldDimensions = (gridInfo['nx'], gridInfo['ny'], 1)
    else:
        gridDimensions = (gridInfo['nx'] + 1, gridInfo['ny'] + 1, gridInfo['nz'] + 1)
        fieldDimensions = (gridInfo['nx'], gridInfo['ny'], gridInfo['nz'])

    grid = pv.ImageData(dimensions=gridDimensions)

    # Dictionary to keep track of encountered BC names
    bcNamesCount = {}

    for bc in BCs:
        bcName = bc.name
        if bcName in bcNamesCount:
            bcNamesCount[bcName] += 1
        else:
            bcNamesCount[bcName] = 0
        bcName += f"_{bcNamesCount[bcName]}"

        if bc.isDynamic:
            bcIndices, _ = bc.update_function(timestep)
        else:
            bcIndices = bc.indices

        # Convert indices to 1D indices
        if gridInfo['dim'] == 2:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions[:-1], order='F')
        else:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions, order='F')

        grid[bcName] = np.zeros(fieldDimensions, dtype=bool).flatten(order='F')
        grid[bcName][bcIndices] = True

    # Save the grid to a VTK file
    output_filename = os.path.join(output_dir,  "BCs_" + f"{timestep:07d}.vtk")

    start = time()
    grid.save(output_filename, binary=True)
    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")


def rotate_geometry(indices, origin, axis, angle):
    """
    Rotates a voxelized mesh around a given axis.

    Parameters
    ----------
    indices : array-like
        The indices of the voxels in the mesh.
    origin : array-like
        The coordinates of the origin of the rotation axis.
    axis : array-like
        The direction vector of the rotation axis. This should be a 3-element sequence.
    angle : float
        The angle by which to rotate the mesh, in radians.

    Returns
    -------
    tuple
        The indices of the voxels in the rotated mesh.

    Notes
    -----
    This function rotates the mesh by applying a rotation matrix to the voxel indices. The rotation matrix is calculated
    using the axis-angle representation of rotations. The origin of the rotation axis is assumed to be at (0, 0, 0).
    """
    indices_rotated = (jnp.array(indices).T - origin) @ axangle2mat(axis, angle) + origin
    return tuple(jnp.rint(indices_rotated).astype('int32').T)

def voxelize_stl(stl_filename, length_lbm_unit=None, tranformation_matrix=None, pitch=None):
    """
    Converts an STL file to a voxelized mesh.

    Parameters
    ----------
    stl_filename : str
        The name of the STL file to be voxelized.
    length_lbm_unit : float, optional
        The unit length in LBM. Either this or 'pitch' must be provided.
    tranformation_matrix : array-like, optional
        A transformation matrix to be applied to the mesh before voxelization.
    pitch : float, optional
        The pitch of the voxel grid. Either this or 'length_lbm_unit' must be provided.

    Returns
    -------
    trimesh.VoxelGrid, float
        The voxelized mesh and the pitch of the voxel grid.

    Notes
    -----
    This function uses the trimesh library to load the STL file and voxelized the mesh. If a transformation matrix is
    provided, it is applied to the mesh before voxelization. The pitch of the voxel grid is calculated based on the
    maximum extent of the mesh and the provided lattice Boltzmann unit length, unless a pitch is provided directly.
    """
    if length_lbm_unit is None and pitch is None:
        raise ValueError("Either 'length_lbm_unit' or 'pitch' must be provided!")
    mesh = trimesh.load_mesh(stl_filename, process=False)
    length_phys_unit = mesh.extents.max()
    if tranformation_matrix is not None:
        mesh.apply_transform(tranformation_matrix)
    if pitch is None:
        pitch = length_phys_unit / length_lbm_unit
    mesh_voxelized = mesh.voxelized(pitch=pitch)
    return mesh_voxelized, pitch


def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From : https://github.com/matthew-brett/transforms3d
    Ref : http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = jnp.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return jnp.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])

@partial(jit)
def q_criterion(u):
    # Compute derivatives
    u_x = u[..., 0]
    u_y = u[..., 1]
    u_z = u[..., 2]

    # Compute derivatives
    u_x_dx = (u_x[2:, 1:-1, 1:-1] - u_x[:-2, 1:-1, 1:-1]) / 2
    u_x_dy = (u_x[1:-1, 2:, 1:-1] - u_x[1:-1, :-2, 1:-1]) / 2
    u_x_dz = (u_x[1:-1, 1:-1, 2:] - u_x[1:-1, 1:-1, :-2]) / 2
    u_y_dx = (u_y[2:, 1:-1, 1:-1] - u_y[:-2, 1:-1, 1:-1]) / 2
    u_y_dy = (u_y[1:-1, 2:, 1:-1] - u_y[1:-1, :-2, 1:-1]) / 2
    u_y_dz = (u_y[1:-1, 1:-1, 2:] - u_y[1:-1, 1:-1, :-2]) / 2
    u_z_dx = (u_z[2:, 1:-1, 1:-1] - u_z[:-2, 1:-1, 1:-1]) / 2
    u_z_dy = (u_z[1:-1, 2:, 1:-1] - u_z[1:-1, :-2, 1:-1]) / 2
    u_z_dz = (u_z[1:-1, 1:-1, 2:] - u_z[1:-1, 1:-1, :-2]) / 2

    # Compute vorticity
    mu_x = u_z_dy - u_y_dz
    mu_y = u_x_dz - u_z_dx
    mu_z = u_y_dx - u_x_dy
    norm_mu = jnp.sqrt(mu_x ** 2 + mu_y ** 2 + mu_z ** 2)

    # Compute strain rate
    s_0_0 = u_x_dx
    s_0_1 = 0.5 * (u_x_dy + u_y_dx)
    s_0_2 = 0.5 * (u_x_dz + u_z_dx)
    s_1_0 = s_0_1
    s_1_1 = u_y_dy
    s_1_2 = 0.5 * (u_y_dz + u_z_dy)
    s_2_0 = s_0_2
    s_2_1 = s_1_2
    s_2_2 = u_z_dz
    s_dot_s = (
        s_0_0 ** 2 + s_0_1 ** 2 + s_0_2 ** 2 +
        s_1_0 ** 2 + s_1_1 ** 2 + s_1_2 ** 2 +
        s_2_0 ** 2 + s_2_1 ** 2 + s_2_2 ** 2
    )

    # Compute omega
    omega_0_0 = 0.0
    omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
    omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
    omega_1_0 = -omega_0_1
    omega_1_1 = 0.0
    omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
    omega_2_0 = -omega_0_2
    omega_2_1 = -omega_1_2
    omega_2_2 = 0.0
    omega_dot_omega = (
        omega_0_0 ** 2 + omega_0_1 ** 2 + omega_0_2 ** 2 +
        omega_1_0 ** 2 + omega_1_1 ** 2 + omega_1_2 ** 2 +
        omega_2_0 ** 2 + omega_2_1 ** 2 + omega_2_2 ** 2
    )

    # Compute q-criterion
    q = 0.5 * (omega_dot_omega - s_dot_s)

    return norm_mu, q



```
