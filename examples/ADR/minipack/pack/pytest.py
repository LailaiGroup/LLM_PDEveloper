import pyvista as pv
import numpy as np

# Load the two VTK files
file1 = "fields_0000300.vtk"
file2 = "benchmark/fields_0000300.vtk"

mesh1 = pv.read(file1)
mesh2 = pv.read(file2)

print(mesh1.array_names)
print(mesh2.array_names)
# Extract the flow field arrays (assuming scalar or vector data is stored as point data)
# Replace "field_name" with the actual name of the data array in your VTK files
field1 = mesh1["phi"]  # Example: velocity, pressure, etc.
field2 = mesh2["phi"]

# Ensure the fields have the same shape
if field1.shape != field2.shape:
    print("The flow fields in the two VTK files have different dimensions!")
    raise ValueError("The flow fields in the two VTK files have different dimensions!")

# Subtract the fields
difference = np.abs(field1 - field2)

# Get the maximum difference
max_diff = np.max(difference)

# Get the maximum in benchmark
max_v = np.max(np.abs(field2))
print(f"The maximum difference between the two flow fields is: {max_diff/max_v}")