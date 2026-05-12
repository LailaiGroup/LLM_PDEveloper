### Code Guidelines
You must provide the complete codes and generate the test code.

0.  **basics**

   0.0 The version of jax you can use is 0.4.34.

   <!-- 0.1 Prefer introducing new functions rather than overriding existing ones. -->

   0.1 Do not override existing functions. Instead, you must introduce new ones.

   0.2 For instance method in Python class, you must use decorators like `@partial(jit, static_argnums=(0,1,2))`.
    
    <!-- not resolved -->
   0.3 Assign a global  to every newly created , even if it represents a constant in the equation.  

   0.4 Upon defining a tensor, provide comments indicating their shapes. Example:  
   ```python
   f = [some operation]  # f.shape = (nx, ny, 9)
   ```
   <!-- All the tensors you create should have the same dimensions! -->
   
   <!-- Must guarantee compatable dimensions of tensors when operating them, i.e., never make "dimension mismatch error". -->
   Never make "dimension mismatch error" when operating multiple tensors, i.e., guarantee dimensions of operated tensors are compatible.

1. **Distribution Function and Solver Implementation**  

   <!-- 1.1 You should always generate correct boundary conditions. -->

   <!-- 1.2 When implementing the solver class in `src/models.py`, do not assign boundary conditions or initialize initial conditions. This file should only contain general solver-related code.   -->

   1.2 You must not assign boundary conditions or initial conditions in the solver class in `src/models.py`.

   1.3 You must not use `jnp.einsum()`.

   1.4 You must define float variable `freq_val` in your test codes.

2. **Formatting and Code Readability**  

   2.1 You must generate code with no blank or empty lines between statements. 

3. **Testing and Debugging**  
   
   <!-- 3.0 When you receive the codes to be corrected, the codes between the symbols `<START> LLM added codes <\START>` and `<END> LLM added codes <\END>` would be deleted. You should regenerate the complete codes between these two symbols. -->

   3.0 When correcting code upon request, follow two steps. 1, Ignore marked sections: Disregard any content enclosed between the markers: `<BEG> LLM added <\BEG>` and `<END> LLM added <\END>`. 2, Regenerate code: Replace the ignored section entirely by generating all necessary code from scratch.

   3.1 In the file `test.py`, you must a constant float variable named `freq_val` and assign it a value.

   3.2 You can only use PyVista for visualization.
    <!-- and capture key data through plots.   -->


4. **Stability Considerations**  

   4.1 You must not use the Python method `momentum_flux` provided by the XLB library.  

   4.2 You must never use library `jmp` to convert the precision. 

   4.3 You must never import `partial` from jax. 
   
   4.4 You must never import `LatticeD2Q9` from `src.models`.

   4.5 You must never import `PeriodicBC` from `src/boundary_conditions.py`.

   4.6 In every file, your must ensure jax is imported as the first import statement.

   4.7 For visualization, you must not use Python method `downsample` from XLB.

   4.8 If any class in `test.py` has the attribute `nz`, then you must assign `nz` a value.

   4.9 You must never import `KBGKSim` in the import statement.
    

