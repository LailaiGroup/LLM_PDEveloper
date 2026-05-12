# The constant value boundary condtion of the advection-diffusion model in LBM

## Abstract
While simulating the evolution of heat using Advection diffusion model, we may regard the temperature of some boundary conditions are known distributions (Direchlet boundary condition). In this paper, we will introduce how to apply Direchlet boundary condition while solving the advection-diffusion model by LBM method.

## Method
### Advection Diffusion Model (ADM)
ADM is used to simulate the variation of scalar $\phi$, like heat and solvents, influenced by the flow. Here is the equations of advection-diffusion model:
$$\frac{\partial\phi}{\partial t} + \boldsymbol{u}\cdot\nabla\phi = \alpha\nabla^{2}\phi + f$$
Here, $\phi$ is a scalar, which represents heat or solute; $\alpha$ is a constant called diffusion coefficient; vector $\boldsymbol{u}$ is the velocity of flow. We usually consider the $\boldsymbol{u}$ as a constant vector field.

### LBM 
To solve the ADM, our LBM equation with single-relaxation time can be written as:
$$f_{i}(\boldsymbol{x} + \boldsymbol{e}_{i}\Delta t, t+ \Delta t) = f_{i}(\boldsymbol{x}, t) - \frac{1}{\tau}[f_{i}(\boldsymbol{x}, t) - f^{eq}_{i}(\boldsymbol{x}, t)] + \\
f^{eq}_{i}(\boldsymbol{x}, t) = \omega_{i}\phi(1 + \frac{\boldsymbol{e}_{i}\cdot\boldsymbol{u}}{c^2})
\\
\phi(\boldsymbol{x},t) = \sum_{i=0}^{8}f_i(\boldsymbol{x},t)\\$$

Here, the $\phi(\boldsymbol{x},t)$ is the scalar field we are simulating; the vector field $\boldsymbol{u}$ refers to the velocity of flow. In the D2Q9 model for 2D flows, $\omega_0 = 4/9$, $\omega_{1,2,3,6} = 1/9$, $\omega_{4,5,7,8} = 1/36$. The discrete velocity vectors $\boldsymbol{e}_{\alpha}$ are given by
$$[\boldsymbol{e}_{\alpha}] = [\boldsymbol{e}_{0}, \boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \boldsymbol{e}_{3}, \boldsymbol{e}_{4}, \boldsymbol{e}_{5}, \boldsymbol{e}_{6}, \boldsymbol{e}_{7}, \boldsymbol{e}_{8}] = c\left(\begin{array}{cc} 
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1\\
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1 
\end{array}\right)$$
The physical meaning of constant scalar $\alpha$ is diffusion coefficient. The scalar lattice speed is a scalar $c = \frac{\Delta x}{\Delta t}$. The scalar $\tau$ is determined by $\tau = \frac{\alpha}{c^2\Delta t} + \frac{1}{2}$.

### Boundary Condition
For the boundary condition of ADM, we introduce constant scalar boundary condition. It describes that the value of scalar at the boundary is a given constant. We use the same symbols as before. The distribution on the boundary could be formulated as $f_i(\boldsymbol{x},t+\Delta t) = C\times[f_{i-opp}(\boldsymbol{x},t)+f_i(\boldsymbol{x},t)]-f_{i-opp}(\boldsymbol{x},t)$. The $f_{i-opp}$ denotes the distribution on the opposite direction of $\boldsymbol{e}_i$; constant $C$ denotes the constant scalar(temperature or others) on the boundary. More accurately, the formulation of this boundary is: 
$$
f_{1} = C(\omega_{2} + \omega_{1}) - f_{2}\\
f_{2} = C(\omega_{1} + \omega_{2}) - f_{1}\\
f_{3} = C(\omega_{6} + \omega_{3}) - f_{6}\\
f_{4} = C(\omega_{5} + \omega_{4}) - f_{5}\\
f_{5} = C(\omega_{4} + \omega_{5}) - f_{4}\\
f_{6} = C(\omega_{3} + \omega_{6}) - f_{3}\\
f_{7} = C(\omega_{8} + \omega_{7}) - f_{8}\\
f_{8} = C(\omega_{7} + \omega_{8}) - f_{7}\\
$$


### Schema
Here is the procedure to apply the Direchlet BC while solving ADM with D2Q9 lattice:

0. Determine the constant velocity of flow $\boldsymbol{u}_0$, determine the constant relation time $\tau$ by constant $\alpha$;
1. Collision step. Compute the equilibrium of ADM with $\phi^{n}$, $f^{eq}_{\alpha}(\boldsymbol{x}, t) = \omega_{\alpha}\phi(1 + \frac{\boldsymbol{e}_{\alpha}\cdot\boldsymbol{u}}{c^2})$. to obtain scalar in next time step $\phi^{n+1}$;  
2. Apply the Direchlet BC and zero-gradient BC that are post-collision; 
3. Get the solution at next step by streaming $f_{\alpha}(\boldsymbol{x} + \boldsymbol{e}_{\alpha}\Delta t, t+ \Delta t) = f_{\alpha}(\boldsymbol{x}, t) - \frac{1}{\tau}[f_{\alpha}(\boldsymbol{x}, t) - f^{eq}_{\alpha}(\boldsymbol{x}, t)] $
4. Apply the boundary conditions that are post-streaming; 
5. Repeat step 1 to step 4 above, until the end of the time;

### Testing
To test the implementation of boundary condition, the below one is a simple case:
1. Geometry of domain: 2D square domain;
2. BCs: Impose constant scalar boundary on left, top, bottom wall. The value on left side wall is constant 1.0; top and the bottom walls are all 0.0; the right wall is zero-gradient bc.
3. Parameters: Diffusion coefficient $\alpha=0.01$, constant flow velocity (0.1, 0.0);
4. The initial distribution: $\phi(\boldsymbol{x},0)=0.5$;

Verification index:
1. the distribution of $\phi$ on the vertical centralline;