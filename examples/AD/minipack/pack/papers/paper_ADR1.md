# The scheme of the advection-diffusion-reaction model in LBM

## Abstract
In some simulations of chemical solvents, we should also consider the reaction effect rather than advection-diffusion only. In this paper, we will introduce how to implement the advection-diffusion-reaction model by LBM method.

## Method
### Advection Diffusion Model (ADM)
ADM is used to simulate the variation of scalar $\phi$, like heat and solvents, influenced by the flow. Here is the equations of advection-diffusion model:
$$\frac{\partial\phi}{\partial t} + \boldsymbol{u}\cdot\nabla\phi = \alpha\nabla^{2}\phi + f + R(\phi)$$
Here, $\phi$ is a scalar, which represents heat or solute; $\alpha$ is a constant called diffusion coefficient; vector $\boldsymbol{u}$ is the velocity of flow. We usually consider the $\boldsymbol{u}$ as a constant vector field.

### LBM 
To solve the ADM, our LBM equation with single-relaxation time can be written as:
$$f_{i}(\boldsymbol{x} + \boldsymbol{e}_{i}\Delta t, t+ \Delta t) = f_{i}(\boldsymbol{x}, t) - \frac{1}{\tau}[f_{i}(\boldsymbol{x}, t) - f^{eq}_{i}(\boldsymbol{x}, t)] + \Delta tS_{i}\\

\\
f^{eq}_{i}(\boldsymbol{x}, t) = \omega_{i}\phi(\boldsymbol{x},t)(1 + \frac{\boldsymbol{e}_{i}\cdot\boldsymbol{u}}{c^2})\\

\\
S_i = \omega_{i}R(\phi(\boldsymbol{x},t))(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c^2})
\\
\phi(\boldsymbol{x},t) = \sum_{i=0}^{8}f_i(\boldsymbol{x},t)\\$$

Here, the $\phi(\boldsymbol{x},t)$ is the scalar field we are simulating; the vector field $\boldsymbol{u}$ refers to the velocity of flow. In the D2Q9 model for 2D flows, $\omega_0 = 4/9$, $\omega_{1,2,3,6} = 1/9$, $\omega_{4,5,7,8} = 1/36$. The discrete velocity vectors $\boldsymbol{e}_{\alpha}$ are given by
$$[\boldsymbol{e}_{\alpha}] = [\boldsymbol{e}_{0}, \boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \boldsymbol{e}_{3}, \boldsymbol{e}_{4}, \boldsymbol{e}_{5}, \boldsymbol{e}_{6}, \boldsymbol{e}_{7}, \boldsymbol{e}_{8}] = c\left(\begin{array}{cc} 
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1\\
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1 
\end{array}\right)$$
The physical meaning of constant scalar $\alpha$ is diffusion coefficient. The scalar lattice speed is a scalar $c = \frac{\Delta x}{\Delta t}$. The relaxation time denoted by scalar $\tau$ is determined by $\tau = \frac{\alpha}{c^2\Delta t} + \frac{1}{2}$.

### Schema
Here is the procedure to solve ADM by D2Q9 schemes:

0. Determine the constant velocity of flow $\boldsymbol{u}_0$, determine the constant relation time $\tau$ by constant $\alpha$;
1. Collision step. Compute the equilibrium of ADM with $\phi^{n}(\boldsymbol{x},t)$, $f^{eq}_{\alpha}(\boldsymbol{x}, t) = \omega_{\alpha}\phi(\boldsymbol{x},t)(1 + \frac{\boldsymbol{e}_{\alpha}\cdot\boldsymbol{u}}{c^2})$. Update the $S_i = \omega_{i}R(\phi^{n}(\boldsymbol{x},t))(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c^2})$;  
2. Apply the boundary conditions that post collision; 
3. Get the solution at next step by streaming $f_{\alpha}(\boldsymbol{x} + \boldsymbol{e}_{\alpha}\Delta t, t+ \Delta t) = f_{\alpha}(\boldsymbol{x}, t) - \frac{1}{\tau}[f_{\alpha}(\boldsymbol{x}, t) - f^{eq}_{\alpha}(\boldsymbol{x}, t)] + \Delta tS_{i}$
4. Apply the boundary conditions that post streaming; 
5. Repeat step 1 to step 4 above, until the end of the time;

### Testing
To test the implementation of boundary condition, the below one is a simple case:
0. The reaction term $R(\phi(\boldsymbol{x},t)) = k\phi^2(\boldsymbol{x},t)$, where $k=0.005$;
1. Geometry of domain: 2D square domain;
2. BCs: all are periodic boundary conditions;
3. Parameters: Diffusion coefficient $\alpha=0.01$, constant flow velocity (0.1, 0.2);
4. The initial distribution: $\phi(\boldsymbol{x},0)$ could be a normal distribution with peak value equals to 0.1;

Verification index:
1. the distribution of $\phi$ on the vertical centralline;
