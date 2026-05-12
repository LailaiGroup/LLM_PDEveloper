# The Scheme of the Advection-Diffusion Model in LBM

## Abstract
This paper presents the implementation of the advection-diffusion model (ADM) using the lattice Boltzmann method (LBM).

## Method

### Advection-Diffusion Model (ADM)
The ADM describes the evolution of a scalar field, $\phi$, such as temperature or solute concentration, under the influence of advection and diffusion. The governing equation is:

$$
\frac{\partial \phi}{\partial t} + \boldsymbol{u} \cdot \nabla \phi = \alpha \nabla^2 \phi + f
$$

where:
- $\phi$ is the scalar field (e.g., temperature or solute concentration),
- $\alpha$ is the diffusion coefficient,
- $\boldsymbol{u}$ is the velocity field, typically considered constant.

### Lattice Boltzmann Method (LBM)
To solve the ADM, the LBM formulation with a single-relaxation-time approach is used:

$$
f_i(\boldsymbol{x} + \boldsymbol{e}_i \Delta t, t+ \Delta t) = f_i(\boldsymbol{x}, t) - \frac{1}{\tau} [ f_i(\boldsymbol{x}, t) - f^{eq}_i(\boldsymbol{x}, t) ]
$$

where the equilibrium distribution function is defined as:

$$
f^{eq}_i(\boldsymbol{x}, t) = \omega_i \phi \left( 1 + \frac{\boldsymbol{e}_i \cdot \boldsymbol{u}}{c^2} \right)
$$

The scalar field is recovered as:

$$
\phi(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t)
$$

For the D2Q9 model in 2D simulations, the weight coefficients are:
- $\omega_0 = \frac{4}{9}$,
- $\omega_{1,2,3,6} = \frac{1}{9}$,
- $\omega_{4,5,7,8} = \frac{1}{36}$.

The discrete velocity vectors are:

$$
[\boldsymbol{e}_{\alpha}] = c \left(
\begin{array}{ccccccccc}
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1 \\
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1 
\end{array}
\right)
$$

where $c = \frac{\Delta x}{\Delta t}$ is the lattice speed. The relaxation time $\tau$ is given by:

$$
\tau = \frac{\alpha}{c^2 \Delta t} + \frac{1}{2}.
$$

### Solution Procedure
The ADM is solved using the D2Q9 scheme as follows:

0. Define the flow velocity $\boldsymbol{u}_0$ and determine the relaxation time $\tau$ based on the diffusion coefficient $\alpha$.
1. **Collision Step**: Compute the equilibrium function using $\phi^{n}$:
   
   $$
f^{eq}_{\alpha}(\boldsymbol{x}, t) = \omega_{\alpha} \phi \left( 1 + \frac{\boldsymbol{e}_{\alpha} \cdot \boldsymbol{u}}{c^2} \right)
   $$
   
   and update the distribution function.
2. **Apply boundary conditions** after the collision step.
3. **Streaming Step**: Update the distribution function by propagating along discrete velocity directions:
   
   $$
f_{\alpha}(\boldsymbol{x} + \boldsymbol{e}_{\alpha} \Delta t, t + \Delta t) = f_{\alpha}(\boldsymbol{x}, t) - \frac{1}{\tau} [ f_{\alpha}(\boldsymbol{x}, t) - f^{eq}_{\alpha}(\boldsymbol{x}, t) ].
   $$
   
4. **Apply boundary conditions** after the streaming step.
5. Repeat steps 1-4 until the desired simulation time is reached.

## Testing
To verify the implementation and accuracy of the boundary conditions, the following test case is considered:

1. **Domain Geometry**: A 2D square computational domain.
2. **Discretization**: A uniform grid with $200 \times 200$ points ($nx = 200$, $ny = 200$).
3. **Boundary Conditions**: Periodic boundary conditions applied in all directions.
4. **Physical Parameters**:
   - Diffusion coefficient: $\alpha = 0.1$,
   - Constant flow velocity: $\boldsymbol{u} = (0.1, 0.0)$.
5. **Initial Condition**: The scalar field is initialized with a Gaussian distribution:

   $$
   \phi(\boldsymbol{x}, 0) = \exp\left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2 \sigma^2}\right)
   $$
   
   where $x_c = \frac{nx}{2}$, $y_c = \frac{ny}{2}$, and $\sigma = 10$.

### Verification Metric
The scalar distribution $\phi$ should be evaluated at $T = 300$, and the simulation should be extended at least to $T = 1000$ to analyze long-term behavior.


