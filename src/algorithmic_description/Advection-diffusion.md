# The Scheme of the Advection-Diffusion Model in LBM

## Advection-Diffusion Equation (ADE)
We solve the following ADE:

$$
\frac{\partial \phi}{\partial t} + \boldsymbol{u} \cdot \nabla \phi = \alpha \nabla^2 \phi + f
$$

where:
- $\phi$ is the scalar field,
- $\alpha$ is the diffusion coefficient,
- $\boldsymbol{u}$ is the velocity field,
- $f$ is the force term.

## Lattice Boltzmann Method (LBM)

We employ the LBM to solve the ADE, utilizing the D2Q9 lattice structure and a single-relaxation-time approach. The LBM framework consists of two fundamental steps: the collision step followed by the streaming step. In our implementation, we first apply the collision step and then proceed with the streaming step.

At the time $t$, the scalar field $\phi(\boldsymbol{x}, t)$ is reconstructed as:

$
\phi(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t),
$

where in D2Q9, $ i = \{ 0,1,\dots,8 \} $.

The equilibrium distribution function for collision step is given by:

$
f^{eq}_i(\boldsymbol{x}, t) = \omega_i \phi(\boldsymbol{x}, t) \left( 1 + \frac{\boldsymbol{e}_i \cdot \boldsymbol{u}}{c^2} \right),
$

$\boldsymbol{e}_{i}$ denotes the discrete velocity set for the D2Q9; $ \Delta x $ and $ \Delta t $ represent the unit length and unit time in LBM respectively; $ c = 1/\sqrt{3} $ denotes the lattice speed. For the D2Q9 lattice in XLB, the weight coefficients $\omega_i$ are:

$ \omega_0 = \frac{4}{9} $,
$ \omega_{1,2,3,6} = \frac{1}{9} $,
$ \omega_{4,5,7,8} = \frac{1}{36} $.

The discrete velocity set for the D2Q9 lattice is defined as follows:

$
[\boldsymbol{e}_{i}] = c \left(
\begin{array}{ccccccccc}
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1 \\  
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1  
\end{array}
\right).
$

The streaming step is formulated as:

$
f_i(\boldsymbol{x} + \boldsymbol{e}_i \Delta t, t+ \Delta t) = f_i(\boldsymbol{x}, t) - \frac{1}{\tau} [ f_i(\boldsymbol{x}, t) - f^{eq}_i(\boldsymbol{x}, t) ].
$

The relaxation time $ \tau $ is determined by:

$
\tau = \frac{\alpha}{c^2 \Delta t} + \frac{1}{2}.
$

## Solution Procedure
The ADE is solved as follows:

0. Define the flow velocity $\boldsymbol{u}_0$ and determine the relaxation time $\tau$ based on the diffusion coefficient $\alpha$.

1. Recover the $\phi$ at time $t$ : $
\phi(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t),
$

2. Collision step: Compute the equilibrium function using:
   
   $
f^{eq}_{i}(\boldsymbol{x}, t) = \omega_{i} \phi(\boldsymbol{x}, t) \left( 1 + \frac{\boldsymbol{e}_{i} \cdot \boldsymbol{u}}{c^2} \right).
   $

3. Streaming step: update the distribution function by  along discrete velocity directions:
   
   $
f_{i}(\boldsymbol{x} + \boldsymbol{e}_{i} \Delta t, t + \Delta t) = f_{i}(\boldsymbol{x}, t) - \frac{1}{\tau} [ f_{i}(\boldsymbol{x}, t) - f^{eq}_{i}(\boldsymbol{x}, t) ].
   $
   
4. Update the time: $t = t + \Delta t$
3. Repeat steps 1-4 until the specified simulation time 1000 is reached.

## Testing
To verify the solver, we conduct the following test simulation:

- **Domain**: $n_x\times n_y$ uniform grid (e.g., $200\times200$); periodic in both directions. 
- **Advection**: $\boldsymbol{u} = (0.1, 0.0)$
- **Diffusion**: $\alpha=0.1$ 
- **Initial condition**: The scalar field is initialized with a Gaussian distribution:

   $$
   \phi(\boldsymbol{x}, 0) = \exp\left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2 \sigma^2}\right)
   $$
   
   where $x_c = \frac{n_x}{2}$, $y_c = \frac{n_y}{2}$, and $\sigma = 10$.
- **Time steps**:Run the test code for 1000 time steps.


