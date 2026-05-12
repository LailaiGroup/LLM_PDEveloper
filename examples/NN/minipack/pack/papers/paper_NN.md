# 2D solver for non-Newtonian power-Law fluid governed by power-law model 

## Governing equation for non-Newtonian power-Law fluid governed by power-law model 
$$\frac{\partial\boldsymbol{u}}{\partial t} + \boldsymbol{u}\nabla\cdot\boldsymbol{u} = -\nabla p+ \nabla\cdot\boldsymbol{\tau},\\
\boldsymbol{\tau} = K(\dot{\gamma})^{n},\\
$$
Here the vector $\boldsymbol{u}$ represents the velocity field; the scalar $p$ represents the pressure; $\dot{\gamma}$ represents the shear rate of fluid, which is a scalar field in 2D domain; $\tau$ is the shear stress; $K$ is a constant; $n$ is a constant related to the feature of fluid. It would be reduced to Newtonian fluid when $n=1$.  


## Lattice Boltzmann Method (LBM)

We employ the LBM to solve the ADE, utilizing the D2Q9 lattice structure and a single-relaxation-time approach. The LBM framework consists of two fundamental steps: the collision step followed by the streaming step. In our implementation, we first apply the collision step and then proceed with the streaming step.

At the time $t$, the fields $\phi(\boldsymbol{x}, t)$ and $\boldsymbol{u}(\boldsymbol{x}, t)$ are reconstructed as:

$
\rho(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t),\\
\rho(\boldsymbol{x}, t)\boldsymbol{u}(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t)\boldsymbol{e}_{i},\\
$

where in D2Q9, $ i = \{ 0,1,\dots,8 \} $.
The discrete velocity set for the D2Q9 lattice is defined as follows:

$
[\boldsymbol{e}_{i}] = c \left(
\begin{array}{ccccccccc}
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1 \\  
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1  
\end{array}
\right).
$
The equilibrium distribution function for collision step is given by:

$
f^{eq}_{i}(\boldsymbol{x}, t) = \omega_{i}\rho(\boldsymbol{x}, t)[1 + \frac{3}{c^2}(\boldsymbol{e}_{i}\cdot\boldsymbol{u}(\boldsymbol{x}, t)) + \frac{9}{c^4}(\boldsymbol{e}_{i}\cdot\boldsymbol{u}(\boldsymbol{x}, t))^2 - \frac{3}{2c^2}(\boldsymbol{u}(\boldsymbol{x}, t)\cdot\boldsymbol{u}(\boldsymbol{x}, t))]\\,
$

$\boldsymbol{e}_{i}$ denotes the discrete velocity set for the D2Q9; $ \Delta x $ and $ \Delta t $ represent the unit length and unit time in LBM respectively; $ c = 1/\sqrt{3} $. For the D2Q9 lattice in XLB, the weight coefficients $\omega_i$ are:

$ \omega_0 = \frac{4}{9} $,
$ \omega_{1,2,3,6} = \frac{1}{9} $,
$ \omega_{4,5,7,8} = \frac{1}{36} $.

The streaming step is formulated as:

$
f_i(\boldsymbol{x} + \boldsymbol{e}_i \Delta t, t+ \Delta t) = f_i(\boldsymbol{x}, t) - \frac{1}{\tau} [ f_i(\boldsymbol{x}, t) - f^{eq}_i(\boldsymbol{x}, t) ].
$

The relaxation time $ \tau $ is determined by:

$\tau(\boldsymbol{x},t) = \frac{\nu(\dot{\gamma})}{c^2\Delta t} + \frac{1}{2},$

the $\nu(\dot{\gamma})$ could be computed by the method mentioned in the next section.

### shear rate in LBM
In LBM we could compute the symmetry tensor of shear stress $S_{\alpha\beta} = -\frac{1}{2\rho c^2\tau\Delta t}\sum^{8}_{i=0}\boldsymbol{e}_{i\alpha}\boldsymbol{e}_{i\beta}[f_{i} - f^{eq}_{i}]$. We let $f^{neq}_{i} = f_{i} - f^{eq}_i$. In D2Q9 we would have: 
$$S_{\alpha\beta} = -\frac{1}{2\rho c^2\tau\Delta t}\begin{bmatrix}f_3 + f_4 + f_5 + f_6 + f_7 + f_8 & -f_4 - f_5 + f_7 + f_8 \\
-f_4 - f_5 + f_7 + f_8 & f_1 + f_2 + f_4 + f_5 + f_7 + f_8\end{bmatrix}
$$.
We have $\Pi = \sum^{2}_{\alpha,\beta = 1}S_{\alpha\beta}S_{\alpha\beta} = S_{11}^2 + S_{22} ^2 + 2S_{12}^2$ and we could obtain the shear rate $\dot{\gamma} = \sqrt{2\Pi}$. Further, we could get the $\nu(\dot{\gamma}) = \frac{1}{\rho}\nu_{0}|\dot{\gamma}|^{n-1}$. Here $\nu_0$ represents the initial viscocity. And  the local relaxation time $\tau(\boldsymbol{x},t) = \frac{\nu(\dot{\gamma})}{c^2\Delta t} + \frac{1}{2}$ to simulate power-law flow.

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
3. Repeat steps 1-4 until the specified simulation time is reached.

## Testing
To verify the solver, we conduct the following test simulation:

1. Domain: A 2D square computational domain.
2. Discretization: A uniform grid with $200 \times 200$ points ($n_x = 200$, $n_y = 200$).
3. BCs: Periodic BCs applied in both directions.
4. Physical parameters:
   - Diffusion coefficient: $\alpha = 0.1$,
   - Uniform flow velocity: $\boldsymbol{u} = (0.1, 0.0)$.
5. Initial condition: The scalar field is initialized with a Gaussian distribution:

   $$
   \phi(\boldsymbol{x}, 0) = \exp\left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2 \sigma^2}\right)
   $$
   
   where $x_c = \frac{n_x}{2}$, $y_c = \frac{n_y}{2}$, and $\sigma = 10$.

### shear rate in LBM
In LBM we could compute the symmetry tensor of shear stress at each time step: $S_{\alpha\beta} = -\frac{1}{2\rho c^2\tau\Delta t}\sum^{8}_{i=0}\boldsymbol{e}_{i\alpha}\boldsymbol{e}_{i\beta}[f_{i} - f^{eq}_{i}]$. In D2Q9 we would have: 
$$S_{\alpha\beta} = -\frac{1}{2\rho c^2\tau\Delta t}\begin{bmatrix}f_3 + f_4 + f_5 + f_6 + f_7 + f_8 & -f_4 - f_5 + f_7 + f_8 \\
-f_4 - f_5 + f_7 + f_8 & f_1 + f_2 + f_4 + f_5 + f_7 + f_8\end{bmatrix}
$$.
We have $\Pi = \sum^{2}_{\alpha,\beta = 1}S_{\alpha\beta}S_{\alpha\beta} = S_{11}^2 + S_{22} ^2 + 2S_{12}^2$ and we could get the shear rate $\dot{\gamma} = \sqrt{2\Pi}$. Then we could get the $\nu(\dot{\gamma}) = \frac{1}{\rho}\nu_{0}|\dot{\gamma}|^{n-1}$. Here $\nu_0$ represents the initial viscocity. And at each time step, we set the local relaxation time $\tau(\boldsymbol{x},t) = \frac{\nu(\dot{\gamma})}{c^2\Delta t} + \frac{1}{2}$ to simulate the power-law non-Newtonian flow.

### Procedure
0. Update the total time $t^{n} =  t^{n-1} + \Delta t$ 
1. At the time $t^{n-1}$, get the macroscopic value $\rho(\boldsymbol{x},t^{n-1}) = \sum^{8}_{i=0}f_{i}(\boldsymbol{x},t^{n-1})$ and $\boldsymbol{u}(\boldsymbol{x},t^{n-1}) = (\sum^{8}_{i=0}f_i(\boldsymbol{x},t^{n-1})\boldsymbol{e}_i)/\rho(\boldsymbol{x},t^{n-1})$. Compute the equilibrium distribution $f^{eq}_{i}(\boldsymbol{x},t^{n-1})$
2. Compute the shear rate $\dot{\gamma}(\boldsymbol{x},t^{n-1})$ and get the $\nu(\dot{\gamma})$. Obtain the local relaxation time $\tau(\boldsymbol{x},t^{n-1}) = \frac{\nu}{c^2\Delta t} + \frac{1}{2}$ at each point. 
3. Update the distribution function $f^{eq}_{i}(\boldsymbol{x} + \boldsymbol{e}_{i}\Delta t, t^{n-1}+ \Delta t) = f_{i}(\boldsymbol{x}, t^{n-1}) - \frac{1}{\tau(\boldsymbol{x}, t^{n-1})}[f_{i}(\boldsymbol{x}, t^{n-1}) - f^{eq}_{i}(\boldsymbol{x}, t^{n-1})]$ 
4. Repeat the step 0 to 3, until the end.

## Test Case

This is a simple 2D cavity case:

1. Domain Geometry: A 2D square domain discretized with $256 \times 8$ grids on each side, i.e. $n_x = 256 \times 8$ and $n_y = 256 \times 8$, with a characteristic length $l_0 = 256 \times 8$.

2. Boundary Conditions: The top wall has a prescribed velocity of 0.1 (i.e., the characteristic velocity $v_0 = 0.1$), while the remaining walls utilize bounce-back conditions.

3. Parameters: The Reynolds number is $Re = 50.0$, the power-law exponent is $n = 1.25$, and the initial viscosity $\nu_0$ is computed using the formula:
   $
   \nu_0 = \frac{v_0^{2-n} \, l_0^n}{Re}.
   $



