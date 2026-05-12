# The scheme of the advection diffusion reaction equation in LBM

## Method
### Advection diffusion reaction (ADR) equation
ADR equation is used to simulate the variation of scalar $\phi$:
$$\frac{\partial\phi}{\partial t} + \boldsymbol{u}\cdot\nabla\phi = \alpha\nabla^{2}\phi + f + R(\phi) ,$$
$\phi$ is a scalar; $\alpha$ is a constant called diffusion coefficient; vector $\boldsymbol{u}$ is the velocity of flow; $R(\phi)$ is the called reaction term.

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
the discrete velocity set for the D2Q9 lattice is defined as follows:

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
f_i(\boldsymbol{x} + \boldsymbol{e}_i \Delta t, t+ \Delta t) = f_i(\boldsymbol{x}, t) - \frac{1}{\tau} [ f_i(\boldsymbol{x}, t) - f^{eq}_i(\boldsymbol{x}, t) ] + \Delta tS_i(\boldsymbol{x}, t).
$

The relaxation time $ \tau $ is determined by:

$
\tau = \frac{\alpha}{c^2 \Delta t} + \frac{1}{2}.
$

And the term $S_i(\boldsymbol{x}, t)$ is:

$
S_i(\boldsymbol{x}, t) = \omega_{i}R(\phi(\boldsymbol{x},t))(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c^2})
$

### Procedure
$\phi(\boldsymbol{x},t)$ represents  at time $t$. The procedure to solve ADR by D2Q9 lattice is:

1. Collision step: compute the equilibrium of ADR equation with $\phi(\boldsymbol{x},t) = \Sigma_{i=0}^{8}f_{i}(\boldsymbol{x}, t)$, $f^{eq}_{i}(\boldsymbol{x}, t) = \omega_{i}\phi(\boldsymbol{x},t)(1 + \frac{\boldsymbol{e}_{i}\cdot\boldsymbol{u}}{c^2})$. And compute $S_i(\boldsymbol{x}, t) = \omega_{i}R(\phi^{n}(\boldsymbol{x},t))(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c^2})$.
2. Post-collision boundary conditions (BCs): apply the boundary conditions that post collision.
3. Streaming:  $f_{i}(\boldsymbol{x} + \boldsymbol{e}_{i}\Delta t, t+ \Delta t) = f_{i}(\boldsymbol{x}, t) - \frac{1}{\tau}[f_{i}(\boldsymbol{x}, t) - f^{eq}_{i}(\boldsymbol{x}, t)] + \Delta tS_{i}(\boldsymbol{x}, t)$.
4. Post-streaming BCs: apply the BCs that post streaming. 
5. Repeat step 1 to step 4 above, until the end of the time.

### Testing
Here is a test case:

0. the reaction term $R(\phi(\boldsymbol{x},t)) = r\phi(\boldsymbol{x},t)(1-\phi(\boldsymbol{x},t))$, where the constant $r=1.0$.
1. Domain: 2D square domain. The number of grid points on each side is $n_x=200, n_y=200$.
2. BCs: we apply periodic BCs in both directions.
3. Parameters: diffusion coefficient $\alpha=0.1$. As there is no advection term in KPP equation, flow velocity is (0.0, 0.0).
4. Initial distribution: $\phi(\boldsymbol{x},0)$ is a normal distribution: $\phi(\boldsymbol{x}, 0) = \exp\left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2 \sigma^2}\right)$, where $x_c = \frac{n_x}{2}$, $y_c = \frac{n_y}{2}$, and $\sigma = 10$.

