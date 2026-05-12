# Boundary Conditions for the Advection–Diffusion Model in D2Q9 LBM

## Advection–Diffusion Equation (ADE)

We consider the passive-scalar ADE:
$$
\frac{\partial \phi}{\partial t} + \boldsymbol{u}\cdot\nabla \phi \;=\; \alpha \,\nabla^2 \phi,
$$
where $\phi$ is the scalar field, $\boldsymbol{u}$ is a prescribed velocity, and $\alpha$ is the diffusion coefficient.

## Lattice Boltzmann Method (LBM) on D2Q9

We solve the ADE using a single-relaxation-time (BGK) LBM on the D2Q9 lattice. At time $t$, the scalar is reconstructed as
$
\phi(\boldsymbol{x},t)=\sum_{i=0}^{8} f_i(\boldsymbol{x},t),
$
where $f_i$ are the distributions along discrete velocities $\{\boldsymbol{e}_i\}_{i=0}^{8}$.

### Discrete velocities and weights

Let $c_s^2=\tfrac{1}{3}$ be the lattice sound speed squared. The weights are
$
\omega_0=\frac{4}{9},\quad \omega_{1,2,3,6}=\frac{1}{9},\quad \omega_{5,4,7,8}=\frac{1}{36}.
$
The velocity set is
$$
[\boldsymbol{e}_i] \;=\;
\begin{pmatrix}
0 & 1 & 0 & -1 & 0 & 1 & -1 & -1 & 1 \\
0 & 0 & 1 & 0 & -1 & 1 & 1 & -1 & -1
\end{pmatrix}.
$$

### Collision and streaming

We employ collide–then–stream. The equilibrium for a passive scalar convected by $\boldsymbol{u}$ is
$
f_i^{eq}(\boldsymbol{x},t)=\omega_i\,\phi(\boldsymbol{x},t)\left(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c_s^2}\right).
$
The post-collision state is
$
\tilde f_i(\boldsymbol{x},t)=f_i(\boldsymbol{x},t)-\frac{1}{\tau}\bigl(f_i(\boldsymbol{x},t)-f_i^{eq}(\boldsymbol{x},t)\bigr),
$
and the streaming step is
$
f_i(\boldsymbol{x}+\boldsymbol{e}_i\,\Delta t,\,t+\Delta t)=\tilde f_i(\boldsymbol{x},t).
$
The relaxation time $\tau$ relates to $\alpha$ via
$
\tau=\frac{\alpha}{c_s^2\,\Delta t}+\frac{1}{2}.
$

## Boundary Conditions: Dirichlet vs Neumann (Implementation Recipes)

We describe two practical boundary closures that are convenient for unit testing and production usage.

### Notation

- $\Gamma_L,\Gamma_T,\Gamma_R,\Gamma_B$: left, top, right, and bottom boundaries.
- $\mathrm{opp}(i)$ is the opposite direction index in D2Q9:
  $
  \mathrm{opp}(0)=0,\ \mathrm{opp}(1)=3,\ \mathrm{opp}(2)=6,\ \mathrm{opp}(3)=1,\ \mathrm{opp}(6)=2,\ 
  \mathrm{opp}(5)=7,\ \mathrm{opp}(4)=8,\ \mathrm{opp}(7)=5,\ \mathrm{opp}(8)=4.
  $
- A push (source-to-destination) streaming is assumed.

### Dirichlet boundary ($\phi=0$)

**Implementation idea:**
1. Perform the bulk collision at boundary nodes to obtain $\tilde f_i$.
2. Apply **on-site bounce-back** on outward-going post-collision populations at the boundary node itself:
   - On $\Gamma_L$: reflect directions with $e_{ix}<0$, i.e., $i\in\{3,6,7\}\to \mathrm{opp}(i)\in\{1,8,5\}$.
   - On $\Gamma_T$: reflect directions with $e_{iy}<0$, i.e., $i\in\{4,7,8\}\to \mathrm{opp}(i)\in\{2,5,6\}$.
   This guarantees no net discrete flux leaving the domain through those links.
3. **Pin the macroscopic value** by resetting the boundary distributions to equilibrium at $\phi_b$:
   $$
   f_i(\boldsymbol{x}_b,t^+)=\omega_i\,\phi_b\left(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c_s^2}\right).
   $$
   For $\phi_b=0$, this reduces to $f_i(\boldsymbol{x}_b,t^+)=0$, which is simple and robust.

### Neumann boundary ($\partial\phi/\partial n=0$)

**Goal:** enforce a zero normal gradient on $\Gamma_R$ and $\Gamma_B$.

**Implementation idea (after streaming):** for each boundary node $\boldsymbol{x}_b\in \Gamma$, identify the adjacent interior node $\boldsymbol{x}_{in}$ one cell inward along the boundary normal, and **copy all distributions**:
$$
f_i(\boldsymbol{x}_b,t^+)=f_i(\boldsymbol{x}_{in},t^+)\quad \forall i=0,\dots,8.
$$
This enforces equal PDFs across the interface and yields a first-order zero normal gradient of $\phi$.


## Complete Time-stepping Scheme (with the above BCs)

At each time step:
1. **Reconstruct the scalar** everywhere:
   $
   \phi(\boldsymbol{x},t)=\sum_i f_i(\boldsymbol{x},t).
   $
2. **Collision** everywhere:
   $
   \tilde f_i=f_i-\tfrac{1}{\tau}(f_i-f_i^{eq}),\quad f_i^{eq}=\omega_i\,\phi\left(1+\tfrac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c_s^2}\right).
   $
3. **Dirichlet sides ($\Gamma_L,\Gamma_T$):** on-site bounce-back of outward-going $\tilde f_i$ at boundary nodes; then reset
   $
   f_i(\boldsymbol{x}_b,t^+)=\omega_i\,\phi_b\left(1+\tfrac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c_s^2}\right).
   $
4. **Streaming** (push) for all nodes to obtain $f_i(\cdot,t+\Delta t)$.
5. **Neumann sides ($\Gamma_R,\Gamma_B$):** copy rule after streaming:
   $
   f_i(\boldsymbol{x}_b,t+\Delta t)=f_i(\boldsymbol{x}_{in},t+\Delta t).
   $
6. Advance time and repeat until the final step.

## Test Case (for Unit Testing)

- **Domain:** uniform Cartesian grid with $n_x=n_y=101$.
- **Parameters:** $\alpha=0.01$, $\boldsymbol{u}=(0.1,\,0.2)$, $\Delta x=\Delta t=1$ (lattice units), hence
  $
  \tau=\frac{\alpha}{c_s^2}+ \frac{1}{2} = \frac{0.01}{1/3}+0.5 = 0.53.
  $
- **Initial condition (Gaussian pulse):**
  $$
  \phi(x,y,0)=\exp\!\left(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2}\right),\qquad
  x_c=\tfrac{n_x+1}{2},\ \ y_c=\tfrac{n_y+1}{2},\ \ \sigma=12.
  $$
- **Boundary conditions:**
  - **Left $\Gamma_L$ and Top $\Gamma_T$ (Dirichlet):** $\phi=0$, implemented via on-site bounce-back plus equilibrium reset to $\phi_b=0$.
  - **Right $\Gamma_R$ and Bottom $\Gamma_B$ (Neumann):** $\partial\phi/\partial n=0$, implemented by copying all $f_i$ from the adjacent interior layer after streaming.