# Advection–Diffusion–Reaction (ADR) Model in LBM (D2Q9)

## 1. Governing Equation

We solve a generic ADR model for the scalar field $\phi(\boldsymbol{x}, t)$:
$$
\frac{\partial \phi}{\partial t} + \boldsymbol{u}\cdot\nabla \phi = \alpha \nabla^2 \phi + R(\phi).
$$

- $\phi$: scalar concentration  
- $\boldsymbol{u}$: velocity field  
- $\alpha$: diffusion coefficient  
- $R(\phi)$: reaction term 

---

## 2. Lattice Boltzmann Discretization (D2Q9, BGK)

We employ a single-relaxation-time (BGK) LBM on a D2Q9 lattice.

### 2.1 Reconstruction
At time $t$, recover the macroscopic scalar:
$$
\phi(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t).
$$

### 2.2 Equilibrium Distribution
For a generic ADR model, the scalar equilibrium takes the standard passive-scalar form
$$
f_i^{eq}(\boldsymbol{x}, t) = \omega_i\,\phi(\boldsymbol{x}, t)\!\left(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c^2}\right),
$$
where $c = \frac{\Delta x}{\Delta t} = 1$ is the lattice speed for D2Q9 and the weights are
$$
\omega_0=\frac{4}{9},\quad
\omega_{1,2,3,6}=\frac{1}{9},\quad
\omega_{4,5,7,8}=\frac{1}{36}.
$$

The discrete velocities are
$$[\boldsymbol{e}_{0}, \boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \boldsymbol{e}_{3}, \boldsymbol{e}_{4}, \boldsymbol{e}_{5}, \boldsymbol{e}_{6}, \boldsymbol{e}_{7}, \boldsymbol{e}_{8}] = c\left(\begin{array}{cc} 
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1\\
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1 
\end{array}\right)$$

### 2.3 Collide–Stream Update
$$
f_i(\boldsymbol{x}+\boldsymbol{e}_i\Delta t,\,t+\Delta t)
= f_i(\boldsymbol{x}, t) - \frac{1}{\tau}\Big[f_i(\boldsymbol{x}, t)-f_i^{eq}(\boldsymbol{x}, t)\Big] + S_i,
$$
where $S_i$ is an optional source projection for $R(\phi)$ if a non-splitting formulation is used.

### 2.4 Relaxation Time–Diffusivity Relation
With lattice sound speed $c_s^2 = 1/3$,
$$
\tau \;=\; \frac{\alpha}{c_s^2}\frac{\Delta t}{\Delta x^2} + \frac{1}{2}
\quad\Longrightarrow\quad
\text{in lattice units }(\Delta x=\Delta t=1):\; \tau = \frac{\alpha}{c_s^2} + \frac{1}{2}.
$$

---

## 3. Reaction Handling 
We project $R(\phi)$ into the discrete source $S_i$ (e.g., $S_i=\omega_i\,R(\phi)$, with optional first-order velocity corrections).

---

## 4. Algorithm 

0. **Initialize** $\phi(\boldsymbol{x}, 0)$ and set $f_i(\boldsymbol{x}, 0)=\omega_i\,\phi(\boldsymbol{x}, 0)\big(1+\frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c^2}\big)$.  
1. **Equilibrium**: build $f_i^{eq}(\phi,\boldsymbol{u})$.  
2. **Source projection**: compute $S_i=\omega_i\,R(\phi)$.  
3. **Collision**: $\tilde f_i \leftarrow f_i - \frac{1}{\tau}(f_i - f_i^{eq}) + S_i$.  
4. **Streaming**: $f_i(\boldsymbol{x}+\boldsymbol{e}_i\Delta t, t+\Delta t) \leftarrow \tilde f_i(\boldsymbol{x}, t)$.  
5. **Macroscopic update**: $\phi \leftarrow \sum_i f_i$.  
6. **Repeat**: Repeat step 1 to step 5 until the final time.
---

## 5. Testing

- **Domain**: $n_x\times n_y$ uniform grid (e.g., $100\times100$); periodic in both directions.  
- **Advection**: $\boldsymbol{u}=\boldsymbol{0}$.  
- **Diffusion**: $\alpha=0.01$ (lattice units unless otherwise noted).  
- **Reaction**: logistic $R(\phi)=r\,\phi(1-\phi)$ with $r=10^{-3}$.  
- **Initial condition** (2D Gaussian centered in the domain):
$$
\phi(\boldsymbol{x}, 0)
= \exp\!\left(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2}\right),
\qquad \sigma=12.5.
$$
- **Time steps**: run the test code for 4000 time steps.