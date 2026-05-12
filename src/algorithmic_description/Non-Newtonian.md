# 2D solver for non-Newtonian power-law fluid governed by the power-law model

## Governing equation for a non-Newtonian power-law fluid
$$
\frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{u}\cdot\nabla \boldsymbol{u}
= -\nabla p + \nabla \cdot \boldsymbol{\tau},\qquad
\boldsymbol{\tau} = K\,(\dot{\gamma})^{n}.
$$
Here, $\boldsymbol{u}$ is the velocity field, $p$ is the pressure, $\dot{\gamma}$ is the shear rate (a scalar field in 2D), $\boldsymbol{\tau}$ is the shear stress, $K$ and $n$ are two constant parameters, the so-called flow behavior index and flow consistency index respectively. The model reduces to the Newtonian case when $n=1$.

## Lattice Boltzmann Method (LBM)

We employ the LBM with a D2Q9 lattice and a single-relaxation-time scheme. The method consists of two steps: a collision step followed by a streaming step. In our implementation, the collision step is applied first and then the streaming step.

At time $t$, the macroscopic fields are reconstructed as
$$
\rho(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t),\qquad
\rho(\boldsymbol{x}, t)\,\boldsymbol{u}(\boldsymbol{x}, t) = \sum_{i=0}^{8} f_i(\boldsymbol{x}, t)\,\boldsymbol{e}_i,
$$
where $i=\{0,1,\dots,8\}$ for D2Q9. The discrete velocities are
$$
[\boldsymbol{e}_{i}] = c \begin{pmatrix}
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1\\
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1
\end{pmatrix}.
$$

The equilibrium distribution for the collision step is
$$
f^{eq}_{i}(\boldsymbol{x}, t) = \omega_{i}\,\rho(\boldsymbol{x}, t)\!\left[
1 + \frac{3}{c^2}\big(\boldsymbol{e}_{i}\!\cdot\!\boldsymbol{u}(\boldsymbol{x}, t)\big)
+ \frac{9}{c^4}\big(\boldsymbol{e}_{i}\!\cdot\!\boldsymbol{u}(\boldsymbol{x}, t)\big)^2
- \frac{3}{2c^2}\big(\boldsymbol{u}(\boldsymbol{x}, t)\!\cdot\!\boldsymbol{u}(\boldsymbol{x}, t)\big)
\right]\!,
$$
where $\Delta x$ and $\Delta t$ are the lattice spacing and time step, and $c = 1/\sqrt{3}$. The D2Q9 weights are
$$
\omega_0 = \tfrac{4}{9},\qquad
\omega_{1,2,3,4} = \tfrac{1}{9},\qquad
\omega_{5,6,7,8} = \tfrac{1}{36}.
$$

The streaming step is written as
$$
f_i(\boldsymbol{x} + \boldsymbol{e}_i \Delta t,\, t+ \Delta t) =
f_i(\boldsymbol{x}, t) - \frac{1}{\tau}\,\big[ f_i(\boldsymbol{x}, t) - f^{eq}_i(\boldsymbol{x}, t) \big].
$$

The relaxation time $\tau$ is determined by
$$
\tau(\boldsymbol{x},t) = \frac{\nu(\dot{\gamma})}{c^2\Delta t} + \frac{1}{2},
$$
where $\nu(\dot{\gamma})$ is computed as described in the next subsection.

### Shear rate in LBM
In LBM, the symmetric shear-rate tensor can be evaluated as
$$
S_{\alpha\beta} = -\frac{1}{2\rho\,c^2\,\tau\,\Delta t}\,
\sum_{i=0}^{8} e_{i\alpha}\,e_{i\beta}\,\big(f_{i}-f^{eq}_{i}\big),
$$
where $f^{neq}_{i} = f_{i} - f^{eq}_i$. For D2Q9,
$$
S_{\alpha\beta} =
-\frac{1}{2\rho\,c^2\,\tau\,\Delta t}
\begin{bmatrix}
f_3 + f_4 + f_5 + f_6 + f_7 + f_8 & -f_4 - f_5 + f_7 + f_8\\[2pt]
-f_4 - f_5 + f_7 + f_8 & f_1 + f_2 + f_4 + f_5 + f_7 + f_8
\end{bmatrix}.
$$
Define $\Pi = \sum_{\alpha,\beta=1}^{2} S_{\alpha\beta}S_{\alpha\beta}
= S_{11}^2 + S_{22}^2 + 2S_{12}^2$. The shear rate is
$$
\dot{\gamma} = \sqrt{2\Pi},
$$
and
$$
\nu(\dot{\gamma}) = \frac{1}{\rho}\,\nu_{0}\,|\dot{\gamma}|^{\,n-1}.
$$
Here, $\nu_0$ is the reference (initial) viscosity. The local relaxation time is then
$$
\tau(\boldsymbol{x},t) = \frac{\nu(\dot{\gamma})}{c^2\Delta t} + \frac{1}{2},
$$
which enables simulation of power-law flow.

## Test Case

A simple 2D cavity case:

1. **Domain geometry:** a square domain discretized with $256$ points on each side, i.e., $n_x = 256$, $n_y = 256$, and characteristic length $l_0 = 256$.  
2. **Boundary conditions:** the top wall has a prescribed velocity of $0.1$ (characteristic velocity $v_0=0.1$); the other walls use bounce-back boundary conditions.  
3. **Parameters:** Reynolds number $Re = 400$; power-law index $n = 1.25$ and $K=1.0$; the initial viscosity $\nu_0$ is set by
$$
\nu_0 = \frac{v_0^{\,2-n}\, l_0^{\,n}}{Re}.
$$
4. **Time steps**: run the test code for 70000 time steps
