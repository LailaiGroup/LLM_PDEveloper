# A new Equilibrium function of reflection equation in LBM
## Abstract
We introduce a novel equilibrium function to solve diffuse reflection problem. This new equilibrium function could achieve higher accuracy.

## Method
### LBM 
The LB equation with single-relaxation time can be written as:
$$f_{\alpha}(\boldsymbol{x} + \boldsymbol{e}_{\alpha}+\Delta t, t+ \Delta t) = f_{\alpha}(\boldsymbol{x}, t) - \frac{1}{\tau}[f_{\alpha}(\boldsymbol{x}, t) - f^{eq}_{\alpha}(\boldsymbol{x}, t)] + \frac{1}{6}\boldsymbol{u}\cdot\boldsymbol{e}\\
f^{eq}_{\alpha}(\boldsymbol{x}, t) = \omega_{\alpha}\rho\{ 1 + \boldsymbol{e}_{\alpha}\cdot\boldsymbol{u} + \frac{1}{2}[(\boldsymbol{e}\cdot\boldsymbol{u})^2 - 3\boldsymbol{u}\cdot\boldsymbol{u} + (T-1)(\boldsymbol{e}\cdot\boldsymbol{e}-D)] \}\\
\rho(\boldsymbol{x},t) = \sum_{0}^{8}f_\alpha(\boldsymbol{x},t)\\
\rho(\boldsymbol{x},t)\boldsymbol{u}(\boldsymbol{x},t) = \sum_{0}^{8}f_\alpha(\boldsymbol{x},t)e_\alpha$$

Here, the parameters $D$ and $T$ are diffusive parameter and temperature parameter respectively; $\rho(\boldsymbol{x},t)$ refers to the scalar of density, the $\boldsymbol{u}$ refers to the vector of velocity. In the D2Q9 model for 2D flows, $\omega_0 = 4/9$, $\omega_{1,2,3,6} = 1/9$, $\omega_{4,5,7,8} = 1/36$. The discrete velocity vectors $\boldsymbol{e}_{\alpha}$ are given by
$$[\boldsymbol{e}_{\alpha}] = [\boldsymbol{e}_{0}, \boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \boldsymbol{e}_{3}, \boldsymbol{e}_{4}, \boldsymbol{e}_{5}, \boldsymbol{e}_{6}, \boldsymbol{e}_{7}, \boldsymbol{e}_{8}] = c\left(\begin{array}{cc} 
0 & 0 & 0 & 1 & -1 & 1 & -1 & 1 & -1\\
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1 
\end{array}\right)$$
<!-- cx = [0, 0, 0, 1, -1, 1, -1, 1, -1]
cy = [0, 1, -1, 0, 1, -1, 0, 1, -1] -->
where the lattice speed $c = \frac{\Delta x}{\Delta t}$

