# Introduction

## Model setup

Let $(S_t)_{t\geq 0}$ be the underlying asset price in $\mathbb{Q}$
measure. Suppose that the stochastic discount factor is $D_t$, we define
the discounted price process as $F_t \equiv D_t S_t$. Where we see that
$F_t$ is a $\mathbb{Q}-$martingale. Now consider a collection of $n$
state processes $(Y_{i,t})_{t\geq 0}$ which are assumed to be
non-negative orthogonal factors, hence lives in independent probability
spaces. We suppose that the state process follows a Feller square root
Jump-Diffusion model $$\begin{aligned}
    dY_{i,t} = \sigma_i \sqrt{Y_{i,t}} dW_{i,t} +  \int_0^\infty z \tilde{N}_i(dt, dz) 
\end{aligned}$$ where the jump term is a compound Poisson jump process
with jump rate $\lambda_i$, with positive jump intensity measure
$\nu_i(dz)$. For the rest of this paper we will assume an exponential
jump intensity with rate parameter of $\gamma_i$ due to its analytical
tractability in deriving closed form affine characteristic functions.
Hence $$\begin{aligned}
    \nu_i(dz) =\lambda_i\gamma_ie^{-\gamma_i z}dz
\end{aligned}$$ Now, let $X_t = \log(F_t)$, we model $X_t$ as a mixture
of the orthogonal factors plus a compensating drift term:
$$\begin{aligned}
    dX_{t} = \sum_{i=1}^n \alpha_{i}(dY_{i,t} + c_{i,t}dt) 
\end{aligned}$$ Where $\{\alpha\}_{1\leq i \leq n}$ are the weighting
coefficients and $\{c_t\}_{1\leq i \leq n}$ are the compensating drift
terms to ensure that $F_t$ is a $\mathbb{Q}$-martingale.

## Compensating to obtain $\mathbb{Q}$-Martingale

To ensure $F_t$ is a $\mathbb{Q}$-martingale, we first use Ito Lemma to
write the SDE for $F_t$. We see that if $F_t = e^{X_t}$ then,
$$\begin{aligned}
    dF_t = F_{t^-}\left(dX_t + \frac{1}{2}d\braket{X_t}^{cont}_t\right) + F_{t^-}\sum_{i=1}^n(e^{\Delta_i X_t}-1)
\end{aligned}$$ The derivation for the continuous portion of the Ito
Lemma is straightforward, on a side note one can see that the jump
contribution can be derived as follows:

Recall that for some function $f(X)$, the generator of the compound
Poisson process is: $$\begin{aligned}
    Qf(x)=\int_0^\infty [f(x+z) - f(x) ]\nu(dz)
\end{aligned}$$ Appropriately choosing $f = e^X$ and noting that a jump
in $Y_i$ contributes to a change of $\alpha_iz_i$ in $X$ we have that
$$\begin{aligned}
    Q_if(x)=\int_0^\infty e^X[e^{\alpha_i z}- 1]\nu_i(dz) = F_{t^-}\int_0^\infty [e^{\alpha_i z}- 1]\nu_i(dz) 
\end{aligned}$$ Hence the total SDE for $F_t$ becomes $$\begin{aligned}
    dF_t = F_{t^-}\left(\sum_{i=1}^n \alpha_i(dY_{i,t} + c_{i,t} dt+ \frac{1}{2}\alpha_i\sigma_i^2 Y_{i,t}dt)\right) + F_{t^-}\sum_{i=1}^n\int_0^\infty(e^{\alpha_i z}-1)\nu_i(dz)
\end{aligned}$$ To ensure that $F_t$ is a martingale, we must have that
$\mathbb{E}[dF_t | \mathcal{F}_t]=0$ or equivalently $$\begin{aligned}
    \sum_{i=1}^n \alpha_{i} (c_{i,t} + \frac{1}{2}\sigma_i^2 \alpha_iY_{i,t})dt + \int_0^\infty\mathbb{E}[(e^{\alpha_iz}-1)\nu_i(dz)] =0
\end{aligned}$$ we compute $$\begin{aligned}
    \int_0^\infty \mathbb{E}[(e^{\alpha_iz}-1)\nu_i(dz)] = \lambda_i\mathbb{E}[e^{\alpha_i Z_i}-1] = \frac{\lambda_i\alpha_i}{\gamma_i-\alpha_i}
\end{aligned}$$ If $c_{i,t} = \beta_i + \delta_i Y_{i,t}$ we see that
$$\begin{aligned}
    \beta_i &= -\frac{\lambda_i}{\gamma_i-\alpha_i}\\
    \delta_i &=-\frac{1}{2}\sigma_i^2\alpha_i
\end{aligned}$$ Which completely determines the dynamics of the
compensated process. Note that to ensure the well defined-ness of $F_t$
specifically $\sup_t \mathbb{E}[||F_t||] <\infty$ then
$\gamma_i > \alpha_i$.

# Exponential Affine Characteristic Function

In this section we will derive the characteristic function of the $X_t$
process. Consider the forward PDE for $\phi(u;t)$ where
$\phi(u;t, x,\mathbf{y}) = \mathbb{E}[e^{iuX_{t}}|X_0 = x, \mathbf{Y}_0 = \mathbf{y}]$.
We find that $$\begin{aligned}
    \partial_t\phi = \sum_{i=1}^n \alpha_i(\beta_i + \delta_i y_i) \phi_x + \frac{1}{2}\sigma_i^2y_i(\alpha_i^2\phi_{xx} + 2\alpha_i\phi_{xy} + \phi_{yy})  + \mathbb{E} [\phi(x+\alpha_iZ_i, y_i + Z_i) - \phi(x,y)]
\end{aligned}$$ Noticing the affine structure we produce an ansatz
$$\begin{aligned}
    \phi(u;t) = \exp(iux + A(t) + \sum_{i=1}^n B_i(t)y_i)
\end{aligned}$$ We find that this gives us $$\begin{aligned}
    (A'(t)+\sum_{i=1}^n B'_i(t)y_i) &= \sum_{i=1}^niu\alpha_i(\beta_i + \delta_iy_i) + \frac{1}{2}\sigma_i^2y_i(-u^2\alpha_i^2 +2\alpha_iiuB(t) + B(t)^2 )\\& + \mathbb{E}[e^{iu\alpha_iZ_i +B_i(t)Z_i} - 1]\nonumber
\end{aligned}$$ Matching the terms we see that $$\begin{aligned}
    A'(t) &= \sum_{i=1}^n iu\alpha_i\beta_i +  \mathbb{E}[e^{iu\alpha_iZ_i +B_i(t)Z_i} - 1];\quad A(0)= 0\\
    B'_i(t) &= iu\alpha_i\delta_i - \frac{1}{2}\sigma_i^2u^2\alpha_i^2 + i\sigma^2_i\alpha_iuB(t) + \frac{1}{2}\sigma_i^2B(t)^2;\quad B(0) =0
\end{aligned}$$ We can substitute in the expectations of the exponential
jump intensity and the compensator term for $\delta_i$ to simplify
$$\begin{aligned}
    A'(t) &= \sum_{i=1}^n \left(iu\alpha_i\beta_i +  \frac{\lambda_i\gamma_i}{\gamma_i -( B_i(t)+ iu\alpha_i)}-1\right);\quad A(0)= 0\\
    B'_i(t) &= -\frac{1}{2}iu\alpha_i^2\sigma_i^2- \frac{1}{2}\sigma_i^2u^2\alpha_i^2 + i\sigma^2_i\alpha_iuB(t) + \frac{1}{2}\sigma_i^2B(t)^2;\quad B(0) =0
\end{aligned}$$ The ODE for $B_i(t)$ can be solved with the given the
initial conditions. The solution after some algebra is $$\begin{aligned}
    B_i(t) =-(-1)^{3/4} \sqrt{u} \alpha _i \tan \left(\tan ^{-1}\left((-1)^{3/4}
   \sqrt{u}\right)-\frac{1}{2} (-1)^{3/4} t \sqrt{u} \alpha _i \sigma _i^2\right)-i u
   \alpha _i
\end{aligned}$$ Moving forward we use the express for $B_i(t)$ to find
$A'(t)$, such a solution exists. We define $$\begin{aligned}
    F_i(t) = \int \left[\frac{\gamma_i}{\gamma_i -( B_i(t)+ iu\alpha_i)}-1\right]dt
\end{aligned}$$ We find that $$\begin{aligned}
    F_i(t) &= -t-\frac{2 \gamma_i  \log \left(\gamma_i +(-1)^{3/4} \sqrt{u} \alpha _i \tan \left(\tan
   ^{-1}\left((-1)^{3/4} \sqrt{u}\right)-\frac{1}{2} (-1)^{3/4} t \sqrt{u} \alpha _i
   \sigma _i^2\right)\right)}{\sigma _i^2 \left(\gamma_i ^2-i u \alpha _i^2\right)}+\\
   &\frac{\gamma_i  \left((-1)^{3/4} \gamma_i +\sqrt{u} \alpha _i\right) \log \left(\tan
   \left(\tan ^{-1}\left((-1)^{3/4} \sqrt{u}\right)-\frac{1}{2} (-1)^{3/4} t \sqrt{u}
   \alpha _i \sigma _i^2\right)+i\right)}{\sqrt{u} \alpha _i \sigma _i^2 \left(\gamma_i
   ^2-i u \alpha _i^2\right)}+\\&\frac{\sqrt[4]{-1} \gamma_i  \left(\gamma_i +\sqrt[4]{-1} \sqrt{u} \alpha _i\right) \log
   \left(-\tan \left(\tan ^{-1}\left((-1)^{3/4} \sqrt{u}\right)-\frac{1}{2} (-1)^{3/4} t
   \sqrt{u} \alpha _i \sigma _i^2\right)+i\right)}{\sqrt{u} \sigma _i^2 \left(u \alpha
   _i^3+i \gamma_i ^2 \alpha _i\right)}
\end{aligned}$$ The solution for $A$ is then $$\begin{aligned}
    A(t) = \sum_{i=1}^n iu\alpha_i\beta_i t +\lambda_i(F_i(t) -F_i(0))
\end{aligned}$$
