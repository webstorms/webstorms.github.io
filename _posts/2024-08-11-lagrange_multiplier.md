## Lagrange multiplier

I recently stumbled upon an interesting problem:

> Determine values $x_i$ that maximizes $\sum_t a_tx_t - w_tx_t^2$ subject to constraint $\sum_t b_tx_t=0$. It is given that all $w_i$ values are positive and that the full array lengths do not fit into memory (hence they need to be loaded in batches).

I initially tried deriving an analytical solution to this problem. If I could not derive a solution, there is always some good ol backprop to fallback on as a backup plan: load arrays as batches and minimize the following objective function:

$$
\mathcal{L}=-\sum_t(a_tx_t - w_tx_t^2) + \lambda \sum_t | b_tx_t |
$$

However, this approach seems a bit iffy: if $\lambda$ is set too low the constraint is not satisfied. Furthermore, if $\lambda$ is set too high the function is not maximized...

After some doodling I recalled there was some constrained optimization trick from my undergraduate calculus course called [Lagrange Multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier), which allows you to find function maxima (or minima) under equation constraints.

**Method of Lagrange Multipliers:**  Suppose we have differentiable functions $f$ and $g$ where we want to maximize $f(\mathbf{x})$ subjected to constraint $g(\mathbf{x})=0$. If $\mathbf{x}$ is a local maximum of $f$ subjected to constraint $g$, and if $\nabla g(\mathbf{x}) \neq 0$, then there exists $\lambda \in \mathbb{R}$ such that the following systems of equations is satisfied by $\lambda$ and $\mathbf{x}$:

$$
\begin{align}
\nabla f(\mathbf{x}) + \lambda \nabla g(\mathbf{x}) &= \mathbf{0}\\
g(\mathbf{x}) &= 0
\end{align}
$$

**Method intuition:** Let's construct function $\mathcal{L}(\mathbf{x}, \lambda)=f(\mathbf{x})+\lambda g(\mathbf{x})$ (which is called the Lagrange function). When our constraint $g(\mathbf{x})=0$ is enforced we have $\mathcal{L}(\mathbf{x}, \lambda)=f(\mathbf{x})$. Thus, optimizing function $f(\mathbf{x})$ under the constraint is equivalent to optimizing function $\mathcal{L}(\mathbf{x}, \lambda)$ under the constraint. To find the extremum of $\mathcal{L}(\mathbf{x}, \lambda)$ we set all partial derivatives $\frac{\partial \mathcal{L}}{\partial x_i}=0$ which gives us 

$$
\begin{aligned} 
\nabla \mathcal{L}(\mathbf{x}, \lambda) &= \nabla f(\mathbf{x}) + \lambda \nabla g(\mathbf{x})\\
\mathbf{0} &= \nabla f(\mathbf{x}) + \lambda \nabla g(\mathbf{x})
\end{aligned}
$$

We also set $\frac{\partial \mathcal{L}}{\partial \lambda}=0$. As $\frac{\partial \mathcal{L}}{\partial \lambda}=g(\mathbf{x})$, then $\frac{\partial \mathcal{L}}{\partial \lambda}=0$ implies that $g(\mathbf{x})=0$ (i.e. our constraint). These conditions give us a system of $n+1$ equations for solving $n+1$ unknown variables ($x_1, x_2, \dots, x_n, \lambda$) which form the solution.

$$
\begin{array}{l}
\frac{\partial}{\partial x_1} f(\mathbf{x}) + \lambda \frac{\partial}{\partial x_1} g(\mathbf{x}) &= 0, \\
\frac{\partial}{\partial x_2} f(\mathbf{x}) + \lambda \frac{\partial}{\partial x_2} g(\mathbf{x}) &= 0, \\
&\vdots\\
\frac{\partial}{\partial x_n} f(\mathbf{x}) + \lambda \frac{\partial}{\partial x_n} g(\mathbf{x}) &= 0, \\
g(\mathbf{x}) &= 0
\end{array}
$$

Note: we require $\nabla g(\mathbf{x}) \neq 0$ otherwise we would find the maxima of $f$ with no dependence on $g$ - but the whole goal is to find the maxima of $f$ constrained by $g$.

**Food for thought, relation to the naive-backprop approach:** In the gradient descent approach I was thinking of minimizing the function

$$
\mathcal{L}=-f(\mathbf{x}) + \lambda |g(\mathbf{x})|
$$

This expression is similar to the Lagrange function with the exception of the L1 norm placed on the constraint function $g$ (also, here the negative of $f$ is minimized rather than maximizing $f$ - which is equivalent). Perhaps minimizing this expression would result in a similar answer as using the Lagrangian multiplier approach if the same $\lambda$ is employed. However, I have not derived this or tested it empirically.

**Solving our problem:**

We start by constructing the Lagrange function $\nabla \mathcal{L}(\mathbf{x}, \lambda)$ of our expression and calculating the partial derivative of it with respect to each $x_i$:

$$
\begin{aligned} 
\frac{\partial}{\partial x_i} \mathcal{L}(\mathbf{x}, \lambda) &= \frac{\partial}{\partial x_i} (f(\mathbf{x}) + g(\mathbf{x}))\\
&= \frac{\partial}{\partial x_i} (\sum_t a_tx_t - w_tx_t^2 + \lambda \sum_t b_t x_t)\\
&= a_i - 2w_ix_i + \lambda b_i
\end{aligned}
$$

Now, setting this expression equal to zero and solving for $x_i$ gives us $x_i = \frac{a_i+\lambda b_i}{2w_i}$. We can then plug each $x_i$ into our constraint $g$ to determine $\lambda$:

$$
\begin{aligned} 
0 &= \sum_t b_t x_t\\
0 &= \sum_t b_t (\frac{a_t+\lambda b_t}{2w_t})\\
0 &=  \sum_t \frac{b_t a_t}{2w_t} + \lambda \sum_t \frac{b_t^2}{2w_t}\\
\lambda &= \frac{-\sum_t \frac{b_t a_t}{2w_t}}{\sum_t \frac{b_t^2}{2w_t}}\\
\end{aligned}
$$

Et voila! This gives us each $x_i$ that maximizes $f$ while enforcing constraint function $g$. Furthermore, we do not need to load all the data into memory (as stipulated in the problem setup) and can calculate $\lambda$ (and $x_i$) in batches of data. Super super cool. Sidenote: we know our solution is a maximum as $\frac{\partial^2}{\partial^2 x_i} \mathcal{L}(\mathbf{x}, \lambda) = -w_i$. As all $w_i$ are positive we get that the second-order partial derivatives are negative, hence we find a maxima solution.
