## Conditional Expectations

Conditional expectations are a powerful tool to have in your maths toolkit. It's an elegant technique that can simplify challenging calculations. I briefly state the method and then showcase some example problems of increasing complexity.

**Conditional Expectation** 
If $S=\bigcup_{i=1}^N E_i$ is a union of pairwise disjoint events (i.e. $E_i \bigcap E_j = \emptyset$ for all $i \neq j$), then
$$
\mathbb{E}[R] = \sum_{i=1}^N \mathbb{E}[R \mid E_i] \, Pr(E_i)
$$
for random variable $R: S \rightarrow \mathbb{R}$ and probability function $Pr: S \rightarrow [0, 1]$.

**Dice role problem**: Let's start with a warmup exercise of calculating the expected outcome $R$ of rolling a die. Typically we would calculate this as $\mathbb{E}[R] = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5$. We can split the sample space $S = \\{1, 2, 3, 4, 5, 6\\}$ into two disjoint sets $S = \\{1, 2, 3\\} \bigcup \\{4, 5, 6\\}$. Now we can apply the conditional expectation technique to calculate the expected outcome of rolling a die. 

$$
\begin{aligned} \mathbb{E}[R] &= \mathbb{E}[R | R \leq 3]Pr(R \leq 3) + \mathbb{E}[R | R > 3]Pr(R > 3) \\ 
&= \frac{1}{3}(1 + 2 + 3)\frac{1}{2} + \frac{1}{3}(4 + 5 + 6)\frac{1}{2} \\ 
&= \frac{2}{2} + \frac{5}{2} = 3.5\\ 
\end{aligned}
$$ 

Et voila! It's the same answer. Let's take a look at a more challenging problem.

**Dice role payout problem**: Let's play a game where we roll a die and we get paid the outcome. If the outcome is $1, 2, 3$ the game ends. If the outcome is $4, 5, 6$ we get to roll again. What is the expected payout of this game?

Here we can use conditional expectations. $R \in \\{1, 2, 3\\}$ and $R \in \\{4, 5, 6\\}$ are disjoint events, so we can write our expectation as:

$$
\begin{aligned} \mathbb{E}[R] &= \mathbb{E}[R | R \leq 3]Pr(R \leq 3) + \mathbb{E}[R | R > 3]Pr(R > 3) \\ 
&= \frac{1}{3}(1 + 2 + 3)\frac{1}{2} + (\frac{1}{3}(4 + 5 + 6)+ \mathbb{E}[R])\frac{1}{2}\\
&= 1 + (5 + \mathbb{E}[R])\frac{1}{2}\\
&= 7\\
\end{aligned}
$$ 

Here we include the expectation in the second term as we get to play the game again.

**Noodles in a bowl problem**: Let's say we have $N$ noodles in a bowl. We randomly select one noodle end and randomly select another noodle end and connect them. We continue doing so until we have connected all noodle ends. What is the expected number of noodle-loops that are formed using this process?

This problem appears pretty daunting at first but it has a simple solution using conditional expectations. Let's consider connecting the last noodle which has two disoint events of 1. connecting with another noodle and 2. self-connecting. We can then write the expectation as follows:

$$
\begin{aligned} \mathbb{E}[R] &= \mathbb{E}[R | \text{$N^{th}$ noodle does not self-connect}]Pr(\text{$N^{th}$ noodle does not self-connect})\\
&+ \mathbb{E}[R | \text{$N^{th}$ noodle does self-connect}]Pr(\text{$N^{th}$ noodle does self-connect}) \\ 
\end{aligned}
$$

The probability of the $N^{th}$ noodle self-connecting can be calculated as follows: for $N$ noodles there are $2N$ noodle ends. If we select one of these ends there remain $2N-1$ noodle ends to choose from with one end that results in a self-connection, hence $Pr(\text{$N^{th}$ noodle does self-connect})=\frac{1}{2N-1}$ and $Pr(\text{$N^{th}$ noodle does not self-connect}) = 1 - Pr(\text{$N^{th}$ noodle does self-connect})$.

$$
\begin{aligned} \mathbb{E}[R] &= \mathbb{E}[R | \text{$N^{th}$ noodle does not self-connect}]Pr(\text{$N^{th}$ noodle does not self-connect})\\
&+ \mathbb{E}[R | \text{$N^{th}$ noodle does self-connect}]Pr(\text{$N^{th}$ noodle does self-connect}) \\ 
&= \mathbb{E}[R-1](1 - \frac{1}{2N-1}) + (\mathbb{E}[R-1] + 1)\frac{1}{2N-1}\\
&= \mathbb{E}[R-1] + \frac{1}{2N-1}\\
\end{aligned}
$$

This gives us a recursive formula for calculating the number of expected noodle-loops! Bon appetit.
