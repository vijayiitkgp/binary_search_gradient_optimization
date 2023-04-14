# Binary Search Gradient Optimization

 A novel stochastic optimization method, which uses the binary search technique with
first order gradient based optimization method, called Binary Search Gradient Optimization (BSG) or BiGrad. In
this optimization setup, a non-convex surface is treated as
a set of convex surfaces. In BSG, at first, a region is defined, assuming region is convex. If region is not convex,
then the algorithm leaves the region very fast and defines
a new one, otherwise, it tries to converge at the optimal
point of the region. In BSG, core purpose of binary search
is to decide, whether region is convex or not in logarithmic
time, whereas, first order gradient based method is primarily applied, to define a new region

## Run Locally
Go to the project directory

```bash
  cd BSG
```

In BSG.py, code of proposed BSG optimizer is written

In other files some definitions of model architecture, different dataset, and its training code is written.

Recommended to use jupyter notebook to run these codes.


