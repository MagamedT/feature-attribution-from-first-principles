# feature-attribution-from-first-principles

Implementation of the feature attribution framework introduced in [“Feature Attribution from First Principles”](https://arxiv.org/abs/2505.24729). This code provides a simple interface to compute both **global** and **local** feature attributions for differentiable models by approximating the integrals described in the paper via Riemann sums or Monte-Carlo sampling.


## Overview

The “Feature Attribution from First Principles” framework derives feature attributions by formulating them as Stieltjes integrals over the input space. In practice, one must approximate these integrals numerically. This repository provides:

1. **`fam.py`**: Core implementation of the Stieltjes attribution methods.  
2. **`framework_illustration.ipynb`**: Jupyter notebook with illustrative examples for linear and small ReLU networks.  

Sum up:
- Supports **global** attribution (explaining model behavior over the entire input distribution).  
- Supports **local** attribution (explaining the prediction at a specific input point).  
- Two approximation strategies:
  - **Riemann‐sum**: grid‐based approach.
  - **Monte Carlo**: sampling approach (can be adapted by overwriting the sampler).

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/MagamedT/feature-attribution-from-first-principles.git
   cd feature-attribution-from-first-principles
