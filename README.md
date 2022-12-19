# Respecting causality is all you need for training physics-informed neural networks

# ⚠️The proposed causal training algorithm cannot be used for commercial purposes (protected by a patent at the University of Pennsylvania).⚠️

Code and data (available upon request) accompanying the manuscript titled "[Respecting causality is all you need for training physics-informed neural networks](https://arxiv.org/abs/2203.07404)", authored by Sifan Wang, Shyam Sankaran, and Paris Perdikaris.

# Abstract

While the popularity of physics-informed neural networks (PINNs) is steadily rising, to this date PINNs have not been successful in simulating dynamical systems whose solution exhibits multi-scale, chaotic or turbulent behavior. In this work we attribute this shortcoming to the inability of existing PINNs formulations to respect the spatio-temporal causal structure that is inherent to the evolution of physical systems. We argue that this is a fundamental limitation and a key source of error that can ultimately steer PINN models to converge towards erroneous solutions. We address this  pathology by proposing a simple re-formulation of PINNs loss functions that can explicitly account for physical causality during model training. We demonstrate that this simple modification alone is enough to introduce significant accuracy improvements, as well as a practical quantitative mechanism for assessing the convergence of a PINNs model. We provide state-of-the-art numerical results across a series of benchmarks for which existing PINNs formulations fail, including the chaotic Lorenz system, the Kuramoto–Sivashinsky equation in the chaotic regime, and the Navier-Stokes equations in the turbulent regime. To the best of our knowledge, this is the first time that PINNs have been successful in simulating such systems, introducing new opportunities for their applicability to problems of industrial complexity.

# Citation

    @article{wang2022respecting,
      title={Respecting causality is all you need for training physics-informed neural networks},
      author={Wang, Sifan and Sankaran, Shyam and Perdikaris, Paris},
      journal={arXiv preprint arXiv:2203.07404},
      year={2022}
    }


# Examples

# Allen–Cahn equation

https://user-images.githubusercontent.com/70182613/160253357-7936e254-ba60-4a9d-abd6-de761e3075c9.mp4

### Kuramoto–Sivashinsky equation

https://user-images.githubusercontent.com/3844367/152894380-3910ee92-6f9b-473b-9942-3d3919f2f22d.mp4

### Navier-Stokes equation

https://user-images.githubusercontent.com/3844367/152894393-6fbc5e1e-f2b0-419e-aa74-3ecb17d0e23e.mp4

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
