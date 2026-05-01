# JAXPI2

This repository is a comprehensive implementation of physics-informed neural networks (PINNs), 
seamlessly integrating several advanced network architectures, training algorithms from these papers 

- [When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions](https://arxiv.org/abs/2604.23528v1)
- [Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective](https://arxiv.org/abs/2502.00604)
- [Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks](https://epubs.siam.org/doi/10.1137/20M1318043)
- [When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective](https://www.sciencedirect.com/science/article/pii/S002199912100663X?casa_token=YlzVQK6hGy8AAAAA:bKwMNg70UoeEuisR1cd1KZnR20xspdvYp1dM4jLkl_wfVDX7O1j2IOlGZsYnC4esu7YcMaO_WOIC)
- [Respecting Causality for Training Physics-informed Neural Networks](https://www.sciencedirect.com/science/article/pii/S0045782524000690)
- [Random Weight Factorization Improves the Training of Continuous Neural Representations](https://arxiv.org/abs/2210.01274)
- [On the Eigenvector Bias of Fourier Feature Networks: From Regression to Solving Multi-Scale PDEs with Physics-Informed Neural Network](https://www.sciencedirect.com/science/article/abs/pii/S0045782521002759)
- [PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks](https://arxiv.org/abs/2402.00326)
- [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
- [A Method for Representing Periodic Functions and Enforcing Exactly Periodic Boundary Conditions with Deep Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0021999121001376)
- [Characterizing Possible Failure Modes in Physics-Informed Neural Networks](https://arxiv.org/abs/2109.01050)

This  repository also releases an extensive range of benchmarking examples, showcasing the effectiveness and robustness of our implementation.
Our implementation supports both **single** and **multi-GPU** training, while evaluation is currently limited to
single-GPU setups.


## Installation

Ensure that you have Python 3.8 or later installed on your system.
Our code is GPU-only.
We highly recommend using the most recent versions of JAX and JAX-lib, along with compatible CUDA and cuDNN versions.
The code has been tested and confirmed to work with the following versions:

- JAX 0.5.36
- CUDA 12.4
- cuDNN 8.9

You can install the latest versions of JAX and JAX-lib with the following commands, but may have to upgrade some APIs to be compatible with the latest versions.
```
pip3 install -U pip
pip3 install --upgrade jax jaxlib
```

Install JAX-PI with the following commands:

``` 
git clone https://github.com/PredictiveIntelligenceLab/jaxpi.git
cd jaxpi
pip install -e .
```

## Quickstart

We use [Weights & Biases](https://wandb.ai/site) to log and monitor training metrics. 
Please ensure you have Weights & Biases installed and properly set up with your account before proceeding. 
You can follow the installation guide provided [here](https://docs.wandb.ai/quickstart).

To illustrate how to use our code, we will use the advection equation as an example. 
First, navigate to the advection directory within the `examples` folder:

``` 
cd jaxpi/examples/advection
``` 
To train the model, we need to specify the configuration file that contains the training settings and hyperparameters.
```
python3 main.py  --config=configs/baseline.py
```

Our code automatically supports multi-GPU execution. 
You can specify the GPUs you want to use with the `CUDA_VISIBLE_DEVICES` environment variable. For example, to use the first two GPUs (0 and 1), use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=configs/baseline.py
```

**Note on Memory Usage**: Different models and examples may require varying amounts of GPU memory. 
If you encounter an out-of-memory error, you can decrease the batch size using the `--config.training.batch_size` option.

To evaluate the model's performance, you can switch to evaluation mode with the following command:



## Citation

    @article{wang2023expert,
      title={An Expert's Guide to Training Physics-informed Neural Networks},
      author={Wang, Sifan and Sankaran, Shyam and Wang, Hanwen and Perdikaris, Paris},
      journal={arXiv preprint arXiv:2308.08468},
      year={2023}
    }

    @article{wang2024piratenets,
      title={Piratenets: Physics-informed deep learning with residual adaptive networks},
      author={Wang, Sifan and Li, Bowen and Chen, Yuhan and Perdikaris, Paris},
      journal={Journal of Machine Learning Research},
      volume={25},
      number={402},
      pages={1--51},
      year={2024}
    }

    @inproceedings{
        wang2025gradient,
        title={Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective},
        author={Sifan Wang and Ananyae Kumar bhartari and Bowen Li and Paris Perdikaris},
        booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
        year={2025},
        url={https://openreview.net/forum?id=iweeVl1RHU}
    }
  
    @article{wang2026pinns,
      title={When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions},
      author={Wang, Sifan and Koohy, Shawn and Lu, Yiping and Perdikaris, Paris},
      journal={arXiv preprint arXiv:2604.23528},
      year={2026}
    }

    

    


    



