# About

JAXPI is developed by [Sifan Wang](https://github.com/sifanexisted) and collaborators. The
library distills the training recipes from the papers listed in [Theory](/theory) into a lean,
tested JAX codebase.

## Citation

If you use JAXPI in your research, please cite:

```bibtex
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
  volume={25}, number={402}, pages={1--51}, year={2024}
}

@inproceedings{wang2025gradient,
  title={Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective},
  author={Sifan Wang and Ananyae Kumar Bhartari and Bowen Li and Paris Perdikaris},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}

@article{wang2026pinns,
  title={When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions},
  author={Wang, Sifan and Koohy, Shawn and Lu, Yiping and Perdikaris, Paris},
  journal={arXiv preprint arXiv:2604.23528},
  year={2026}
}
```

## License

Apache License 2.0 — see
[`LICENSE`](https://github.com/sifanexisted/jaxpi2/blob/main/LICENSE).

## Contributing

Issues and pull requests are welcome on
[GitHub](https://github.com/sifanexisted/jaxpi2). The core is covered by a CPU-runnable test
suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

Please keep multi-component residuals as dicts, run the tests, and add a regression test with
any bug fix.

## Building these docs

```bash
cd docs
npm install
npm run docs:dev     # local preview
npm run docs:build   # static build
```

Gallery assets are generated from the reference data with
`python docs/scripts/gen_gallery.py`; API pages with `python docs/scripts/gen_api.py`.
