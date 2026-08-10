# StreamArena — Full Dataset Tables

Exact instance/feature counts computed directly from each CSV. `#Classes` is the number of unique values in the label column; for regression, the `Target` column name is shown instead. For anomaly detection, the percentage shown is the minority-class (anomaly) rate.

`Imbalance (IR)` is the class-imbalance ratio — the majority class count divided by the minority class count (1.0 = perfectly balanced). Datasets with IR ≥ 3 are flagged as imbalanced on the [dataset catalog](https://techynilesh.github.io/StreamArena/datasets.html); on those, prefer the chance-corrected kappa metrics over raw accuracy.

The `Source` column gives a best-effort attribution to each dataset's original public origin, keyed to full BibTeX entries in the [References](#references) section at the bottom. These are best-effort identifications based on well-known dataset naming conventions — verify against the original publication before citing precisely in academic work.

## Classification

### Real-world

| Dataset | #Instances | #Features | #Classes | Imbalance (IR) | Source |
|---|---|---|---|---|---|
| adult | 48,842 | 14 | 4 | 6.4 | UCI [uci] — Adult / Census Income |
| airlines | 539,383 | 7 | 2 | 1.2 | OpenML [openml] — Airlines |
| electricity | 45,312 | 8 | 2 | 1.4 | Harries [harries1999] — Electricity Market |
| forest_cover | 581,012 | 54 | 7 | 103.1 | UCI [uci] — Covertype |
| insects | 52,848 | 33 | 6 | 1.0 | Souza et al. [souza2020] |
| nomao | 34,465 | 118 | 2 | 2.5 | UCI [uci] — Nomao |
| poker | 829,201 | 10 | 10 | 207763.0 | UCI [uci] — Poker Hand |
| real_powersupply | 29,928 | 2 | 24 | 1.0 | Zhu [zhu2010] |
| real_sensor | 2,219,803 | 5 | 55 | 32.2 | Zhu [zhu2010] |
| weather | 18,159 | 8 | 2 | 2.2 | Elwell & Polikar [elwell2011] |

### Synthetic

| Dataset | #Instances | #Features | #Classes | Imbalance (IR) | Source |
|---|---|---|---|---|---|
| RBF | 10,000 | 10,000 | 2 | 1.1 | MOA [bifet2010] — RandomRBF generator |
| RBFm_100k | 100,000 | 10 | 5 | 3.2 | MOA [bifet2010] — RandomRBF generator |
| RTG_2abrupt | 100,000 | 30 | 5 | 13.4 | MOA [bifet2010] — RandomTreeGenerator |
| RTG_highdim_10k | 10,000 | 450 | 2 | 1.1 | MOA [bifet2010] — RandomTreeGenerator |
| hyperplane_high_gradual_drift | 500,000 | 10 | 2 | 1.0 | MOA/River [bifet2010][montiel2021] — Hyperplane |
| movingRBF | 200,000 | 10 | 5 | 3.3 | MOA [bifet2010] — Moving RandomRBF |
| sea_high_abrupt_drift | 500,000 | 3 | 2 | 1.6 | Street & Kim [street2001] — SEA generator |
| sea_high_mixed_drift | 500,000 | 3 | 2 | 1.4 | Street & Kim [street2001] — SEA generator |
| sine_stream_with_drift | 50,000 | 4 | 2 | 1.0 | MOA/River [bifet2010][montiel2021] — Sine |
| synth_blobs_sudden | 100,000 | 5 | 5 | 1.0 | scikit-learn [pedregosa2011] — make_blobs |
| synth_rbf_fast | 100,000 | 5 | 5 | 1.3 | MOA [bifet2010] — RandomRBF generator |
| synth_rbf_gradual | 100,000 | 5 | 5 | 1.6 | MOA [bifet2010] — RandomRBF generator |
| synth_rbf_random | 100,000 | 4 | 4 | 3.4 | MOA [bifet2010] — RandomRBF generator |


## Regression

### Real-world

| Dataset | #Instances | #Features | Target | Source |
|---|---|---|---|---|
| MetroTraffic | 48,204 | 7 | target | UCI [uci] — Metro Interstate Traffic Volume |
| ailerons | 13,750 | 40 | target | DELVE [delve] — Ailerons |
| bike | 17,379 | 12 | cnt | UCI/Kaggle [uci] — Bike Sharing Demand |
| california_housing | 20,640 | 8 | medianHouseValue | StatLib — California Housing |
| elevators | 16,599 | 18 | target | DELVE [delve] — Elevators |
| fifa | 19,178 | 28 | wage_eur | Kaggle — FIFA player ratings |
| sarcos | 48,933 | 27 | V22 | SARCOS — robot arm inverse dynamics |
| superconductivity | 21,263 | 81 | critical_temp | UCI [uci] — Superconductivity Data |
| video_transcoding_noID | 68,784 | 19 | utime | UCI [uci] — Online Video Characteristics and Transcoding Time |
| wave_energy | 72,000 | 48 | energy_total | UCI [uci] — Large-scale Wave Energy Farm |

### Synthetic

| Dataset | #Instances | #Features | Target | Source |
|---|---|---|---|---|
| FriedmanGra | 100,000 | 10 | target | River [montiel2021] — FriedmanDrift (global recurring abrupt) |
| FriedmanGsg | 99,971 | 10 | target | River [montiel2021] — FriedmanDrift (global slow gradual) |
| FriedmanLea | 100,000 | 10 | target | River [montiel2021] — FriedmanDrift (local expanding abrupt) |
| fried | 40,768 | 10 | target | Friedman [friedman1991] |
| hyperA | 500,000 | 10 | target | MOA/River [bifet2010][montiel2021] — Hyperplane (regression) |


## Clustering

### Real-world

| Dataset | #Instances | #Features | #Classes | Imbalance (IR) | Source |
|---|---|---|---|---|---|
| adult | 48,842 | 14 | 4 | 6.4 | UCI [uci] — Adult / Census Income |
| electricity | 45,312 | 8 | 2 | 1.4 | Harries [harries1999] — Electricity Market |
| forest_cover | 581,012 | 54 | 7 | 103.1 | UCI [uci] — Covertype |
| insects | 52,848 | 33 | 6 | 1.0 | Souza et al. [souza2020] |
| new_airlines | 539,383 | 7 | 2 | 1.2 | OpenML [openml] — Airlines |
| vehicle_sensIT | 98,528 | 100 | 3 | 2.2 | UCI [uci] — SensIT Vehicle |

### Synthetic

| Dataset | #Instances | #Features | #Classes | Imbalance (IR) | Source |
|---|---|---|---|---|---|
| hyperplane_high_gradual_drift | 500,000 | 10 | 2 | 1.0 | MOA/River [bifet2010][montiel2021] — Hyperplane |
| movingRBF | 200,000 | 10 | 5 | 3.3 | MOA [bifet2010] — Moving RandomRBF |
| moving_squares | 200,000 | 2 | 4 | 1.0 | MOA [bifet2010] — Moving Squares |
| sea_high_abrupt_drift | 500,000 | 3 | 2 | 1.6 | Street & Kim [street2001] — SEA generator |
| sea_high_mixed_drift | 500,000 | 3 | 2 | 1.4 | Street & Kim [street2001] — SEA generator |
| synth_RandomRBFDrift | 100,000 | 4 | 4 | 3.4 | MOA [bifet2010] — RandomRBF generator |
| synthetic_blobs_100k_samples_5features_8clusters | 100,000 | 5 | 8 | 1.0 | scikit-learn [pedregosa2011] — make_blobs |


## Anomaly Detection

All 51 datasets are drawn from the ODDS Library [rayana2016] and/or the ADBench [han2022] outlier-detection benchmark suite (both aggregate datasets originally published individually, e.g. via UCI [uci]).

| Dataset | #Instances | #Features | Anomaly Rate | Source |
|---|---|---|---|---|
| 13_fraud | 284,807 | 29 | 0.17% | ODDS [rayana2016] / ADBench [han2022] |

## References

Full BibTeX for every source cited in the tables above.

```bibtex
@misc{uci,
  title        = {{UCI} Machine Learning Repository},
  author       = {Dua, Dheeru and Graff, Casey},
  year         = {2019},
  institution  = {University of California, Irvine, School of Information and Computer Sciences},
  url          = {https://archive.ics.uci.edu}
}

@article{openml,
  title   = {{OpenML}: Networked Science in Machine Learning},
  author  = {Vanschoren, Joaquin and van Rijn, Jan N. and Bischl, Bernd and Torgo, Luis},
  journal = {ACM SIGKDD Explorations Newsletter},
  volume  = {15},
  number  = {2},
  pages   = {49--60},
  year    = {2013}
}

@misc{delve,
  title        = {{DELVE}: Data for Evaluating Learning in Valid Experiments},
  author       = {Rasmussen, Carl Edward and Neal, Radford M. and Hinton, Geoffrey and van Camp, Drew and Revow, Michael and Ghahramani, Zoubin and Kustra, Rafal and Tibshirani, Robert},
  institution  = {University of Toronto, Department of Computer Science},
  url          = {https://www.cs.toronto.edu/~delve/}
}

@inproceedings{guyon2004,
  title     = {Result Analysis of the {NIPS} 2003 Feature Selection Challenge},
  author    = {Guyon, Isabelle and Gunn, Steve and Ben-Hur, Asa and Dror, Gideon},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {17},
  year      = {2004}
}

@article{lecun1998,
  title   = {Gradient-Based Learning Applied to Document Recognition},
  author  = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal = {Proceedings of the IEEE},
  volume  = {86},
  number  = {11},
  pages   = {2278--2324},
  year    = {1998}
}

@techreport{harries1999,
  title       = {{SPLICE-2} Comparative Evaluation: Electricity Pricing},
  author      = {Harries, Michael},
  institution = {University of New South Wales, School of Computer Science and Engineering},
  year        = {1999}
}

@article{souza2020,
  title   = {Challenges in Benchmarking Stream Learning Algorithms with Real-World Data},
  author  = {Souza, Vinicius M. A. and dos Reis, Denis M. and Maletzke, Andre G. and Batista, Gustavo E. A. P. A.},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {34},
  pages   = {1805--1858},
  year    = {2020}
}

@misc{zhu2010,
  title  = {Stream Data Mining Repository},
  author = {Zhu, Xingquan},
  year   = {2010},
  url    = {https://www.cse.fau.edu/~xqzhu/stream.html}
}

@article{elwell2011,
  title   = {Incremental Learning of Concept Drift in Nonstationary Environments},
  author  = {Elwell, Ryan and Polikar, Robi},
  journal = {IEEE Transactions on Neural Networks},
  volume  = {22},
  number  = {10},
  pages   = {1517--1531},
  year    = {2011}
}

@inproceedings{street2001,
  title     = {A Streaming Ensemble Algorithm ({SEA}) for Large-Scale Classification},
  author    = {Street, W. Nick and Kim, YongSeog},
  booktitle = {Proceedings of the 7th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {377--382},
  year      = {2001}
}

@article{agrawal1993,
  title   = {Database Mining: A Performance Perspective},
  author  = {Agrawal, Rakesh and Imielinski, Tomasz and Swami, Arun},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {5},
  number  = {6},
  pages   = {914--925},
  year    = {1993}
}

@article{bifet2010,
  title   = {{MOA}: Massive Online Analysis},
  author  = {Bifet, Albert and Holmes, Geoffrey and Kirkby, Richard and Pfahringer, Bernhard},
  journal = {Journal of Machine Learning Research},
  volume  = {11},
  pages   = {1601--1604},
  year    = {2010}
}

@article{montiel2021,
  title   = {River: Machine Learning for Streaming Data in Python},
  author  = {Montiel, Jacob and Halford, Max and Mastelini, Saulo Martiello and Bolmier, Geoffrey and Sourty, Raphael and Vaysse, Robin and Zouitine, Adil and Gomes, Heitor Murilo and Read, Jesse and Abdessalem, Talel and Bifet, Albert},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  number  = {110},
  pages   = {1--8},
  year    = {2021}
}

@article{friedman1991,
  title   = {Multivariate Adaptive Regression Splines},
  author  = {Friedman, Jerome H.},
  journal = {The Annals of Statistics},
  volume  = {19},
  number  = {1},
  pages   = {1--67},
  year    = {1991}
}

@article{pedregosa2011,
  title   = {Scikit-learn: Machine Learning in {P}ython},
  author  = {Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and Vanderplas, Jake and Passos, Alexandre and Cournapeau, David and Brucher, Matthieu and Perrot, Matthieu and Duchesnay, {\'E}douard},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}

@article{katakis2010,
  title   = {Tracking Recurring Contexts Using Ensemble Classifiers: An Application to Email Filtering},
  author  = {Katakis, Ioannis and Tsoumakas, Grigorios and Vlahavas, Ioannis},
  journal = {Knowledge and Information Systems},
  volume  = {22},
  number  = {3},
  pages   = {371--391},
  year    = {2010}
}

@misc{rayana2016,
  title        = {{ODDS} Library},
  author       = {Rayana, Shebuti},
  year         = {2016},
  institution  = {Stony Brook University, Department of Computer Science},
  url          = {http://odds.cs.stonybrook.edu}
}

@inproceedings{han2022,
  title     = {{ADBench}: Anomaly Detection Benchmark},
  author    = {Han, Songqiao and Hu, Xiyang and Huang, Hailiang and Jiang, Mingqi and Zhao, Yue},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  volume    = {35},
  year      = {2022}
}
```

