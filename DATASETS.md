# StreamArena — Full Dataset Tables

Exact instance/feature counts computed directly from each CSV. `#Classes` is the number of unique values in the label column; for regression, the `Target` column name is shown instead. For anomaly detection, the percentage shown is the minority-class (anomaly) rate.

The `Source` column gives a best-effort attribution to each dataset's original public origin, keyed to full BibTeX entries in the [References](#references) section at the bottom. These are best-effort identifications based on well-known dataset naming conventions — verify against the original publication before citing precisely in academic work.

## Classification

### Real-world

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| Dota | 11,055 | 30 | 2 | OpenML [openml] |
| Gisette | 7,000 | 5,000 | 2 | NIPS 2003 FS Challenge [guyon2004] |
| HAR | 10,299 | 561 | 2 | UCI [uci] — Human Activity Recognition Using Smartphones |
| KDDCup99 | 100,000 | 41 | 2 | UCI [uci] — KDD Cup 1999 |
| MNIST | 70,000 | 784 | 2 | LeCun et al. [lecun1998] |
| Spambase | 4,601 | 57 | 2 | UCI [uci] — Spambase |
| Usenet | 18,846 | 658 | 20 | Katakis et al. [katakis2010] |
| adult | 48,842 | 14 | 4 | UCI [uci] — Adult / Census Income |
| airlines | 539,383 | 7 | 2 | OpenML [openml] — Airlines |
| electricity | 45,312 | 8 | 2 | Harries [harries1999] — Electricity Market |
| forest_cover | 581,012 | 54 | 7 | UCI [uci] — Covertype |
| insects | 52,848 | 33 | 6 | Souza et al. [souza2020] |
| nomao | 34,465 | 118 | 2 | UCI [uci] — Nomao |
| poker | 829,201 | 10 | 10 | UCI [uci] — Poker Hand |
| real_kdd99 | 100,655 | 38 | 12 | UCI [uci] — KDD Cup 1999 (resampled) |
| real_pendigits | 10,992 | 16 | 10 | UCI [uci] — Pen-Based Recognition of Handwritten Digits |
| real_powersupply | 29,928 | 2 | 24 | Zhu [zhu2010] |
| real_sensor | 2,219,803 | 5 | 55 | Zhu [zhu2010] |
| real_shuttle | 58,000 | 9 | 7 | UCI [uci] — Statlog (Shuttle), resampled |
| shuttle | 58,000 | 9 | 7 | UCI [uci] — Statlog (Shuttle) |
| spam | 4,601 | 57 | 2 | Concept-drift spam benchmark [katakis2010] |
| vehicle_sensIT | 98,528 | 100 | 3 | UCI [uci] — SensIT Vehicle |
| weather | 18,159 | 8 | 2 | Elwell & Polikar [elwell2011] |

### Synthetic

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| Madelon | 2,600 | 500 | 2 | Guyon et al. [guyon2004] |
| RBF | 10,000 | 10,000 | 2 | MOA [bifet2010] — RandomRBF generator |
| RBFm_100k | 100,000 | 10 | 5 | MOA [bifet2010] — RandomRBF generator |
| RTG_2abrupt | 100,000 | 30 | 5 | MOA [bifet2010] — RandomTreeGenerator |
| RTG_highdim_10k | 10,000 | 450 | 2 | MOA [bifet2010] — RandomTreeGenerator |
| hyperplane_high_gradual_drift | 500,000 | 10 | 2 | MOA/River [bifet2010][montiel2021] — Hyperplane |
| movingRBF | 200,000 | 10 | 5 | MOA [bifet2010] — Moving RandomRBF |
| moving_squares | 200,000 | 2 | 4 | MOA [bifet2010] — Moving Squares |
| sea_high_abrupt_drift | 500,000 | 3 | 2 | Street & Kim [street2001] — SEA generator |
| sea_high_mixed_drift | 500,000 | 3 | 2 | Street & Kim [street2001] — SEA generator |
| sine_stream_with_drift | 50,000 | 4 | 2 | MOA/River [bifet2010][montiel2021] — Sine |
| synth_agrawal | 100,000 | 9 | 2 | Agrawal et al. [agrawal1993] |
| synth_blobs_expanding | 100,000 | 5 | 5 | scikit-learn [pedregosa2011] — make_blobs |
| synth_blobs_gradual | 100,000 | 5 | 5 | scikit-learn [pedregosa2011] — make_blobs |
| synth_blobs_incremental | 100,000 | 5 | 5 | scikit-learn [pedregosa2011] — make_blobs |
| synth_blobs_merge_split | 100,000 | 5 | 5 | scikit-learn [pedregosa2011] — make_blobs |
| synth_blobs_sudden | 100,000 | 5 | 5 | scikit-learn [pedregosa2011] — make_blobs |
| synth_rbf_fast | 100,000 | 5 | 5 | MOA [bifet2010] — RandomRBF generator |
| synth_rbf_gradual | 100,000 | 5 | 5 | MOA [bifet2010] — RandomRBF generator |
| synth_rbf_random | 100,000 | 4 | 4 | MOA [bifet2010] — RandomRBF generator |


## Regression

### Real-world

| Dataset | #Instances | #Features | Target | Source |
|---|---|---|---|---|
| House8L | 22,784 | 8 | target | DELVE [delve] |
| MetroTraffic | 48,204 | 7 | target | UCI [uci] — Metro Interstate Traffic Volume |
| WGN0331_data | 15,628 | 7 | Avg$PerMWHr | NZ Electricity Market (EMI) — node pricing data |
| abalone | 4,977 | 8 | target | UCI [uci] — Abalone |
| ailerons | 13,750 | 40 | target | DELVE [delve] — Ailerons |
| bike | 17,379 | 12 | cnt | UCI/Kaggle [uci] — Bike Sharing Demand |
| brazilian_houses | 10,692 | 12 | total | OpenML [openml] — Brazilian Houses to Rent |
| california_housing | 20,640 | 8 | medianHouseValue | StatLib — California Housing |
| cps88wages | 28,155 | 6 | wage | Current Population Survey 1988 |
| cpu_activity | 8,192 | 21 | usr | DELVE [delve] — Comp-Activ |
| diamonds | 53,940 | 9 | price | ggplot2/Kaggle — Diamonds |
| elevators | 16,599 | 18 | target | DELVE [delve] — Elevators |
| fifa | 19,178 | 28 | wage_eur | Kaggle — FIFA player ratings |
| health_insurance | 22,272 | 12 | whrswk | AER package (R) — Health Insurance |
| kin8nm | 8,192 | 8 | y | DELVE [delve] — Kinematics of Robot Arm |
| kings_county | 21,613 | 21 | price | Kaggle — King County House Sales |
| miami_housing | 13,932 | 16 | SALE_PRC | Kaggle — Miami Housing |
| naval_propulsion_plant | 11,934 | 15 | gt_compressor_decay_state_coefficient | UCI [uci] — Naval Propulsion Plants |
| physiochemical_protein | 45,730 | 9 | RMSD | UCI [uci] — Physicochemical Properties of Protein Tertiary Structure |
| pumadyn32nh | 8,192 | 32 | thetadd6 | DELVE [delve] — Puma Dynamics |
| sarcos | 48,933 | 27 | V22 | SARCOS — robot arm inverse dynamics |
| superconductivity | 21,263 | 81 | critical_temp | UCI [uci] — Superconductivity Data |
| video_transcoding_noID | 68,784 | 19 | utime | UCI [uci] — Online Video Characteristics and Transcoding Time |
| wave_energy | 72,000 | 48 | energy_total | UCI [uci] — Large-scale Wave Energy Farm |
| white_wine | 4,898 | 11 | quality | UCI [uci] — Wine Quality (white) |

### Synthetic

| Dataset | #Instances | #Features | Target | Source |
|---|---|---|---|---|
| FriedmanGra | 100,000 | 10 | target | River [montiel2021] — FriedmanDrift (global recurring abrupt) |
| FriedmanGsg | 99,972 | 10 | target | River [montiel2021] — FriedmanDrift (global slow gradual) |
| FriedmanLea | 100,000 | 10 | target | River [montiel2021] — FriedmanDrift (local expanding abrupt) |
| fried | 40,768 | 10 | target | Friedman [friedman1991] |
| hyperA | 500,000 | 10 | target | MOA/River [bifet2010][montiel2021] — Hyperplane (regression) |


## Clustering

### Real-world

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| adult | 48,842 | 14 | 4 | UCI [uci] — Adult / Census Income |
| electricity | 45,312 | 8 | 2 | Harries [harries1999] — Electricity Market |
| forest_cover | 581,012 | 54 | 7 | UCI [uci] — Covertype |
| insects | 52,848 | 33 | 6 | Souza et al. [souza2020] |
| new_airlines | 539,383 | 7 | 2 | OpenML [openml] — Airlines |
| vehicle_sensIT | 98,528 | 100 | 3 | UCI [uci] — SensIT Vehicle |

### Synthetic

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| hyperplane_high_gradual_drift | 500,000 | 10 | 2 | MOA/River [bifet2010][montiel2021] — Hyperplane |
| movingRBF | 200,000 | 10 | 5 | MOA [bifet2010] — Moving RandomRBF |
| moving_squares | 200,000 | 2 | 4 | MOA [bifet2010] — Moving Squares |
| sea_high_abrupt_drift | 500,000 | 3 | 2 | Street & Kim [street2001] — SEA generator |
| sea_high_mixed_drift | 500,000 | 3 | 2 | Street & Kim [street2001] — SEA generator |
| synth_RandomRBFDrift | 100,000 | 4 | 4 | MOA [bifet2010] — RandomRBF generator |
| synthetic_blobs_100k_samples_5features_8clusters | 100,000 | 5 | 8 | scikit-learn [pedregosa2011] — make_blobs |


## Anomaly Detection

All 51 datasets are drawn from the ODDS Library [rayana2016] and/or the ADBench [han2022] outlier-detection benchmark suite (both aggregate datasets originally published individually, e.g. via UCI [uci]).

| Dataset | #Instances | #Features | Anomaly Rate | Source |
|---|---|---|---|---|
| 10_cover | 286,048 | 10 | 0.96% | ODDS [rayana2016] / ADBench [han2022] |
| 11_donors | 619,326 | 10 | 5.93% | ODDS [rayana2016] / ADBench [han2022] |
| 12_fault | 1,941 | 27 | 34.67% | ODDS [rayana2016] / ADBench [han2022] |
| 13_fraud | 284,807 | 29 | 0.17% | ODDS [rayana2016] / ADBench [han2022] |
| 14_glass | 214 | 7 | 4.21% | ODDS [rayana2016] / ADBench [han2022] |
| 15_Hepatitis | 80 | 19 | 16.25% | ODDS [rayana2016] / ADBench [han2022] |
| 16_http | 567,498 | 3 | 0.39% | ODDS [rayana2016] / ADBench [han2022] |
| 17_InternetAds | 1,966 | 1,555 | 18.72% | ODDS [rayana2016] / ADBench [han2022] |
| 18_Ionosphere | 351 | 32 | 35.9% | ODDS [rayana2016] / ADBench [han2022] |
| 19_landsat | 6,435 | 36 | 20.71% | ODDS [rayana2016] / ADBench [han2022] |
| 1_ALOI | 49,534 | 27 | 3.04% | ODDS [rayana2016] / ADBench [han2022] |
| 20_letter | 1,600 | 32 | 6.25% | ODDS [rayana2016] / ADBench [han2022] |
| 21_Lymphography | 148 | 18 | 4.05% | ODDS [rayana2016] / ADBench [han2022] |
| 22_magic.gamma | 19,020 | 10 | 35.16% | ODDS [rayana2016] / ADBench [han2022] |
| 23_mammography | 11,183 | 6 | 2.32% | ODDS [rayana2016] / ADBench [han2022] |
| 24_mnist | 7,603 | 100 | 9.21% | ODDS [rayana2016] / ADBench [han2022] |
| 25_musk | 3,062 | 166 | 3.17% | ODDS [rayana2016] / ADBench [han2022] |
| 26_optdigits | 5,216 | 64 | 2.88% | ODDS [rayana2016] / ADBench [han2022] |
| 27_PageBlocks | 5,393 | 10 | 9.46% | ODDS [rayana2016] / ADBench [han2022] |
| 28_pendigits | 6,870 | 16 | 2.27% | ODDS [rayana2016] / ADBench [han2022] |
| 29_Pima | 768 | 8 | 34.9% | ODDS [rayana2016] / ADBench [han2022] |
| 2_annthyroid | 7,200 | 6 | 7.42% | ODDS [rayana2016] / ADBench [han2022] |
| 30_satellite | 6,435 | 36 | 31.64% | ODDS [rayana2016] / ADBench [han2022] |
| 31_satimage-2 | 5,803 | 36 | 1.22% | ODDS [rayana2016] / ADBench [han2022] |
| 32_shuttle | 49,097 | 9 | 7.15% | ODDS [rayana2016] / ADBench [han2022] |
| 33_skin | 245,057 | 3 | 20.75% | ODDS [rayana2016] / ADBench [han2022] |
| 34_smtp | 95,156 | 3 | 0.03% | ODDS [rayana2016] / ADBench [han2022] |
| 35_SpamBase | 4,207 | 57 | 39.91% | ODDS [rayana2016] / ADBench [han2022] |
| 36_speech | 3,686 | 400 | 1.65% | ODDS [rayana2016] / ADBench [han2022] |
| 37_Stamps | 340 | 9 | 9.12% | ODDS [rayana2016] / ADBench [han2022] |
| 38_thyroid | 3,772 | 6 | 2.47% | ODDS [rayana2016] / ADBench [han2022] |
| 39_vertebral | 240 | 6 | 12.5% | ODDS [rayana2016] / ADBench [han2022] |
| 3_backdoor | 95,329 | 196 | 2.44% | ODDS [rayana2016] / ADBench [han2022] |
| 40_vowels | 1,456 | 12 | 3.43% | ODDS [rayana2016] / ADBench [han2022] |
| 41_Waveform | 3,443 | 21 | 2.9% | ODDS [rayana2016] / ADBench [han2022] |
| 42_WBC | 223 | 9 | 4.48% | ODDS [rayana2016] / ADBench [han2022] |
| 43_WDBC | 367 | 30 | 2.72% | ODDS [rayana2016] / ADBench [han2022] |
| 44_Wilt | 4,819 | 5 | 5.33% | ODDS [rayana2016] / ADBench [han2022] |
| 45_wine | 129 | 13 | 7.75% | ODDS [rayana2016] / ADBench [han2022] |
| 46_WPBC | 198 | 33 | 23.74% | ODDS [rayana2016] / ADBench [han2022] |
| 47_yeast | 1,484 | 8 | 34.16% | ODDS [rayana2016] / ADBench [han2022] |
| 48_chess | 28,056 | 6 | 0.1% | ODDS [rayana2016] / ADBench [han2022] |
| 49_kddcup99_prob | 64,759 | 6 | 6.43% | ODDS [rayana2016] / ADBench [han2022] |
| 4_breastw | 683 | 9 | 34.99% | ODDS [rayana2016] / ADBench [han2022] |
| 50_bank | 41,188 | 10 | 11.27% | ODDS [rayana2016] / ADBench [han2022] |
| 51_kddcup99_u2r | 60,821 | 6 | 0.37% | ODDS [rayana2016] / ADBench [han2022] |
| 5_campaign | 41,188 | 62 | 11.27% | ODDS [rayana2016] / ADBench [han2022] |
| 6_cardio | 1,831 | 21 | 9.61% | ODDS [rayana2016] / ADBench [han2022] |
| 7_Cardiotocography | 2,114 | 21 | 22.04% | ODDS [rayana2016] / ADBench [han2022] |
| 8_celeba | 202,599 | 39 | 2.24% | ODDS [rayana2016] / ADBench [han2022] |
| 9_census | 299,285 | 500 | 6.2% | ODDS [rayana2016] / ADBench [han2022] |

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

