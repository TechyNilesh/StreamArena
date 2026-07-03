# StreamArena — Full Dataset Tables

Exact instance/feature counts computed directly from each CSV. `#Classes` is the number of unique values in the label column; for regression, the `Target` column name is shown instead. For anomaly detection, the percentage shown is the minority-class (anomaly) rate.

Sources are best-effort attributions to the datasets' original public origin (UCI, OpenML, DELVE, MOA/River synthetic generators, ODDS/ADBench, etc.). Verify against the original publication before citing in academic work.

## Classification

### Real-world

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| Dota | 11,055 | 30 | 2 | OpenML / UCI — Dota2 Games Results |
| Gisette | 7,000 | 5,000 | 2 | NIPS 2003 Feature Selection Challenge / UCI — Gisette |
| HAR | 10,299 | 561 | 2 | UCI — Human Activity Recognition Using Smartphones |
| KDDCup99 | 100,000 | 41 | 2 | UCI / KDD Cup 1999 — network intrusion detection |
| MNIST | 70,000 | 784 | 2 | LeCun et al. — MNIST handwritten digits |
| Spambase | 4,601 | 57 | 2 | UCI — Spambase |
| Usenet | 18,846 | 658 | 20 | Katakis et al. — Usenet recurring-drift benchmark |
| adult | 48,842 | 14 | 4 | UCI — Adult / Census Income |
| airlines | 539,383 | 7 | 2 | OpenML — Airlines (flight delay) stream benchmark |
| electricity | 45,312 | 8 | 2 | Harries (1999) — Electricity Market, classic drift benchmark |
| forest_cover | 581,012 | 54 | 7 | UCI — Covertype |
| insects | 52,848 | 33 | 6 | Souza et al. — Insects, real-world drift benchmark |
| nomao | 34,465 | 118 | 2 | UCI — Nomao |
| poker | 829,201 | 10 | 10 | UCI — Poker Hand |
| real_kdd99 | 100,655 | 38 | 12 | UCI / KDD Cup 1999 (resampled variant) |
| real_pendigits | 10,992 | 16 | 10 | UCI — Pen-Based Recognition of Handwritten Digits |
| real_powersupply | 29,928 | 2 | 24 | Zhu et al. — PowerSupply drift benchmark |
| real_sensor | 2,219,803 | 5 | 55 | Zhu et al. — Sensor stream drift benchmark |
| real_shuttle | 58,000 | 9 | 7 | UCI — Statlog (Shuttle) (resampled variant) |
| shuttle | 58,000 | 9 | 7 | UCI — Statlog (Shuttle) |
| spam | 4,601 | 57 | 2 | Concept-drift spam/Twitter-spam benchmark |
| vehicle_sensIT | 98,528 | 100 | 3 | UCI — SensIT Vehicle |
| weather | 18,159 | 8 | 2 | Elwell & Polikar — NOAA Weather, real-world drift benchmark |

### Synthetic

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| Madelon | 2,600 | 500 | 2 | Guyon et al. — NIPS 2003 Feature Selection Challenge (synthetic) |
| RBF | 10,000 | 10,000 | 2 | MOA/River — RandomRBF generator |
| RBFm_100k | 100,000 | 10 | 5 | MOA/River — RandomRBF generator |
| RTG_2abrupt | 100,000 | 30 | 5 | MOA/River — RandomTreeGenerator |
| RTG_highdim_10k | 10,000 | 450 | 2 | MOA/River — RandomTreeGenerator |
| hyperplane_high_gradual_drift | 500,000 | 10 | 2 | MOA/River — Hyperplane generator |
| movingRBF | 200,000 | 10 | 5 | MOA/River — Moving RandomRBF generator |
| moving_squares | 200,000 | 2 | 4 | MOA — Moving Squares synthetic generator |
| rtg | 100,000 | 30 | 5 | MOA/River — RandomTreeGenerator |
| sea_high_abrupt_drift | 500,000 | 3 | 2 | Street & Kim (2001) — SEA generator |
| sea_high_mixed_drift | 500,000 | 3 | 2 | Street & Kim (2001) — SEA generator |
| sine_stream_with_drift | 50,000 | 4 | 2 | MOA/River — Sine generator |
| synth_agrawal | 100,000 | 9 | 2 | Agrawal et al. (1993) — Agrawal generator |
| synth_blobs_expanding | 100,000 | 5 | 5 | scikit-learn make_blobs — synthetic clustering-drift generator |
| synth_blobs_gradual | 100,000 | 5 | 5 | scikit-learn make_blobs — synthetic clustering-drift generator |
| synth_blobs_incremental | 100,000 | 5 | 5 | scikit-learn make_blobs — synthetic clustering-drift generator |
| synth_blobs_merge_split | 100,000 | 5 | 5 | scikit-learn make_blobs — synthetic clustering-drift generator |
| synth_blobs_sudden | 100,000 | 5 | 5 | scikit-learn make_blobs — synthetic clustering-drift generator |
| synth_rbf_fast | 100,000 | 5 | 5 | MOA/River — RandomRBF generator |
| synth_rbf_gradual | 100,000 | 5 | 5 | MOA/River — RandomRBF generator |
| synth_rbf_random | 100,000 | 4 | 4 | MOA/River — RandomRBF generator |


## Regression

### Real-world

| Dataset | #Instances | #Features | Target | Source |
|---|---|---|---|---|
| House8L | 22,784 | 8 | target | DELVE — House8L |
| MetroTraffic | 48,204 | 7 | target | UCI — Metro Interstate Traffic Volume |
| WGN0331_data | 15,628 | 7 | Avg$PerMWHr | NZ Electricity Market — node pricing data |
| abalone | 4,977 | 8 | target | UCI — Abalone |
| ailerons | 13,750 | 40 | target | DELVE — Ailerons (F16 aircraft control) |
| bike | 17,379 | 12 | cnt | UCI/Kaggle — Bike Sharing Demand |
| brazilian_houses | 10,692 | 12 | total | Kaggle/OpenML — Brazilian Houses to Rent |
| california_housing | 20,640 | 8 | medianHouseValue | StatLib — California Housing |
| cps88wages | 28,155 | 6 | wage | Current Population Survey 1988 — Wages (econometrics) |
| cpu_activity | 8,192 | 21 | usr | DELVE — Comp-Activ / CPU Activity |
| diamonds | 53,940 | 9 | price | ggplot2/Kaggle — Diamonds |
| elevators | 16,599 | 18 | target | DELVE — Elevators (F16 aircraft control) |
| fifa | 19,178 | 28 | wage_eur | Kaggle — FIFA player ratings |
| health_insurance | 22,272 | 12 | whrswk | AER package — Health Insurance (econometrics) |
| kin8nm | 8,192 | 8 | y | DELVE — Kinematics of Robot Arm (kin8nm) |
| kings_county | 21,613 | 21 | price | Kaggle — King County House Sales |
| miami_housing | 13,932 | 16 | SALE_PRC | Kaggle — Miami Housing |
| naval_propulsion_plant | 11,934 | 15 | gt_compressor_decay_state_coefficient | UCI — Condition Based Maintenance of Naval Propulsion Plants |
| physiochemical_protein | 45,730 | 9 | RMSD | UCI — Physicochemical Properties of Protein Tertiary Structure |
| pumadyn32nh | 8,192 | 32 | thetadd6 | DELVE — Puma Dynamics (pumadyn-32nh) |
| sarcos | 48,933 | 27 | V22 | SARCOS — robot arm inverse dynamics |
| superconductivity | 21,263 | 81 | critical_temp | UCI — Superconductivity Data |
| video_transcoding_noID | 68,784 | 19 | utime | UCI — Online Video Characteristics and Transcoding Time |
| wave_energy | 72,000 | 48 | energy_total | UCI — Large-scale Wave Energy Farm |
| white_wine | 4,898 | 11 | quality | UCI — Wine Quality (white) |

### Synthetic

| Dataset | #Instances | #Features | Target | Source |
|---|---|---|---|---|
| FriedmanGra | 100,000 | 10 | target | River — FriedmanDrift generator (global recurring abrupt) |
| FriedmanGsg | 99,972 | 10 | target | River — FriedmanDrift generator (global slow gradual) |
| FriedmanLea | 100,000 | 10 | target | River — FriedmanDrift generator (local expanding abrupt) |
| fried | 40,768 | 10 | target | Friedman (1991) — synthetic Friedman function |
| hyperA | 500,000 | 10 | target | MOA/River — Hyperplane generator (regression variant) |


## Clustering

### Real-world

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| adult | 48,842 | 14 | 4 | UCI — Adult / Census Income |
| electricity | 45,312 | 8 | 2 | Harries (1999) — Electricity Market, classic drift benchmark |
| forest_cover | 581,012 | 54 | 7 | UCI — Covertype |
| insects | 52,848 | 33 | 6 | Souza et al. — Insects, real-world drift benchmark |
| new_airlines | 539,383 | 7 | 2 | OpenML — Airlines (flight delay) stream benchmark |
| vehicle_sensIT | 98,528 | 100 | 3 | UCI — SensIT Vehicle |

### Synthetic

| Dataset | #Instances | #Features | #Classes | Source |
|---|---|---|---|---|
| hyperplane_high_gradual_drift | 500,000 | 10 | 2 | MOA/River — Hyperplane generator |
| movingRBF | 200,000 | 10 | 5 | MOA/River — Moving RandomRBF generator |
| moving_squares | 200,000 | 2 | 4 | MOA — Moving Squares synthetic generator |
| sea_high_abrupt_drift | 500,000 | 3 | 2 | Street & Kim (2001) — SEA generator |
| sea_high_mixed_drift | 500,000 | 3 | 2 | Street & Kim (2001) — SEA generator |
| synth_RandomRBFDrift | 100,000 | 4 | 4 | MOA/River — RandomRBF generator |
| synthetic_blobs_100k_samples_5features_8clusters | 100,000 | 5 | 8 | scikit-learn make_blobs — synthetic clustering generator |


## Anomaly Detection

All 51 datasets are from the ODDS Library (Rayana, Stony Brook) / ADBench (Han et al., 2022) outlier-detection benchmark suite.

| Dataset | #Instances | #Features | Anomaly Rate |
|---|---|---|---|
| 10_cover | 286,048 | 10 | 0.96% |
| 11_donors | 619,326 | 10 | 5.93% |
| 12_fault | 1,941 | 27 | 34.67% |
| 13_fraud | 284,807 | 29 | 0.17% |
| 14_glass | 214 | 7 | 4.21% |
| 15_Hepatitis | 80 | 19 | 16.25% |
| 16_http | 567,498 | 3 | 0.39% |
| 17_InternetAds | 1,966 | 1,555 | 18.72% |
| 18_Ionosphere | 351 | 32 | 35.9% |
| 19_landsat | 6,435 | 36 | 20.71% |
| 1_ALOI | 49,534 | 27 | 3.04% |
| 20_letter | 1,600 | 32 | 6.25% |
| 21_Lymphography | 148 | 18 | 4.05% |
| 22_magic.gamma | 19,020 | 10 | 35.16% |
| 23_mammography | 11,183 | 6 | 2.32% |
| 24_mnist | 7,603 | 100 | 9.21% |
| 25_musk | 3,062 | 166 | 3.17% |
| 26_optdigits | 5,216 | 64 | 2.88% |
| 27_PageBlocks | 5,393 | 10 | 9.46% |
| 28_pendigits | 6,870 | 16 | 2.27% |
| 29_Pima | 768 | 8 | 34.9% |
| 2_annthyroid | 7,200 | 6 | 7.42% |
| 30_satellite | 6,435 | 36 | 31.64% |
| 31_satimage-2 | 5,803 | 36 | 1.22% |
| 32_shuttle | 49,097 | 9 | 7.15% |
| 33_skin | 245,057 | 3 | 20.75% |
| 34_smtp | 95,156 | 3 | 0.03% |
| 35_SpamBase | 4,207 | 57 | 39.91% |
| 36_speech | 3,686 | 400 | 1.65% |
| 37_Stamps | 340 | 9 | 9.12% |
| 38_thyroid | 3,772 | 6 | 2.47% |
| 39_vertebral | 240 | 6 | 12.5% |
| 3_backdoor | 95,329 | 196 | 2.44% |
| 40_vowels | 1,456 | 12 | 3.43% |
| 41_Waveform | 3,443 | 21 | 2.9% |
| 42_WBC | 223 | 9 | 4.48% |
| 43_WDBC | 367 | 30 | 2.72% |
| 44_Wilt | 4,819 | 5 | 5.33% |
| 45_wine | 129 | 13 | 7.75% |
| 46_WPBC | 198 | 33 | 23.74% |
| 47_yeast | 1,484 | 8 | 34.16% |
| 48_chess | 28,056 | 6 | 0.1% |
| 49_kddcup99_prob | 64,759 | 6 | 6.43% |
| 4_breastw | 683 | 9 | 34.99% |
| 50_bank | 41,188 | 10 | 11.27% |
| 51_kddcup99_u2r | 60,821 | 6 | 0.37% |
| 5_campaign | 41,188 | 62 | 11.27% |
| 6_cardio | 1,831 | 21 | 9.61% |
| 7_Cardiotocography | 2,114 | 21 | 22.04% |
| 8_celeba | 202,599 | 39 | 2.24% |
| 9_census | 299,285 | 500 | 6.2% |
