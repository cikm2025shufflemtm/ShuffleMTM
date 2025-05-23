# ShuffleMTM

### This repository provides the official implementation of ShuffleMTM: Learning Cross-Channel Dependence in Channel-Independent Time Series Masked Pre-training.


# Requirements

- Python 3.9.0
- torch==2.0.1
- numpy==1.24.3
- pandas==1.5.3
- scikit-learn==1.2.2
- matplotlib==3.7.1
- tensorboardX==2.6.2.2

Dependencies can be installed using the following command:

    pip install -r requirements.txt

# Getting Started

## 1. Prepare Data

All benchmark datasets can be obtained from [Google Drive](), and arrange the folder as:

    ShuffleMTM/
    |-- datasets/
        |-- ETTh1.csv
        |-- ETTh2.csv
        |-- ETTm1.csv
        |-- ETTm2.csv
        |-- Weather.csv
        |-- Electricity.csv
        |-- exchange_rate.csv
        |-- Traffic.csv

## 2. Experimental reproduction

- We provide the scripts for pre-training and finetuning for each dataset with the best hyper-parameters in our experiment at `./scripts/`.

### Pre-training and fine-tuning at once

To implement pre-training and fine-tuning sequentially, the scripts in `./scripts/`. For example, to perform both steps at once for the ETTh1 dataset:

    bash scripts/etth1.sh
    
- If the paper is accepted, I will release the scripts for the other datasets along with the best-performing hyperparameters.
- To maintain anonymity for double-blind review, the dataset download link is not provided at this time. It will be shared upon paper acceptance.


# Additional experiments

## 1. Ablation of the shuffling operation

We validate the effectiveness of masked series shuffling on the PEMS08 dataset, which contains 170 channels and is known known to exhibit strong spatial dependencies [1]. Table 1 presents the comparison results on this dataset. ShuffleMTM outperforms its counterpart without the shuffling operation, confirming again the validity of the shuffling mechanism in modeling cross-channel dependencies. This corresponds to the ablation study on the synthetic datasets (Section 5.1.1), which demonstrates the effectiveness of the shuffling mechanism on two datasets exhibiting clear and complex cross-channel dependencies.

| Pred_len | &nbsp;&nbsp;Shuffle | MTM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| ShuffleMTM | w/o Shuffle|
|:----------:|:-------------:|:-------------:|:---------:|:---------:|
||MSE|MAE|MSE|MAE|
| 96       | **0.380**       | **0.418**       | 0.383   | 0.421   |
| 192      | **0.438**       | **0.446**       | 0.442   | 0.449   | 
| 336      | **0.394**      | **0.407**       | 0.397   | 0.409   | 
| 720      | **0.465**       | **0.459**       | 0.468   | 0.461   |

Table 1) Ablations of the shuffling operation on the PEMS08 dataset

[1] BasicTS+, Z, Shao et al., TKDE 2024

## 2. Comparison with BTSF

We investigated self-supervised learning approaches designed to capture cross-channel dependencies and compared with BTSF (ICML 2022) [2], which incorporates the spatial information in feature space through contrastive learning. While BTSF aims to distinguish representations of each channel, ShuffleMTM integrates temporal and channel dependencies for reconstruction. Accordingly, ShuffleMTM fundamentally differs from BTSF in its objective: BTSF learns channel-wise representations, whereas ShuffleMTM learns time series representations that incorporate both cross-time and cross-channel dependencies. Furthermore, ShuffleMTM is the first masked pre-training framework that incorporates cross-time and cross-channel dependencies. As shown in Table 3, ShuffleMTM outperforms BTSF in time series forecasting. For the performance of BTSF, we follow the results in the original paper.

|||ShuffleMTM||BTSF||
|:--------:|:--------:|:-------:|:------:|:-------:|:------:|
| Dataset  | Pred_len |   MSE   |  MAE   |   MSE   |  MAE   |
| ETTh1    |    48    |  **0.334**  | **0.373**  |  0.613  | 0.524  |
|          |   168    |  **0.412**  | **0.419**  |  0.640  | 0.532  |
|          |   336    |  **0.456**  | **0.446**  |  0.864  | 0.689  |
|          |   720    |  **0.474**  | **0.471**  |  0.993  | 0.712  |
| ETTm1    | 48 | **0.275** | **0.326** | 0.395 | 0.387 | 
 | | 168 | **0.353** | **0.375** | 0.438 | 0.399 | 
  |  | 366 | **0.390** | **0.402** | 0.675 | 0.429 | 
   |  | 720| **0.446** | **0.435** | 0.721 | 0.643 | 
| Weather  |    48    |  **0.131**  | **0.168**  |  0.366  | 0.427  |
|          |   168    |  **0.202**  | **0.241**  |  0.543  | 0.477  |
|          |   336    |  **0.271**  | **0.293**  |  0.568  | 0.487  |
|          |   720    |  **0.348**  | **0.343**  |  0.601  | 0.522  |

Table 2) Comparison between ShuffleMTM and BTSF in time series forecasting

[2] BTSF, Yang and Hong, ICML 2022


## 3. Classification evaluation on HAR dataset

We additionally evluate ShuffleMTM on the human activity recognition (HAR) dataset for time series classification benchmark, which is a widely used biosignal dataset. We compared ShuffleMTM with the two best-performing baselines in the classification task, TimeSiam and COMET. As shown in Table 3, ShuffleMTM shows the second-best performance among the best-performing baselines. The superior performance over TimeSiam demonstrates the effectiveness of cross-channel dependency modeling. COMET showed the best performance using additional meta-information, not recorded in the time series. As human activity is highly dependent on each subject’s behavioral patterns, using subject indicator information in COMET was effective for the HAR dataset [3]. Accordingly, excluding meta-information (e.g., subject and trial IDs), as in COMET (w/o meta info), results in a notable decline in classification performance.

| Method | ACC | Prec | Recall | F1|
|:----------:|:-------------:|:-------------:|:---------:|:---------:|
| ShuffleMTM | 0.904 | 0.908 | 0.905 | 0.905 |     
| TimeSiam | 0.843 | 0.854 | 0.840 | 0.843 |     
| COMET  | 0.814 | 0.933 | 0.821 | 0.920 |      
| COMET (w/o meta info)  | 0.871 | 0.913 | 0.881 | 0.776 | 

Table 3) Comparison of ShuffleMTM with TimeSiam and COMET on the HAR dataset. COMET (w/o meta info) refers to COMET trained with the same input as ShuffleMTM, without using additional meta information for contrastive learning.

In total, we constructed three classification benchmarks from representative biomedical domains and demonstrated that ShuffleMTM outperforms masked pre-training baselines and achieves comparable performance to COMET, a state-of-the-art medical self-supervised model. ShuffleMTM achieved seven best and five second-best results across 12 classification scenarios. It is notable that in the second-best cases it follows COMET, which leverages additional meta-information for representation learning.

[3] Multi-sensor fusion based on asymmetric decision weighting for robust activity recognition, O, Banos et al., Sensors 2014


## 4. Classification Embedding Visualization

We visualize the learned embeddings on the AD dataset using UMAP. For comparison, we include TimeSiam, which has shown the best classification performance among all MTM baselines. We also compute the average pairwise Euclidean distance between negative (healthy) and positive (Alzheimer's) classes in the UMAP space. As shown in Figure 1, ShuffleMTM yields more compact intra-class clusters and a larger inter-class distance than TimeSiam, indicating better class separability of the embedding space of ShuffleMTM.


