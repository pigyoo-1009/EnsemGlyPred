# EnsemGlyPred

**EnsemGlyPred: A Multi-Level Framework for Lysine Glycation Site Prediction by Integrating Deep Semantic Features and Sequence Information**

This study proposes a predictive framework for lysine glycation sites based on multi-feature fusion and weighted ensemble learning. By combining traditional sequence-based features and advanced protein language model representations, the framework significantly enhances prediction accuracy and reliability.

## File Description

**code files folder**: Contains core code files for model training and testing.
- **Train folder**: Includes training scripts for three base models. These models are built using AAC features, PAAC features, and ProGen2 features with XGBoost and BiLSTM classifiers, respectively. Ensemble training scripts are also included.
- **Test folder**: Contains model testing and ensemble prediction scripts corresponding to the trained base models.

**dataset folder**: Contains carefully de-redundant processed datasets, including a training set (7,146 samples) and an independent test set (400 samples), both with balanced positive and negative samples.

**feature_npy folder**: Stores the extracted feature vectors, including:
- AAC features (20 dimensions)
- PAAC features (25 dimensions)
- ProGen2 features reduced via PCA (300 dimensions)
- Corresponding label files

**model files folder**: Includes trained base model files.Due to the large size of the pre-trained ProGen2 model, it is not uploaded to GitHub.Please download it from HuggingFace: https://huggingface.co/hugohrban/progen2-base. To ensure reproducibility given the stochastic nature of large model parameters, please set the random seed.The seed used in this study is 58.

## Model Architecture

This study presents a multi-feature fusion-based weighted ensemble classification framework, which includes three base models:

- **AAC-XGBoost model**: An XGBoost classifier using amino acid composition (AAC) features.
- **PAAC-XGBoost model**: An XGBoost classifier using pseudo-amino acid composition (PAAC) features.
- **ProGen2-BiLSTM model**: A BiLSTM classifier utilizing protein language model (ProGen2) features.

The ensemble strategy employs **weighted soft voting** with the following weights:
- AAC-XGBoost: `0.33`
- PAAC-XGBoost: `0.40`
- ProGen2-BiLSTM: `0.27`

## Contact

For any questions or issues, please contact:  
Email: 1191230413@stu.jiangnan.edu.cn
