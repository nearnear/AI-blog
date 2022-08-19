# Natural Language Processing Notebooks
This repository is an archive of NLP notebooks, mainly in the purpose of setting baseline for each tasks.


## Notebooks
### 01. Sentiment Analysis
_Updated 28 Mar, 2022_

- Sentiment binary classification on short sentences with log likelihood-ratio method.
- Task : Sequence Classification
- Dataset : [Google GoEmotion](https://github.com/google-research/google-research/tree/master/goemotions)

- naive_bayes_sentiment
    - Task : Binary Sentiment Classification
    - Model : Naive Bayes
    - Result : 81 % Accuracy

### 02. Score Clinical Patient Notes
_Updated 12 Jun, 2022_

- These notebooks are written for [kaggle competition](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes).
- Goal : Automatically extract feature text from human-written patient interview notes.
- Task : Token Classification ([Segal et al, 2020](https://arxiv.org/pdf/1909.13375v4.pdf))
- Dataset : [NBME kaggle dataset](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/data)

- NBME_hf
    - Uses HuggingFace API
    - Model : DistilBERT
    - Result : In Process
- NBME_pt
    - Uses PyTorch and Weights & Biases for hyperparameter tuning.
    - Model : DistilBERT
    - Result : Binary Cross Entropy 0.0139
