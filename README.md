# Sentiment Analysis of Tweets with TF-IDF and Machine Learning

**Course:** CSE570 – Machine Learning with Python (Lovely Professional University) · **Task:** Skill Based Assignment

An end-to-end ML project that classifies tweets as **positive** or **negative**. It covers data cleaning, EDA with statistical tests, four tuned models compared against three baselines (including the pretrained VADER lexicon), and a saved model you can query from the command line.

## Problem and objectives
- **Task:** binary text classification (target: positive = 1, negative = 0).
- **Success criteria (set before training):** F1 ≥ 0.70 and ROC-AUC ≥ 0.75 on a held-out test set; beat the majority-class and VADER baselines; leakage-free pipeline; interpretable word-level insights.

## Dataset
NLTK Twitter Samples (`twitter_samples`) – 5,000 positive + 5,000 negative tweets. Source: Bird, Klein & Loper (2009), *Natural Language Processing with Python*; data repository https://www.nltk.org/nltk_data/. Downloaded automatically by the notebook.
Labels come from emoticons, so emoticons are **removed from the model input** (otherwise the task is trivial). After removing empty, conflicting and near-duplicate tweets: 9,160 tweets.

## Results (held-out test set, 1,832 tweets)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority class (baseline) | 0.518 | 0.000 | 0.000 | 0.000 | 0.500 |
| Metadata-only LR (baseline) | 0.643 | 0.663 | 0.526 | 0.586 | 0.690 |
| VADER (pretrained) | 0.651 | 0.656 | 0.582 | 0.617 | 0.719 |
| Random Forest | 0.722 | 0.732 | 0.668 | 0.699 | 0.796 |
| Multinomial NB | 0.761 | 0.792 | 0.683 | 0.734 | 0.845 |
| Linear SVM | 0.764 | 0.780 | 0.711 | 0.744 | 0.845 |
| **Logistic Regression (final)** | **0.763** | **0.778** | **0.712** | **0.744** | **0.845** |

The final model was selected by 5-fold cross-validation on the training data only. The three linear/NB models are statistically indistinguishable on the test set (McNemar p > 0.7).

## How to run
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb               # Run All (about 3 minutes; Random Forest tuning is the slow part)
python predict.py "I love this phone, battery life is amazing!"
```

## Repository structure
```
├── sentiment_analysis.ipynb   # full workflow with outputs
├── predict.py                 # command-line predictions
├── requirements.txt
├── models/sentiment_model.joblib
├── figures/                   # all charts used in the report
├── results/                   # test_metrics.csv, cv_results.csv, summary.json
└── report/Sentiment_Analysis_Report.docx
```

## Limitations
Noisy emoticon-derived labels, English only, two classes, bag-of-words ignores context/sarcasm, and some dataset topics (e.g. specific hashtags) leak into the vocabulary. Future work: human-labelled product reviews, transformer fine-tuning (DistilBERT/RoBERTa), a neutral class.

## References
Bird, Klein & Loper (2009) NLTK · Hutto & Gilbert (2014) VADER · Pedregosa et al. (2011) scikit-learn · Go, Bhayani & Huang (2009) Twitter sentiment classification using distant supervision.
