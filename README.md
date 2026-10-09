# SHL Hiring Assessment 2026 — Grammar Score Prediction

## 1. Project Overview

This project was developed for the SHL Hiring Assessment 2026 Kaggle challenge. The objective was to predict a grammar score between 0 and 5 from spoken audio recordings.

Since the dataset contained only 769 labeled training samples, I focused on extracting useful information from both audio and transcripts rather than training a large deep learning model from scratch.

I experimented with multiple regression algorithms and feature representations to find a model that performed well on unseen data.

## 2. Dataset

| Property | Details |
|---|---|
| Training samples | 769 |
| Test samples | 216 |
| Target variable | Grammar score (0–5) |
| Problem type | Regression |
| Audio format | WAV |

The target labels were distributed across scores from 0 to 5, including half-point values.

## 3. Project Workflow

The overall workflow consisted of audio transcription, feature extraction, text representation, model training, cross-validation, and test prediction generation.

### Step 1: Audio Transcription

I used OpenAI Whisper to convert the audio recordings into text transcripts.

The transcripts provided linguistic information that could be used to estimate grammar quality.

### Step 2: Text Feature Engineering

I extracted six numerical features from the transcripts:

- Word count
- Character count
- Average word length
- Unique word ratio
- Sentence count
- Average sentence length

These features helped represent basic properties of each spoken response.

### Step 3: Audio Feature Extraction

I used Librosa to extract three audio features:

- **Duration:** length of the recording.
- **RMS energy:** a measure of the signal's energy.
- **Zero-crossing rate (ZCR):** how frequently the audio waveform crosses zero.

These features provided additional information about the recordings beyond the transcripts.

## 4. Model Experiments

I tested different approaches incrementally and compared their cross-validation performance using five folds with the same random seed.

### Experiment 1: Ridge Regression with Word TF-IDF

I started with Ridge Regression and word-level TF-IDF features.

TF-IDF represents words and word sequences numerically based on their importance in the transcripts. Ridge Regression provides a regularized linear baseline.

- Model: Ridge Regression
- Features: Word TF-IDF
- Regularization parameter: alpha = 10
- Initial single-split RMSE: 1.2818

I then added six numerical text features.

- Word TF-IDF + text features
- Initial single-split RMSE: 1.0556

Adding basic transcript statistics improved the initial result.

### Experiment 2: Adding Audio Features

Next, I added duration, RMS energy, and zero-crossing rate.

This allowed the model to use both transcript information and basic acoustic properties.

- Features: Word TF-IDF + six text features + three audio features
- Initial single-split RMSE: 0.8785
- Five-fold CV RMSE with Ridge (alpha = 10): 0.8902

The audio features helped in the initial experiment, although the regularization parameter still needed tuning.

### Experiment 3: Ridge Hyperparameter Tuning

I tested different alpha values to find a better balance between fitting the data and regularization.

| Alpha | Mean CV RMSE |
|---:|---:|
| 0.1 | 0.8236 |
| 1 | 0.8183 |
| 5 | 0.8665 |
| 10 | 0.8902 |
| 20 | 0.9083 |
| 50 | 0.9241 |
| 100 | 0.9347 |

Alpha = 1 performed better than the original alpha = 10 configuration.

I also experimented with different word n-gram ranges. Including unigrams, bigrams, and trigrams improved the result slightly, reaching approximately 0.8165 CV RMSE with Ridge Regression.

### Experiment 4: Switching from Ridge to SVR

I then tried Support Vector Regression (SVR) with an RBF kernel to capture nonlinear relationships between the features and grammar scores.

The initial word-TF-IDF SVR model achieved approximately 0.8059 mean CV RMSE.

I experimented with different values of C and epsilon. Increasing C improved the result, and C = 5 produced approximately 0.7689 mean CV RMSE with word TF-IDF.

The selected model configuration was:

- Kernel: RBF
- C: 5.0
- Epsilon: 0.1
- Gamma: scale

### Experiment 5: Adding Character-Level TF-IDF

Word-level TF-IDF captures word and phrase patterns, but character-level TF-IDF can capture smaller spelling and morphological patterns.

I tested three representations with the same SVR setup.

| Feature representation | Mean CV RMSE |
|---|---:|
| Word TF-IDF | 0.7689 |
| Character TF-IDF | 0.7595 |
| Word + character TF-IDF | 0.7495 |

Combining word and character features improved the model further.

I then tuned the character-feature weight. A weight of 3.0 performed best in those experiments.

- Word TF-IDF + 3.0 × character TF-IDF
- Six text features + three audio features
- Mean CV RMSE: approximately 0.7377

This became my strongest TF-IDF-based configuration.

### Experiment 6: Testing BGE Sentence Embeddings

I next experimented with the pretrained `BAAI/bge-small-en-v1.5` sentence embedding model.

BGE converts each transcript into a 384-dimensional vector designed to represent semantic information.

First, I evaluated BGE embeddings with the nine numerical text and audio features.

- Model: SVR with an RBF kernel
- Mean CV RMSE: 0.8012

This was weaker than the existing TF-IDF-based model, so I did not replace TF-IDF with embeddings alone.

### Experiment 7: Combining BGE with TF-IDF

Instead of using BGE as a replacement, I combined it with the existing word and character TF-IDF features and the nine numerical features.

The model used:

- Word TF-IDF
- Character TF-IDF, weighted by 3.0
- BGE sentence embeddings
- Six text features
- Three audio features
- SVR with an RBF kernel

The vectorizers were fitted separately within each training fold for the cross-validation comparison.

| Model | Mean CV RMSE |
|---|---:|
| TF-IDF + numerical features | 0.7377 |
| TF-IDF + BGE + numerical features | 0.7335 |

The combined model improved mean CV RMSE by approximately 0.0042. The improvement was small but consistent across all five folds.

## 5. Best Experimental Configuration

My best cross-validation configuration combined word-level TF-IDF, character-level TF-IDF, BGE embeddings, and numerical text and audio features.

| Component | Configuration |
|---|---|
| Word representation | TF-IDF, n-grams (1, 3) |
| Character representation | TF-IDF, n-grams (3, 5) |
| Character feature weight | 3.0 |
| Sentence embeddings | BAAI/bge-small-en-v1.5 |
| Numerical features | Six text + three audio features |
| Regressor | SVR |
| Kernel | RBF |
| C | 5.0 |
| Epsilon | 0.1 |
| Gamma | scale |
| Mean five-fold CV RMSE | Approximately 0.7335 |

Predictions were clipped to the valid score range of 0 to 5 before creating the submission file.

## 6. Evaluation and Limitations

I used five-fold cross-validation to compare feature combinations and model configurations.

The main limitations were:

- The training dataset was relatively small.
- Whisper transcription errors could affect text-based features.
- Duration, RMS energy, and ZCR provide only a limited representation of speech quality and fluency.
- Repeated experiments on the same cross-validation folds can lead to optimistic model selection.
- Cross-validation RMSE and the competition leaderboard score may not be directly comparable if the competition uses a different evaluation metric.

The reported RMSE values describe validation experiments; they should not be interpreted as the final test-set error.

## 7. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Librosa
- OpenAI Whisper
- Sentence Transformers
- SciPy
- Kaggle Notebooks

## 8. Key Learnings

This project helped me understand that model performance depends on both the learning algorithm and the quality of the feature representation.

I learned how to:

- Convert audio into text using a pretrained speech recognition model.
- Extract numerical features from text and audio.
- Use word and character TF-IDF representations.
- Generate semantic embeddings with a pretrained sentence transformer.
- Compare Ridge Regression and SVR.
- Tune model hyperparameters and evaluate models using cross-validation.
- Combine sparse text features with dense embeddings and numerical features.
- Interpret small performance improvements without assuming they will automatically improve the competition leaderboard.

## 9. Future Improvements

Possible next steps include testing additional regression algorithms, extracting more meaningful speech fluency and pause features, and validating improvements against the official competition metric.

## Acknowledgement

This project was developed as part of the SHL Hiring Assessment 2026 Kaggle challenge.
