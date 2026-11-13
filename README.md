# Next Word Prediction using LSTM

A deep learning project that predicts the next word in a sentence using a stacked **LSTM (Long Short-Term Memory)** network built with TensorFlow/Keras. The model is trained on a text corpus from Kaggle and learns the probability of the next word given the words that came before it.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Making Predictions](#making-predictions)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Technologies Used](#technologies-used)

## Overview

Next word prediction is a core Natural Language Processing (NLP) task used in smartphone keyboards, search engines, and email auto-complete. This project:

1. Cleans and preprocesses raw text
2. Converts sentences into n-gram sequences of word indices
3. Trains a two-layer LSTM with a softmax output over the vocabulary
4. Predicts the most likely next word for a given input

## Dataset

- **Source:** [Next Word Prediction dataset on Kaggle](https://www.kaggle.com/datasets/ronikdedhia/next-word-prediction) (`ronikdedhia/next-word-prediction`)
- **Size:** about 558,000 characters of raw text
- The notebook downloads the dataset automatically with `kagglehub`.

## Project Pipeline

| Step | Description |
|------|-------------|
| 1. Load data | Download the dataset with `kagglehub` and read the text file (`charmap` encoding) |
| 2. Clean text | Convert to lowercase and keep only letters `a-z` and whitespace using regex |
| 3. Lemmatize | Reduce words to their base form with NLTK `WordNetLemmatizer` |
| 4. Sentence split | Split the text into sentences using `sent_tokenize` |
| 5. Tokenize | Convert words to integer indices using the Keras `Tokenizer` |
| 6. Build n-grams | For each sentence, create growing sequences (`w1 w2 w3`, `w1 w2 w3 w4`, ...) |
| 7. Split X / y | All words except the last are the input (`X`); the last word is the target (`y`) |
| 8. Pad | Pad every input sequence to a fixed length of 251 tokens |
| 9. Train/test split | 77% training, 23% validation (`random_state=42`) |
| 10. Train | Fit the LSTM model with sparse categorical cross-entropy |

## Model Architecture

```
Input (251, 1)
   │
LSTM (512 units, return_sequences=True)
   │
LSTM (256 units)
   │
Dense (8646 units, softmax)   → probability of every word in the vocabulary
```

- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam
- **Metric:** Accuracy
- **Epochs:** 5

## Results

After 5 epochs, the training accuracy is about 6 to 7% and the validation accuracy is about 6%. The training loss drops from about 7.1 to below 6 during the first epochs.

| Epoch | Train Accuracy | Val Accuracy |
|-------|----------------|--------------|
| 1 | 4.3% | 5.6% |
| 2 | 6.3% | 5.8% |
| 3 | 6.8% | 5.9% |

> These are baseline numbers. Next word prediction over a vocabulary of thousands of words is a hard task, and the model has only been trained for a few epochs. See the improvements section below.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install tensorflow nltk scikit-learn numpy kagglehub
```

You will also need to download the NLTK data (the notebook does this for you):

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
```

## How to Run

**Option 1: Google Colab (recommended)**

1. Open `Next_Word_Prediction_By_LSTM.ipynb` in Google Colab
2. Select **Runtime → Change runtime type → GPU**
3. Run all cells (**Runtime → Run all**)

**Option 2: Locally**

```bash
jupyter notebook Next_Word_Prediction_By_LSTM.ipynb
```

## Making Predictions

After training, you can predict the next word for any input text with a helper like this (it reuses the notebook's `tok` tokenizer and `model`, and the same padding settings used in training):

```python
import numpy as np
from tensorflow.keras.utils import pad_sequences

def predict_next_word(seed_text, top_k=3):
    seed_text = seed_text.lower()
    seq = tok.texts_to_sequences([seed_text])[0]
    seq = pad_sequences([seq], maxlen=251, padding='post')
    probs = model.predict(seq, verbose=0)[0]
    top_idx = probs.argsort()[-top_k:][::-1]
    return [(tok.index_word.get(i, '?'), float(probs[i])) for i in top_idx]

print(predict_next_word("the king was"))
```

## Limitations and Future Improvements

- **Add an `Embedding` layer.** The model currently feeds raw word indices to the LSTM as a single numeric feature. An embedding layer would let it learn meaningful word vectors and should improve accuracy considerably.
- **Train longer.** Five epochs is very short. Use more epochs with `EarlyStopping` to avoid overfitting.
- **Use shorter sequences.** Padding every input to 251 tokens wastes computation. A smaller window (for example 20 to 50 words) is faster and often works better.
- **Use `padding='pre'`.** Pre-padding usually works better for LSTMs, since the most recent words sit next to the output.
- **Add Dropout.** This helps reduce overfitting in the stacked LSTMs.
- **Try pre-trained embeddings** such as GloVe or Word2Vec, or move to a Transformer-based model.
- **Evaluate with perplexity** and top-k accuracy, which suit language models better than plain accuracy.

## Technologies Used

- Python 3
- TensorFlow / Keras
- NLTK
- scikit-learn
- NumPy
- Google Colab
- KaggleHub

## License

This project is for educational purposes. Feel free to use and modify it.
