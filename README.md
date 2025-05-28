# FastText Word Embedding Assignment: Yelp Dataset Analysis

This project focuses on exploring FastText word embeddings by training, evaluating, and comparing different models using text data from the Yelp academic dataset. The core tasks involve preprocessing text data, training a custom FastText model, utilizing a pre-trained model, fine-tuning the pre-trained model with custom data, and comparing the word similarity and dissimilarity results across these different model variations.

## Project Overview

The primary objective is to gain practical experience with the FastText algorithm for generating word embeddings. This involves leveraging the `gensim` library in Python. The project follows a structured approach, starting from data acquisition and preparation, moving through model training and evaluation, and culminating in a comparative analysis of the different FastText models generated or used.

## Dataset

The project utilizes the Yelp academic dataset, specifically focusing on the `yelp_academic_dataset_tip.json` file. Only the textual content found in the 'text' column of this dataset is used for the analysis. The dataset is sourced from Kaggle. For manageability, the notebook processes a random sample of 5000 entries from the 'text' column after removing any missing values.

## Text Preprocessing

Before training the FastText models, the raw text data undergoes several preprocessing steps to ensure quality and consistency. These steps are implemented using the Natural Language Toolkit (`nltk`) library and include:

1.  **Tokenization:** Breaking down the text into individual words or tokens using `nltk.tokenize.word_tokenize`.
2.  **Lowercasing:** Converting all tokens to lowercase to ensure uniformity.
3.  **Stopword Removal:** Eliminating common English words (like 'the', 'is', 'in') that typically do not carry significant meaning, using `nltk.corpus.stopwords`.
4.  **Punctuation Removal:** Removing punctuation marks to focus on the semantic content of the words.
5.  **Lemmatization:** Reducing words to their base or dictionary form (lemma) using `nltk.stem.WordNetLemmatizer` to group together different inflected forms of a word.

## Methodology

The assignment is structured into three main steps, each involving a different approach to using FastText:

1.  **Step 1: Custom Model Training:** A FastText model is trained from scratch using the `gensim` library on the preprocessed text data extracted from the Yelp dataset. This custom-trained model is then saved for later use and evaluation.
2.  **Step 2: Pre-trained Model Evaluation:** A widely-used pre-trained FastText model (`cc.en.300.bin.gz`, trained on Common Crawl data) is downloaded. This model, already containing embeddings for a vast vocabulary, is tested directly.
3.  **Step 3: Fine-tuning Pre-trained Model:** The pre-trained model from Step 2 is further trained (fine-tuned) using the project's specific Yelp dataset. This step aims to adapt the general-purpose embeddings to the specific domain or style of the Yelp tips text.

## Evaluation

Each of the three models (custom-trained, pre-trained, and fine-tuned pre-trained) is evaluated by testing its ability to identify semantic relationships between words. For a given input word, each model is queried to find the 10 most similar words and the 10 most dissimilar (opposite) words based on the learned embeddings.

## Requirements and Setup

This project requires Python 3 and several libraries. Specific versions of `numpy`, `gensim`, and `scipy` are needed for compatibility, as indicated in the notebook's setup cells:

- `pandas`: For data manipulation and reading the JSON dataset.
- `numpy==1.23.5`: For numerical operations.
- `gensim==4.3.2`: The core library for training and using FastText models.
- `scipy==1.10.1`: Scientific computing library, often a dependency for `gensim`.
- `nltk`: For text preprocessing tasks.
- `torch`: Included in imports, potentially for underlying computations or future extensions, though not directly used in the core FastText steps shown.
- `kagglehub`: For interacting with Kaggle datasets.
- `zipfile`: For extracting the dataset.

**Setup Steps:**

1.  **Install Libraries:** Execute the `pip install` commands provided in the notebook to install the required libraries, including the specific versions for `numpy`, `gensim`, and `scipy`. Note that installing these specific versions might require uninstalling existing versions first, and the notebook includes commands for this.
2.  **Download NLTK Data:** Run the `nltk.download()` commands in the notebook to obtain necessary resources like 'punkt' (tokenizer models), 'stopwords', and 'wordnet' (for lemmatization).
3.  **Kaggle API Setup:** To download the Yelp dataset, you need a Kaggle API token. Place your `kaggle.json` file in the appropriate directory (`~/.kaggle/`) and set the correct permissions (`chmod 600 ~/.kaggle/kaggle.json`), as shown in the notebook.
4.  **Download Dataset:** Use the `kaggle datasets download` command to fetch the Yelp dataset zip file.
5.  **Extract Data:** Unzip the downloaded file to extract `yelp_academic_dataset_tip.json`.

## Usage

The project is implemented within the provided Jupyter Notebook (`FastText_Assignment (1).ipynb`). To run the analysis, execute the notebook cells sequentially. This will perform the setup, data loading, preprocessing, model training (Step 1), pre-trained model loading (Step 2), model fine-tuning (Step 3), and evaluations.

## Output and Comparison

The primary output consists of the lists of similar and dissimilar words generated by each of the three models for the same input words. The assignment requires comparing these outputs to understand the differences in embeddings learned by the custom model, the general pre-trained model, and the fine-tuned model. These comparisons and conclusions are expected to be documented, potentially in a separate PDF report as mentioned in the notebook's description.

## Authors

This assignment was completed by:

- Ro'aa Fathi (20210140)
- Selsabeel Asim (20210714)
- Abdelrahman Mohamed (20210518)
- Nourhan Hossam (20221185)

*(Task distribution details are mentioned within the notebook.)*


