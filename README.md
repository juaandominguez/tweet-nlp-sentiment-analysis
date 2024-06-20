# Tweet Sentiment analysis with Python using NLP

This repository contains code for performing sentiment analysis on tweets using Natural Language Processing (NLP) techniques in Python. The project uses the Sentiment140 dataset from Kaggle to train a logistic regression model to classify the sentiment of tweets as either positive or negative.

## Dataset

The dataset used is the Sentiment140 dataset, which contains 1.6 million tweets with sentiment labels. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Installation

To get started, clone this repository

```sh
git clone https://github.com/juaandominguez/tweet-nlp-sentiment-analysis
```

## Usage

Simply run the Jupyter Notebook _(the stemming process may take some time)_

## Concepts

### Natural Language Processing (NLP)

NLP is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to read, decipher, understand, and make sense of human language in a valuable way. NLP is used to apply algorithms to identify and extract the natural language rules such that the unstructured language data is converted into a form that computers can understand.

### Stemming

Stemming is the process of reducing a word to its word stem, base, or root form. For instance, the words "running", "runner", and "ran" are reduced to "run". This is useful in text processing because it reduces the number of different forms of a word, which simplifies the analysis and comparison of text data.

### TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). The TF-IDF value increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general. This technique is used to transform text data into a format that can be used for machine learning models.

### Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. In the context of sentiment analysis, logistic regression can be used to classify the sentiment of a tweet as either positive or negative based on the features extracted from the text.

### Model Evaluation

Model evaluation involves assessing the performance of a machine learning model. Common metrics for classification tasks include accuracy, precision, recall, and F1-score. In this project, we use accuracy to measure the proportion of correctly classified tweets in the training and test datasets.

## Results

The model's performance is evaluated using accuracy metrics on both the training and test datasets. The training accuracy and test accuracy will be printed during the execution.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
