#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_training_set, load_test_set

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.priors = {}
        self.word_probs = {}
        self.alpha = alpha

    def train(self, pos_train, neg_train, vocab):
        # Calculate priors
        total_docs = len(pos_train) + len(neg_train)
        self.priors['positive'] = len(pos_train) / total_docs
        self.priors['negative'] = len(neg_train) / total_docs

        # Count word occurrences in positive and negative training sets
        word_counts_pos = defaultdict(int)
        word_counts_neg = defaultdict(int)

        for doc in pos_train:
            for word in doc:
                word_counts_pos[word] += 1

        for doc in neg_train:
            for word in doc:
                word_counts_neg[word] += 1

        self.word_probs['positive'] = self.calculate_word_probs(word_counts_pos, vocab)
        self.word_probs['negative'] = self.calculate_word_probs(word_counts_neg, vocab)

    def calculate_word_probs(self, word_counts, vocab):
        word_probs = {}
        total_words = sum(word_counts.values()) + len(vocab)

        for word in vocab:
            if word:
                word_probs[word] = (word_counts[word] + self.alpha) / (total_words + self.alpha * len(vocab))
            else:
                word_probs[word] = 0

        return word_probs

    def test(self, pos_test, neg_test, use_log=False):
        correct_predictions = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for doc_set, true_label in [(pos_test, 'positive'), (neg_test, 'negative')]:
            for doc in doc_set:
                predicted_label = self.classify(doc, use_log)

                # Update confusion matrix
                if predicted_label == true_label:
                    correct_predictions += 1
                    if predicted_label == 'positive':
                        true_positive += 1
                    else:
                        true_negative += 1
                else:
                    if predicted_label == 'positive':
                        false_positive += 1
                    else:
                        false_negative += 1

        accuracy = correct_predictions / (len(pos_test) + len(neg_test))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        confusion_matrix = {
            'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative
        }

        return accuracy, precision, recall, confusion_matrix

    def classify(self, doc, use_log=False):
        # Calculating probabilities or log probabilities based on the flag
        if use_log:
            return self.classify_log_probabilities(doc)
        else:
            return self.classify_probabilities(doc)


    
    
    def classify_probabilities(self, doc):
        prob_pos = self.priors['positive']
        prob_neg = self.priors['negative']

        for word in doc:
            if word in self.word_probs['positive']:
                prob_pos *= self.word_probs['positive'][word]
            else:
                prob_pos *= 0  # Set probability to 0 for unseen words

            if word in self.word_probs['negative']:
                prob_neg *= self.word_probs['negative'][word]
            else:
                prob_neg *= 0  # Set probability to 0 for unseen words

        return 'positive' if prob_pos > prob_neg else 'negative'


    def classify_log_probabilities(self, doc):
        log_prob_pos = np.log(self.priors['positive'])
        log_prob_neg = np.log(self.priors['negative'])

        for word in doc:
            if word in self.word_probs['positive']:
                log_prob_pos += np.log(self.word_probs['positive'][word] + 1e-10)
            else:
                log_prob_pos += np.log(1e-10)

            if word in self.word_probs['negative']:
                log_prob_neg += np.log(self.word_probs['negative'][word] + 1e-10)
            else:
                log_prob_neg += np.log(1e-10)

        return 'positive' if log_prob_pos > log_prob_neg else 'negative'
    
    def plot_results(results):
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(results['alpha'], results['accuracy'], marker='o')
        plt.xscale('log')
        plt.title('Model Accuracy vs. Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(results['alpha'], results['precision'], label='Precision', marker='o')
        plt.plot(results['alpha'], results['recall'], label='Recall', marker='o')
        plt.xscale('log')
        plt.title('Model Precision and Recall vs. Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('Score')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, confusion_matrix):
        labels = ['Positive', 'Negative']
        cm_array = np.array([[confusion_matrix['true_positive'], confusion_matrix['false_positive']],
                             [confusion_matrix['false_negative'], confusion_matrix['true_negative']]])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

