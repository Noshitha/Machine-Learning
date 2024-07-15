#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load 50% of the training set and 100% of the test set
pos_train_50, neg_train_50, vocab_train_50 = load_training_set(0.5, 0.5)

# Initialize and train the model with the optimal α from the previous experiment
model_q4 = MultinomialNaiveBayes(alpha=1)
model_q4.train(pos_train_50, neg_train_50, vocab_train_50)

# Evaluate the model
accuracy_q4, precision_q4, recall_q4, confusion_matrix_q4 = model_q4.test(pos_test, neg_test, use_log=True)

# Report results
print("Q.4 Results:")
print(f"Accuracy: {accuracy_q4:.3f}")
print(f"Precision: {precision_q4:.3f}")
print(f"Recall: {recall_q4:.3f}")
print("Confusion Matrix:")
print(confusion_matrix_q4)

# Compare with Q.3
print("\nComparison with Q.3:")
print(f"Accuracy Change: {accuracy_q4 - accuracy_q3:.3f}")
print(f"Precision Change: {precision_q4 - precision_q3:.3f}")
print(f"Recall Change: {recall_q4 - recall_q3:.3f}")

