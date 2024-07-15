#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load 10% of positive and 50% of negative training instances
pos_train_unbalanced, neg_train_unbalanced, vocab_train_unbalanced = load_training_set(0.1, 0.5)

# Initialize and train the model with the optimal α from the previous experiment
model_q6 = MultinomialNaiveBayes(alpha=1)
model_q6.train(pos_train_unbalanced, neg_train_unbalanced, vocab_train_unbalanced)

# Evaluate the model
accuracy_q6, precision_q6, recall_q6, confusion_matrix_q6 = model_q6.test(pos_test, neg_test, use_log=True)

# Report results
print("Q.6 Results:")
print(f"Accuracy: {accuracy_q6:.3f}")
print(f"Precision: {precision_q6:.3f}")
print(f"Recall: {recall_q6:.3f}")
print("Confusion Matrix:")
print(confusion_matrix_q6)

# Compare with Q.4
print("\nComparison with Q.4:")
print(f"Accuracy Change: {accuracy_q6 - accuracy_q4:.3f}")
print(f"Precision Change: {precision_q6 - precision_q4:.3f}")
print(f"Recall Change: {recall_q6 - recall_q4:.3f}")

