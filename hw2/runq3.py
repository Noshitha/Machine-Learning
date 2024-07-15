#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load 100% of the training and test sets
pos_train, neg_train, vocab_train = load_training_set(1.0, 1.0)
pos_test, neg_test = load_test_set(1.0, 1.0)

# Initialize and train the model with the optimal α =1
model_q3 = MultinomialNaiveBayes(alpha=1)
model_q3.train(pos_train, neg_train, vocab_train)

# Evaluate the model
accuracy_q3, precision_q3, recall_q3, confusion_matrix_q3 = model_q3.test(pos_test, neg_test, use_log=True)

# Report results
print("Q.3 Results:")
print(f"Accuracy: {accuracy_q3:.3f}")
print(f"Precision: {precision_q3:.3f}")
print(f"Recall: {recall_q3:.3f}")
print("Confusion Matrix:")
print(confusion_matrix_q3)

