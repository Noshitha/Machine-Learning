#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pos_train, neg_train, vocab = load_training_set(0.2, 0.2)
pos_test, neg_test = load_test_set(0.2, 0.2)

model_without_laplace = MultinomialNaiveBayes(alpha=0)
model_without_laplace.train(pos_train, neg_train, vocab)

accuracy, precision, recall, confusion_matrix = model_without_laplace.test(pos_test, neg_test)
print("\nEvaluation without Laplace Smoothing:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print("Confusion Matrix:")
model_without_laplace.plot_confusion_matrix(confusion_matrix)

#print(plot_confusion_matrix(confusion_matrix))

model_log_without_laplace = MultinomialNaiveBayes(alpha=0)
model_log_without_laplace.train(pos_train, neg_train, vocab)

accuracy_log, precision_log, recall_log, confusion_matrix_log = model_log_without_laplace.test(pos_test, neg_test, use_log=True)
print("\nEvaluation using Log Probabilities without Laplace Smoothing:")
print(f"Accuracy: {accuracy_log:.3f}")
print(f"Precision: {precision_log:.3f}")
print(f"Recall: {recall_log:.3f}")
print("Confusion Matrix:")
model_log_without_laplace.plot_confusion_matrix(confusion_matrix)


print("\n Q2:")
print("\nEvaluation with Laplace Smoothing:")
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 100, 1000]
# Evaluate with α = 1
model_alpha_1 = MultinomialNaiveBayes(alpha=1)
model_alpha_1.train(pos_train, neg_train, vocab)
accuracy_1, _, _, _ = model_alpha_1.test(pos_test, neg_test, use_log=True)

print(f"Evaluation with α = 1:")
print(f"Accuracy: {accuracy_1:.3f}")

#results = evaluate_model_with_alpha(pos_train, neg_train, pos_test, neg_test, vocab, alpha_values)

results = {'alpha': [], 'accuracy': [], 'precision': [], 'recall': []}
for alpha in alpha_values:
    model = MultinomialNaiveBayes(alpha=alpha)
    model.train(pos_train, neg_train, vocab)

    accuracy, precision, recall, _ = model.test(pos_test, neg_test, use_log=True)
    results['alpha'].append(alpha)
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    
plot_results(results)

model_with_laplace = MultinomialNaiveBayes(alpha=1.0)
model_with_laplace.train(pos_train, neg_train, vocab)

accuracy_laplace, precision_laplace, recall_laplace, confusion_matrix_laplace = model_with_laplace.test(pos_test, neg_test,use_log=True)

print(f"Accuracy: {accuracy_laplace:.3f}")
print(f"Precision: {precision_laplace:.3f}")
print(f"Recall: {recall_laplace:.3f}")
print("Confusion Matrix:")
model_with_laplace.plot_confusion_matrix(confusion_matrix)

