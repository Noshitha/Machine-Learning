#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=3, example_subsample_rate=0.5, attr_subsample_rate=0.5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.trees = []
        self.subsampled_attributes = []

    def gini(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        gini_value = 1 - np.sum(probabilities**2)
        return gini_value

    def gini_index(self, y, x):
        parent_gini = self.gini(y)
        gini_a = 0
        for value in set(x):
            partition_indices = x[x == value].index
            partition_gini = self.gini(y[partition_indices])
            gini_a += len(partition_indices) / len(x) * partition_gini
        return gini_a

    def decision_tree(self, X_train, y_train, current_depth=0):
        if len(set(y_train)) == 1 or current_depth == self.max_depth or len(X_train.columns) == 0:
            class_counts = Counter(y_train)
            majority_class = class_counts.most_common(1)[0][0]
            return {"class_label": majority_class}
        
        gini_indices = {attr: self.gini_index(y_train, X_train[attr]) for attr in X_train.columns}
        best_attr = min(gini_indices, key=gini_indices.get)
        node = {"attribute": best_attr, "leaf": {}}
        unique_values = X_train[best_attr].unique()
        for value in unique_values:
            partition_indices = X_train[X_train[best_attr] == value].index
            node["leaf"][value] = self.decision_tree(X_train.loc[partition_indices], y_train.loc[partition_indices], current_depth + 1)
        return node

    def fit(self, X_train, y_train):
        for i in range(self.n_trees):
            bootstrapped_X, bootstrapped_y = self.bootstrap_sampling(X_train, y_train)
            subsampled_attr_indexes = np.random.choice(range(X_train.shape[1]), int(X_train.shape[1] * self.attr_subsample_rate), replace=False)
            self.subsampled_attributes.append(subsampled_attr_indexes.tolist())
            subsampled_X = bootstrapped_X.iloc[:, subsampled_attr_indexes]
            tree = self.decision_tree(subsampled_X, bootstrapped_y)
            self.trees.append(tree)

    def bootstrap_sampling(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        return X.iloc[indices], y.iloc[indices]

    def classify_random_forest(self, X_test):
        class_labels = []
        for _, test_row in X_test.iterrows():
            tree_votes = []
            for tree, sub_attributes in zip(self.trees, self.subsampled_attributes):
                test_features = pd.DataFrame(test_row[sub_attributes]).T
                predicted_label = self.classify(tree, test_features)
                tree_votes.append(predicted_label[0])  # Append predicted label
            class_labels.append(max(set(tree_votes), key=tree_votes.count))  # Perform majority voting
        return class_labels

    def classify(self, tree, features):
        class_labels = []
        for _, feature in features.iterrows():
            node = tree
            while "class_label" not in node:
                split_attr = node["attribute"]
                feature_value = feature[split_attr]
                if feature_value in node["leaf"]:
                    node = node["leaf"][feature_value]
                else:
                    class_labels.append(max(node["leaf"].items(), key=lambda x: len(x[1]))[0])
                    break
            else:
                class_labels.append(node["class_label"])
        return class_labels

    def fit_random_forest(self, X_train, y_train):
        self.fit(X_train, y_train)

    def confusion_matrix(self, y_true, y_pred):
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for i, true_label in enumerate(classes):
            for j, pred_label in enumerate(classes):
                conf_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

        return conf_matrix

    def calculate_metrics(self, conf_matrix):
        TP = np.diag(conf_matrix)
        FP = np.sum(conf_matrix, axis=0) - TP
        FN = np.sum(conf_matrix, axis=1) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        accuracy = np.sum(TP) / np.sum(conf_matrix)
        
        precision = np.where(TP + FP == 0, 0, TP / (TP + FP))
        recall = np.where(TP + FN == 0, 0, TP / (TP + FN))
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return accuracy, precision, recall, f1_score

    def stratified_cross_validation(self, X, y, n_folds=10):
        fold_size = len(X) // n_folds
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size

            X_train_fold = pd.concat([X[:start], X[end:]])
            y_train_fold = pd.concat([y[:start], y[end:]])

            X_validation_fold = X[start:end]
            y_validation_fold = y[start:end]
            
            self.fit_random_forest(X_train_fold, y_train_fold)
            predictions = self.classify_random_forest(X_validation_fold)
            
            # Convert y_validation_fold to list
            y_validation_fold_list = y_validation_fold.tolist()
            
            # Calculate metrics
            conf_matrix = self.confusion_matrix(y_validation_fold_list, predictions)
            acc, prec, rec, f1 = self.calculate_metrics(conf_matrix)
            
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)

        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean([np.mean(precision, axis=0) for precision in precisions], axis=0)
        mean_recall = np.mean([np.mean(recall, axis=0) for recall in recalls], axis=0)
        mean_f1_score = np.nanmean([np.nanmean(f1_score, axis=0) for f1_score in f1_scores], axis=0)

        return mean_accuracy, mean_precision, mean_recall, mean_f1_score


# In[2]:


if __name__ == "__main__":
    df_wine = pd.read_csv("/Users/noshitha/Downloads/hw3/datasets/hw3_wine.csv", delimiter="\t")

    # Shuffle the dataset
    df_wine_shuffle = shuffle(df_wine)

    # Split the dataset into features and target variable
    X = df_wine_shuffle.iloc[:, 1:]  # Assuming the first column is the target variable
    y = df_wine_shuffle.iloc[:, 0] 
    
    n_trees_list = [1, 5, 10, 20, 30, 40, 50]
    n_folds = 10
    max_depth = 3
    example_subsample_rate = 0.5
    attr_subsample_rate = 0.5
    
    accuracy  = []
    precision = []
    recall    = []
    f1_score  = []
    
    rf_classifier = RandomForestClassifier(max_depth=max_depth, example_subsample_rate=example_subsample_rate, attr_subsample_rate=attr_subsample_rate)
    
    for num_trees in n_trees_list:
        print("num_trees: ",num_trees)
        accuracies, precisions, recalls, f1_scores = rf_classifier.stratified_cross_validation(X, y, n_folds=n_folds)
        print("Accuracies:", accuracies)
        print("Precisions:", precisions)
        print("Recalls:", recalls)
        print("F1-scores:", f1_scores)
        accuracy.append(accuracies)
        precision.append(precisions)
        recall.append(recalls)
        f1_score.append(f1_scores)


# ## Accuracy

# In[7]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, accuracy, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees - Wine Dataset')
plt.grid(True)
plt.show()


# ## Precision

# In[8]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, precision, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Precision')
plt.title('Precision vs Number of Trees - Wine Dataset')
plt.grid(True)
plt.show()


# ## Recall

# In[9]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, precision, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Recall')
plt.title('Recall vs Number of Trees - Wine Dataset')
plt.grid(True)
plt.show()


# ## F1 score

# In[10]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, precision, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('F1_score')
plt.title('F1_score vs Number of Trees - Wine Dataset')
plt.grid(True)
plt.show()


# In[2]:


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

class DecisionTree:
    def __init__(self):
        pass

    def gini(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        gini_value = 1 - np.sum(probabilities ** 2)
        return gini_value

    def gini_index(self, y, x):
        parent_gini = self.gini(y)
        gini_a = 0
        for value in set(x):
            partition_indices = x[x == value].index
            partition_gini = self.gini(y[partition_indices])
            gini_a += len(partition_indices) / len(x) * partition_gini
        return gini_a

    def fit(self, X_train, y_train, max_depth):
        self.tree = self._decision_tree(X_train, y_train, max_depth)

    def _decision_tree(self, X_train, y_train, max_depth, current_depth=0):
        if len(set(y_train)) == 1 or current_depth == max_depth or len(X_train.columns) == 0:
            class_counts = Counter(y_train)
            majority_class = class_counts.most_common(1)[0][0]
            return {"class_label": majority_class}

        gini_indices = {attr: self.gini_index(y_train, X_train[attr]) for attr in X_train.columns}
        best_attr = min(gini_indices, key=gini_indices.get)
        node = {"attribute": best_attr, "leaf": {}}
        unique_values = X_train[best_attr].unique()
        for value in unique_values:
            partition_indices = X_train[X_train[best_attr] == value].index
            node["leaf"][value] = self._decision_tree(X_train.loc[partition_indices], y_train.loc[partition_indices], max_depth, current_depth + 1)
        return node

    def predict(self, X_test):
        predictions = []
        for _, test_row in X_test.iterrows():
            predictions.append(self._classify(self.tree, test_row))
        return predictions

    def _classify(self, node, features):
        while "class_label" not in node:
            split_attr = node["attribute"]
            feature_value = features[split_attr]
            if feature_value in node["leaf"]:
                node = node["leaf"][feature_value]
            else:
                return max(node["leaf"].items(), key=lambda x: len(x[1]))[0]
        return node["class_label"]


class RandomForestClassifier:
    def __init__(self, num_trees=10, max_depth=3, example_subsample_rate=0.5, attr_subsample_rate=0.5):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.trees = []

    def fit(self, X_train, y_train):
        for _ in range(self.num_trees):
            bootstrapped_X, bootstrapped_y = self._bootstrap_sampling(X_train, y_train)
            subsampled_attr_indexes = np.random.choice(range(X_train.shape[1]), int(X_train.shape[1] * self.attr_subsample_rate), replace=False)
            subsampled_X = bootstrapped_X.iloc[:, subsampled_attr_indexes]
            tree = DecisionTree()
            tree.fit(subsampled_X, bootstrapped_y, self.max_depth)
            self.trees.append(tree)

    def _bootstrap_sampling(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        return X.iloc[indices], y.iloc[indices]

    def predict(self, X_test):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X_test))
        predictions = np.array(predictions).T
        final_predictions = []
        for pred_row in predictions:
            final_predictions.append(Counter(pred_row).most_common(1)[0][0])
        return final_predictions


def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            conf_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return conf_matrix


def calculate_metrics(conf_matrix):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    accuracy = np.sum(TP) / np.sum(conf_matrix)
    precision = np.where(TP + FP == 0, 0, TP / (TP + FP))
    recall = np.where(TP + FN == 0, 0, TP / (TP + FN))
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


def stratified_cross_validation(X, y, n_folds, num_trees, max_depth, example_subsample_rate, attr_subsample_rate):
    fold_size = len(X) // n_folds
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    conf_matrices = []

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size

        X_train_fold = pd.concat([X[:start], X[end:]])
        y_train_fold = pd.concat([y[:start], y[end:]])

        X_validation_fold = X[start:end]
        y_validation_fold = y[start:end]

        rf_classifier = RandomForestClassifier(num_trees=num_trees, max_depth=max_depth, example_subsample_rate=example_subsample_rate, attr_subsample_rate=attr_subsample_rate)
        rf_classifier.fit(X_train_fold, y_train_fold)
        predictions = rf_classifier.predict(X_validation_fold)

        # Convert y_validation_fold to list
        y_validation_fold_list = y_validation_fold.tolist()

        # Calculate metrics
        conf_matrix = confusion_matrix(y_validation_fold_list, predictions)
        acc, prec, rec, f1 = calculate_metrics(conf_matrix)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        conf_matrices.append(conf_matrix)

    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean([np.mean(precision, axis=0) for precision in precisions], axis=0)
    mean_recall = np.mean([np.mean(recall, axis=0) for recall in recalls], axis=0)
    mean_f1_score = np.nanmean([np.nanmean(f1_score, axis=0) for f1_score in f1_scores], axis=0)

    return mean_accuracy, mean_precision, mean_recall, mean_f1_score


if __name__ == "__main__":
    # Read the dataset
    df_voting = pd.read_csv("/Users/noshitha/Downloads/hw3/datasets/hw3_house_votes_84.csv")

    # Shuffle the dataset
    df_voting_shuffle = shuffle(df_voting)

    # Split the dataset into features and target variable
    X = df_voting_shuffle.iloc[:, :-1]
    y = df_voting_shuffle.iloc[:, -1]

    n_trees_list = [1, 5, 10, 20, 30, 40, 50]
    n_folds = 10
    max_depth = 3
    example_subsample_rate = 0.5
    attr_subsample_rate = 0.5

    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for num_trees in n_trees_list:
        print("num_trees: ", num_trees)
        accuracies, precisions, recalls, f1_scores = stratified_cross_validation(X, y, n_folds, num_trees, max_depth,
                                                                                  example_subsample_rate,
                                                                                  attr_subsample_rate)
        print("Accuracies:", accuracies)
        print("Precisions:", precisions)
        print("Recalls:", recalls)
        print("F1-scores:", f1_scores)
        accuracy.append(accuracies)
        precision.append(precisions)
        recall.append(recalls)
        f1_score.append(f1_scores)


# ### Accuracy

# In[7]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, accuracy, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees - Votes Dataset - Gini')
plt.grid(True)
plt.show()


# ### Precision

# In[8]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, precision, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Precision')
plt.title('Precision vs Number of Trees - Votes Dataset - Gini')
plt.grid(True)
plt.show()


# ### Recall

# In[9]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, precision, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Recall')
plt.title('Recall vs Number of Trees - Votes Dataset - Gini')
plt.grid(True)
plt.show()


# ### F1_Score

# In[10]:


import matplotlib.pyplot as plt

plt.plot(n_trees_list, precision, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('F1_score')
plt.title('F1_score vs Number of Trees - Votes Dataset - Gini')
plt.grid(True)
plt.show()


# In[ ]:




