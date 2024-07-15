#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd


# In[59]:


df_iris=pd.read_csv("/Users/noshitha/Downloads/iris.csv")


# In[60]:


df_iris.head()


# In[61]:


df_iris.count()


# In[62]:


from sklearn.utils import shuffle
df_iris_shuffled = shuffle(df_iris)

# First 5 rows
print(df_iris_shuffled.head(5))


# In[63]:


from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df_iris_shuffled, test_size=0.2, random_state=34)

# Shapes of the training and testing sets
print("\nTraining Set Shape:", train_df.shape)
print("Testing Set Shape:", test_df.shape)


# In[64]:


target_column = 'Species'  
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]


# In[65]:


X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)


# ## k-NN Algorithm with Normalization
# 

# In[66]:


import numpy as np
from collections import Counter

def features_normalization(X):
    X_num = X.apply(pd.to_numeric) #Typecasting to Numeric
    min_value = X_num.min()
    max_value = X_num.max()
    X_normalized = (X_num - min_value) / (max_value - min_value)
    return X_normalized

def distance(X1, X2):
    Euclidean_distance = np.sqrt(np.sum((X1 - X2) ** 2)) # Euclidean distance calculation
    return Euclidean_distance

def knn_predict(X_train, y_train, X_test, k):
    final_output = []
    for i in range(len(X_test)):
        distances = [(distance(X_train.iloc[j], X_test.iloc[i]), j) for j in range(len(X_train))]
        distances.sort()
        neighbors = distances[:k]
        neighbor_labels = [y_train.iloc[j[1]] for j in neighbors]
        predicted_label = Counter(neighbor_labels).most_common(1)[0][0]
        final_output.append(predicted_label)
    return final_output


# In[67]:


X_train_normalized = features_normalization(X_train)


# #### Q1.1 (10 Points) 
# In the first graph, you should show the value of k on the horizontal axis, and on the vertical axis, the average accuracy of models trained over the training set, given that particular value of k. Also show, for each point in the graph, the corresponding standard deviation; you should do this by adding error bars to each point. 

# In[68]:


std_devs_train = []
std_devs_test = []

accuracy_values_train = []
accuracy_values_test = []

avg_accuracies_train = []
avg_accuracies_test = []

training_errors = []
testing_errors = []

avg_training_errors = []
avg_testing_errors = []
k_values = [k for k in list(range(1, 51, 2))]
iterations = 20
for k in k_values:
    for i in range(iterations):
        df_iris_shuffled = shuffle(df_iris)
        train_df, test_df = train_test_split(df_iris_shuffled, test_size=0.2, random_state=34)
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        X_train_normalized = features_normalization(X_train)
        X_test_normalized = features_normalization(X_test)
        
        # Training the k-NN algorithm
        predictions_train = knn_predict(X_train_normalized, y_train, X_train_normalized, k)
        accuracy_train = (np.array(predictions_train) == y_train).sum() / len(y_train)
        accuracy_values_train.append(accuracy_train)
        training_error = 1 - accuracy_train
        training_errors.append(training_error)

        # Evaluating the k-NN model
        predictions_test = knn_predict(X_train_normalized, y_train, X_test_normalized, k)
        accuracy_test = (np.array(predictions_test) == y_test).sum() / len(y_test)
        accuracy_values_test.append(accuracy_test)
        testing_error = 1 - accuracy_test
        testing_errors.append(testing_error)
        

    # Compute average and standard deviation for training set
    avg_accuracy_train = np.mean(accuracy_values_train)
    std_dev_train = np.std(accuracy_values_train)
    avg_accuracies_train.append(avg_accuracy_train)
    std_devs_train.append(std_dev_train)
    avg_training_error = np.mean(training_errors)
    avg_training_errors.append(avg_training_error)
    # Compute average and standard deviation for testing set
    avg_accuracy_test = np.mean(accuracy_values_test)
    std_dev_test = np.std(accuracy_values_test)
    avg_accuracies_test.append(avg_accuracy_test)
    std_devs_test.append(std_dev_test)
    avg_testing_error = np.mean(testing_errors)
    avg_testing_errors.append(avg_testing_error) 
 


# In[69]:


print("avg_accuracies_train: ",len(avg_accuracies_train))
print("std_devs_train: ",len(std_devs_train))
print("avg_accuracies_test: ",len(avg_accuracies_test))
print("std_devs_test: ",len(std_devs_test))
print("avg_training_errors: ",len(avg_training_errors))
print("avg_testing_errors: ",len(avg_testing_errors))


# In[70]:


print("avg_accuracies_train:", avg_accuracies_train)
print("std_devs_train:", std_devs_train)
print("avg_accuracies_test:", avg_accuracies_test)
print("std_devs_test:", std_devs_test)
print("avg_training_errors:", avg_training_errors)
print("avg_testing_errors:", avg_testing_errors)


# In[71]:


print("avg_accuracies_train: ",len(avg_accuracies_train))
print("std_devs_train: ",len(std_devs_train))
print("avg_accuracies_test: ",len(avg_accuracies_test))
print("std_devs_test: ",len(std_devs_test))


# In[72]:


import matplotlib.pyplot as plt
plt.errorbar(k_values, avg_accuracies_train, yerr=std_devs_train,fmt='o-',barsabove=False, label='training set')
plt.xlabel('k values')
plt.ylabel('Average Accuracy')
plt.title('k-NN Model Accuracy on TRAINING DATA')
plt.legend()
plt.show()


# In[73]:


import matplotlib.pyplot as plt
plt.errorbar(k_values, avg_accuracies_test, yerr=std_devs_test,fmt='o-',label='testing set')
plt.xlabel('k values')
plt.ylabel('Average Accuracy')
plt.title('k-NN Model Accuracy on TESTING DATA')
plt.legend()
plt.show()


# Improvement with k (k=1 to 5): Initially, as k increases from 1 to 5, testing accuracy improves. 
# 
# Optimal Generalization (k=15 to 25): There is a range of k values (approximately 15 to 25) where testing accuracy is highest. During this phase, the model effectively balances capturing patterns in the training set without sacrificing its ability to generalize to the testing set.
# 
# Overfitting (k > 25): Beyond a certain point (e.g., k > 25), testing accuracy starts to decline. This is a sign of overfitting to the training set, as the model becomes too generalized, missing important patterns present in the testing set.

# In[ ]:



Initially, with very low k values, the model fits the training data perfectly (overfitting), resulting in high accuracy.
Initial Overfitting (k=1): When k is very small (e.g., 1), the model is highly sensitive to individual data points, leading to a perfect fit for the training data. However, this is indicative of overfitting, as the model memorizes the training set but fails to generalize well to new instances.

(k=3 to 15): As k increases from 1 to around 15, training accuracy decreases. This is a crucial transition phase where the model starts to become less reliant on individual data points and captures more general trends. It strikes a better balance between fitting the training set and avoiding overfitting.

Underfitting (k > 15): For k values greater than 15, the model becomes too generalized, resulting in underfitting. It starts to neglect important patterns in the training set, leading to a decline in training accuracy.


# In[46]:


plt.plot(k_values, avg_training_errors, label='Training Error', marker='o')
plt.plot(k_values, avg_testing_errors, label='Testing Error', marker='o')
plt.xlabel('k values')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff Curve')
plt.legend()
plt.show()


# ## k-NN Algorithm without Normalization

# In[52]:


std_devs_train = []
std_devs_test = []

accuracy_values_train = []
accuracy_values_test = []

avg_accuracies_train = []
avg_accuracies_test = []

training_errors = []
testing_errors = []

avg_training_errors = []
avg_testing_errors = []
k_values = [k for k in list(range(1, 51, 2))]
iterations = 20
for k in k_values:
    for i in range(iterations):
        df_iris_shuffled = shuffle(df_iris)
        train_df, test_df = train_test_split(df_iris_shuffled, test_size=0.2, random_state=34)
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        print("i:",i)
        # Training the k-NN algorithm
        predictions_train = knn_predict(X_train, y_train, X_train, k)
        accuracy_train = (np.array(predictions_train) == y_train).sum() / len(y_train)
        accuracy_values_train.append(accuracy_train)
        training_error = 1 - accuracy_train
        training_errors.append(training_error)

        # Evaluating the k-NN model
        predictions_test = knn_predict(X_train, y_train, X_test, k)
        accuracy_test = (np.array(predictions_test) == y_test).sum() / len(y_test)
        accuracy_values_test.append(accuracy_test)
        testing_error = 1 - accuracy_test
        testing_errors.append(testing_error)
        

    # Compute average and standard deviation for training set
    avg_accuracy_train = np.mean(accuracy_values_train)
    std_dev_train = np.std(accuracy_values_train)
    avg_accuracies_train.append(avg_accuracy_train)
    std_devs_train.append(std_dev_train)
    avg_training_error = np.mean(training_errors)
    avg_training_errors.append(avg_training_error)
    # Compute average and standard deviation for testing set
    avg_accuracy_test = np.mean(accuracy_values_test)
    std_dev_test = np.std(accuracy_values_test)
    avg_accuracies_test.append(avg_accuracy_test)
    std_devs_test.append(std_dev_test)
    avg_testing_error = np.mean(testing_errors)
    avg_testing_errors.append(avg_testing_error) 
    
    print("k:",k)


# In[57]:


print("avg_accuracies_train:", avg_accuracies_train)
print("std_devs_train:", std_devs_train)
print("avg_accuracies_test:", avg_accuracies_test)
print("std_devs_test:", std_devs_test)
print("avg_training_errors:", avg_training_errors)
print("avg_testing_errors:", avg_testing_errors)


# In[53]:


print("avg_accuracies_train: ",len(avg_accuracies_train))
print("std_devs_train: ",len(std_devs_train))
print("avg_accuracies_test: ",len(avg_accuracies_test))
print("std_devs_test: ",len(std_devs_test))
print("avg_training_errors: ",len(avg_training_errors))
print("avg_testing_errors: ",len(avg_testing_errors))


# In[54]:


import matplotlib.pyplot as plt
plt.errorbar(k_values, avg_accuracies_train, yerr=std_devs_train,fmt='o-',barsabove=False, label='training set')
plt.xlabel('k values')
plt.ylabel('Average Accuracy')
plt.title('k-NN Model Accuracy on TRAINING DATA')
plt.legend()
plt.show()


# In[55]:


import matplotlib.pyplot as plt
plt.errorbar(k_values, avg_accuracies_test, yerr=std_devs_test,fmt='o-',label='testing set')
plt.xlabel('k values')
plt.ylabel('Average Accuracy')
plt.title('k-NN Model Accuracy on TESTING DATA')
plt.legend()
plt.show()


# In[56]:


plt.plot(k_values, avg_training_errors, label='Training Error', marker='o')
plt.plot(k_values, avg_testing_errors, label='Testing Error', marker='o')
plt.xlabel('k values')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff Curve')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:




