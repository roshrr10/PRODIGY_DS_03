#!/usr/bin/env python
# coding: utf-8

# # TASK 3

# AIM: To build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. This involves preprocessing the data, training the model, evaluating its performance, and visualizing the decision tree to understand the decision-making process.

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset
file_path = 'Social_Network_Ads.csv'
data = pd.read_csv(file_path)


# In[3]:


# Display the first few rows of the dataset
data.head()


# In[4]:


# Preprocess the data
# Convert categorical variables to numeric using one-hot encoding
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)


# In[5]:


# Drop the 'User ID' column as it is not needed for prediction
data = data.drop('User ID', axis=1)


# In[6]:


# Split the dataset into features (X) and target (y)
X = data.drop('Purchased', axis=1)  # Features
y = data['Purchased']  # Target variable


# In[7]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[8]:


# Initialize the classifier
clf = DecisionTreeClassifier(random_state=42)


# In[9]:


# Train the classifier
clf.fit(X_train, y_train)


# In[10]:


# Make predictions on the test set
y_pred = clf.predict(X_test)


# In[11]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# The accuracy score provides a quick measure of the model's performance. A higher accuracy indicates that the model is correctly predicting the target variable for a large proportion of the test data.

# In[12]:


# Print classification report
print(classification_report(y_test, y_pred))


# The classification report provides a detailed view of the model's performance for each class (Purchased vs. Not Purchased).

# In[13]:


# Print confusion matrix
print(confusion_matrix(y_test, y_pred))


# The confusion matrix provides a detailed breakdown of the model's predictions vs. actual values.

# In[14]:


# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=['Not Purchased', 'Purchased'], rounded=True)
plt.show()


# - The visualization helps understand how the model makes decisions.
# - Each node represents a feature and a threshold, and each leaf node represents a class prediction.
# - The depth and structure of the tree can provide insights into how complex the model is.
