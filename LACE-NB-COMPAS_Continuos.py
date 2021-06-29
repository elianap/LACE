#!/usr/bin/env python
# coding: utf-8

# In[2]:


from scipy.io.arff import loadarff
from src.dataset import Dataset


# # Import data

# In[3]:


from src.import_datasets import import_process_compas, getAttributes

dataset_name = "COMPAS"
cl = "Medium-Low"


# In[4]:


discretize = False
featureMasking = True

data = import_process_compas(discretize=discretize, ageTmp=True).drop(
    columns="age_cat"
)


# In[5]:


data.head()


# ## Discretized data

# In[6]:


data_X_discretized = (
        import_process_compas(discretize=True)
        .drop(columns="class")
        .rename(columns={"age_cat": "age"})
    )


# # Train and explain dataset

# In[7]:


from sklearn.model_selection import train_test_split

df_train, df_explain = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["class"]
)


# In[8]:


attributes = getAttributes(data, featureMasking=featureMasking)


# In[9]:


d_train = Dataset(
    df_train.values,
    attributes,
    featureMasking=featureMasking,
    discreteDataset=data_X_discretized.loc[df_train.index],
)


# In[10]:


d_explain = Dataset(
    df_explain.values,
    attributes,
    column_encoders=d_train._column_encoders,
    featureMasking=True,
    discreteDataset=data_X_discretized.loc[df_explain.index],
)


# # Classifier training

# In[11]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(d_train.X_numpy(), d_train.Y_numpy())


# # X-PLAIN

# In[12]:


from src.LACE_explainer import LACE_explainer

explainer = LACE_explainer(clf, d_train, dataset_name="COMPAS")


# # Explain instances

# Input: instance to explain and the target class (the class w.r.t. the prediction difference is computed)

# In[13]:


i=2
instance = d_explain[i]
infos = {"d": dataset_name, "model": "RF"}
instance = d_explain[i]
instance_discretized = d_explain.getDiscretizedInstance(i)
# print(instance_discretized)
# print(instance)
explanation_fm = explainer.explain_instance(
    instance,
    cl,
    featureMasking=featureMasking,
    discretizedInstance=instance_discretized,
)
explanation_fm.plotExplanation(saveFig=True, figName=f"expl_{i}{featureMasking}_NB")
print(explanation_fm.diff_single)
print(explanation_fm.map_difference)
print(d_explain.X_decoded().iloc[i])
print(explanation_fm.estimateUserRule([1, 2], cl))
print(explanation_fm.checkCorrectness(cl))


# In[1]:


import os
os.getcwd()


# In[ ]:




