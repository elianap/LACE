# coding: utf-8

# In[1]:


from scipy.io.arff import loadarff
from src.dataset import Dataset

# from src.utils import *
from src.import_datasets import importArff, getAttributes


# In[2]:


# # Import and discretize data

# In[3]:


dataset_name = "monks"
data = importArff("./datasets/monks.arff")


# In[4]:


data.head()


# # Train and explain dataset

# In[5]:


from sklearn.model_selection import train_test_split

df_train, df_explain = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["y"]
)


# In[6]:


attributes = getAttributes(data)
d_train = Dataset(df_train.values, attributes)
d_explain = Dataset(
    df_explain.values, attributes, column_encoders=d_train._column_encoders
)


# In[7]:


df_train.head(5)


# # Classifier training

# In[8]:


from sklearn.neural_network import MLPClassifier

type_clf = "NN"
clf = MLPClassifier(random_state=42)
clf.fit(d_train.X_numpy(), d_train.Y_numpy())


# In[9]:


d_train.X().head()


# # X-PLAIN

# X-PLAIN input: classifier (model agnostic) and the training data to compute the locality

# In[10]:


from src.LACE_explainer import LACE_explainer

explainer = LACE_explainer(clf, d_train, dataset_name="monks")


l = {}
i = 5
cl = 0
instance = d_explain[i]

explanation_fm = explainer.explain_instance(instance, f"{cl}", featureMasking=True)
explanation = explainer.explain_instance(instance, f"{cl}", featureMasking=False)

# l[i] = {
#     k: explanation_fm.diff_single[k] - explanation.diff_single[k]
#     for k in range(0, len(explanation.diff_single))
# }

# rule = [1, 2, 3, 4, 5, 6]
# rem_all = explanation_fm.estimateUserRule(rule, f"{cl}", featureMasking=True)
# prior_prob = explanation_fm.LACE_explainer_o.train_dataset.Y().value_counts()[cl] / len(
#     explanation_fm.LACE_explainer_o.train_dataset.Y()
# )
# diff_fm = (explanation_fm.prob - prior_prob) - rem_all[",".join(map(str, rule))]
# rem_all = explanation.estimateUserRule(rule, f"{cl}", featureMasking=True)
# prior_prob = explanation.LACE_explainer_o.train_dataset.Y().value_counts()[cl] / len(
#     explanation.LACE_explainer_o.train_dataset.Y()
# )
# diff = (explanation.prob - prior_prob) - rem_all[",".join(map(str, rule))]
# print(diff_fm, diff)
# print(diff_fm - diff)
# print(diff - diff_fm)
# print(l[i])
