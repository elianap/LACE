import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", -1)

dirO = "./out/out_artificial/RF"

outputF = False

toCompute = False


import numpy as np
import pandas as pd

np.random.seed(0)
data = np.random.randint(10, size=(5000, 10))
# dataset_name="artificial_10"
features = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
df = pd.DataFrame(data=data, columns=features)
df["class"] = 0
indexes = df.loc[(df["a"] < 5) & (df["b"] < 5)].index
df.loc[indexes, "class"] = 1
indexes = df.loc[(df["h"] > 6) & (df["i"] > 6) & (df["j"] > 6)].index
df.loc[indexes, "class"] = 1
# df["class"].value_counts()
# data=np.concatenate([np.full((5000, 10), 0), np.full((5000, 10), 1)])
# np.random.seed(7)
# for i in range(data.shape[1]):
#     np.random.shuffle(data[:,i])
# df_artificial_3=pd.DataFrame(data=data, columns=features_3)
# df_artificial_3["class"]=0
# indexes=df_artificial_3.loc[((df_artificial_3["a"]==df_artificial_3["b"])& (df_artificial_3["a"]==df_artificial_3["c"]))].index
# df_artificial_3.loc[indexes, "class"]=1
# df_artificial_3["class"].value_counts()
df.head()


feature_names = df.columns.drop(["class"])
categorical_features = list(
    df[feature_names].select_dtypes(include=["object", "category"]).columns
)
categorical_features_pos = [
    i for i, k in enumerate(feature_names) if k in categorical_features
]
continuos_features = list(set(feature_names) - set(categorical_features))
continuos_features_pos = [
    i for i, k in enumerate(feature_names) if k in continuos_features
]


# ## Encode class


from sklearn.preprocessing import LabelEncoder

labels = df["class"].values
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_


# ## Label encoding

# For LIME input


def getLabelEncodingMapping(df, feature_names, categorical_features_pos):
    from sklearn.preprocessing import LabelEncoder

    data_LE = df.copy()
    categorical_names = {}
    encoders = {}
    for i in categorical_features_pos:
        feature = feature_names[i]
        le = LabelEncoder()
        le.fit(data_LE[feature].values)
        data_LE[feature] = le.transform(data_LE[feature].values).astype(
            df[feature].dtype
        )
        categorical_names[i] = le.classes_
        encoders[feature] = le
    return data_LE, categorical_names, encoders


# categorical_features_pos_names = [ for i in categorical_features_pos]

data_LE, categorical_names_LE, encoders = getLabelEncodingMapping(
    df, feature_names, categorical_features_pos
)


# ## Split Train and Test


from sklearn import model_selection

np.random.seed(1)
train, test, labels_train, labels_test = model_selection.train_test_split(
    data_LE[feature_names], labels, train_size=0.8
)


# train=data_LE[feature_names].copy()
# test=data_LE[feature_names].copy()
# labels_train=labels
# labels_test=labels


test


# ## One hot encoding & Scale continuos

# For training the classifier


toScale = False


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df_X_encoded = pd.DataFrame()
df_X_continuos = pd.DataFrame()

encoder = OneHotEncoder(sparse=False)

if categorical_features:
    df_X_encoded = pd.DataFrame(
        encoder.fit_transform(train[categorical_features]), index=train.index
    )

    df_X_encoded.columns = encoder.get_feature_names(categorical_features)

if continuos_features:
    df_X_continuos = train.drop(categorical_features, axis=1)

    if toScale:
        from sklearn import preprocessing

        x_cont = df_X_continuos.copy()
        min_max_scaler = preprocessing.MinMaxScaler()
        x_cont_scaled = min_max_scaler.fit_transform(x_cont.values)
        df_X_continuos = pd.DataFrame(
            x_cont_scaled, columns=df_X_continuos.columns, index=train.index
        )

OH_X_train = pd.concat([df_X_encoded, df_X_continuos], axis=1)
OH_X_train.head()


# ## Training


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(OH_X_train.values, labels_train)


# ## Predict function


predict_fn = lambda x: clf.predict_proba(x)


# predict_fn =lambda x: clf.predict_proba(np.hstack((encoder.transform(x[:,categorical_features_pos]),\
#                                         min_max_scaler.transform(x[:,continuos_features_pos]))))
#                                                  #x[:,continuos_features_pos].astype(float)))    )


# ## Predict


test.head()


predicted = np.argmax(predict_fn(test.values), axis=1)
FP = [i for i, p in enumerate(predicted) if p == 1 and labels_test[i] == 0]
FN = [i for i, p in enumerate(predicted) if p == 0 and labels_test[i] == 1]
mispredicted = [i for i, p in enumerate(predicted) if p != labels_test[i]]
correct_prediction = [i for i, p in enumerate(predicted) if p == labels_test[i]]


from sklearn.metrics import accuracy_score

print("Accuracy: ", round(accuracy_score(labels_test, predicted), 4))


# # Processing DivExplorer


df_test = df[feature_names].loc[test.index]


from src.import_datasets import discretize

test_discretized_labels = df_test.copy()


predicted = [class_names[v] for v in np.argmax(predict_fn(test.values), axis=1)]
true_labels = [class_names[v] for v in labels_test]


test_discretized_labels["class"] = true_labels
test_discretized_labels["predicted"] = predicted


from sklearn.metrics import accuracy_score

print(
    "Accuracy: ",
    round(
        accuracy_score(
            test_discretized_labels["class"], test_discretized_labels["predicted"]
        ),
        4,
    ),
)


# # Processing SHAP


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df_X_test_encoded = pd.DataFrame()
df_X_test_continuos = pd.DataFrame()

if categorical_features:
    df_X_test_encoded = pd.DataFrame(
        encoder.transform(test[categorical_features]), index=test.index
    )

    df_X_test_encoded.columns = encoder.get_feature_names(categorical_features)

if continuos_features:
    df_X_test_continuos = test.drop(categorical_features, axis=1)

    if toScale:
        from sklearn import preprocessing

        x_test_cont = df_X_test_continuos.copy()
        x_test_cont_scaled = min_max_scaler.transform(x_test_cont.values)
        df_X_test_continuos = pd.DataFrame(
            x_test_cont_scaled, columns=df_X_test_continuos.columns, index=test.index
        )


OH_X_test = pd.concat([df_X_test_encoded, df_X_test_continuos], axis=1)
OH_X_test.head()


categorical_names_map = {
    categorical_features[k]: val for k, val in categorical_names_LE.items()
}


oh_columns = list(OH_X_test.columns)
oh_columns_categorical = [
    f'{"_".join(c.split("_")[0:-1])}_{categorical_names_map["_".join(c.split("_")[0:-1])][int(float(c.split("_")[-1]))]}'
    for c in oh_columns
    if "_".join(c.split("_")[0:-1]) in categorical_features
]
oh_columns = [
    f'{"_".join(c.split("_")[0:-1])}_{categorical_names_map["_".join(c.split("_")[0:-1])][int(float(c.split("_")[-1]))]}'
    if "_".join(c.split("_")[0:-1]) in categorical_features
    else c
    for c in oh_columns
]
print(oh_columns)


OH_X_test_cols = OH_X_test.copy()
OH_X_test_cols.columns = oh_columns
OH_X_test_cols.head()


# # Processing SliceFinder


y_predict_prob_test = predict_fn(test.values)
# clf.predict_proba(OH_X_test)


y_predict_test = np.argmax(predict_fn(test.values), axis=1)
# clf.predict(OH_X_test)


from sklearn.metrics import log_loss
import functools

metric = log_loss
from sklearn.utils.multiclass import unique_labels

classes = unique_labels(labels_train)

y_p_test = list(map(functools.partial(np.expand_dims, axis=0), y_predict_prob_test))
y_test = list(map(functools.partial(np.expand_dims, axis=0), labels_test))
loss_list_FP_test = np.array(
    list(map(functools.partial(metric, labels=classes), y_test, y_p_test))
)


from collections import Counter

print(
    {
        k: (v, round(v / len(labels_test), 4))
        for k, v in dict(Counter(labels_test)).items()
    }
)
print(
    {
        k: (v, round(v / len(y_predict_test), 4))
        for k, v in dict(Counter(y_predict_test)).items()
    }
)


# ## Discretization


train_discretized = train.copy()
train_discretized.reset_index(drop=True, inplace=True)
from src.import_datasets import discretize

train_discretized[continuos_features] = discretize(
    train_discretized, attributes=continuos_features
)
# test_discretized.head()
# test_discretized=test.copy()
# test_discretized.reset_index(drop=True, inplace=True)
train_discretized.head()


test_discretized = test.copy()
test_discretized.reset_index(drop=True, inplace=True)
from src.import_datasets import discretize

test_discretized[continuos_features] = discretize(
    test_discretized, attributes=continuos_features
)
# test_discretized.head()
# test_discretized=test.copy()
# test_discretized.reset_index(drop=True, inplace=True)
test_discretized.head()


# # LIME
toComputeLime = False
if toComputeLime:
    import sklearn
    import sklearn.datasets
    import sklearn.ensemble
    import lime
    import lime.lime_tabular

    np.random.seed(1)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        train.values,
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_features_pos,
        categorical_names=categorical_names_LE,
        kernel_width=3,
    )

    np.random.seed(1)
    i = 0
    exp = lime_explainer.explain_instance(
        test.iloc[i].values, predict_fn, num_features=6
    )
    exp.show_in_notebook(labels=[1], show_predicted_value=False)

    def plot_lime_explanation(
        lime_explanation, label=1, pred=None, true_label=None, fontsize=14
    ):
        import matplotlib.pyplot as plt

        exp = lime_explanation.as_list(label=label)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ["#fa8023" if x > 0 else "#3574b2" for x in vals]
        pos = np.arange(len(exp)) + 0.5
        plt.barh(pos, vals, align="center", color=colors, height=0.4)
        plt.yticks(pos, names, fontsize=fontsize)
        plt.xlabel(f"Class {lime_explanation.class_names[label]}", fontsize=fontsize)
        if lime_explanation.mode == "classification":
            title = f"Local explanation for class {lime_explanation.class_names[label]}\n predicted={pred}  true class={true_label}"
        else:
            title = "Local explanation"
        plt.title(title, fontsize=fontsize)
        return fig

    if outputF:
        from pathlib import Path

        outputdir = f"{dirO}/lime_out"
        Path(outputdir).mkdir(parents=True, exist_ok=True)
    for i in range(1, 2):
        exp = lime_explainer.explain_instance(
            test.iloc[i].values, predict_fn, num_features=6
        )
        fig = plot_lime_explanation(
            exp,
            pred=class_names[np.argmax(predict_fn(test.iloc[i : i + 1].values))],
            true_label=class_names[labels_test[i]],
        )
        fig.set_size_inches(7.25, 4.25)
        fig.tight_layout()

        if outputF:
            fig.savefig(
                f"{outputdir}/Lime" + str(i) + ".png",
                facecolor="white",
                transparent=False,
                bbox_inches="tight",
            )

    # ## False Positives

    if toCompute:
        if outputF:
            from pathlib import Path

            outputdir = f"{dirO}/lime_out/FP"
            Path(outputdir).mkdir(parents=True, exist_ok=True)
        for i in FP[0:5]:
            exp = lime_explainer.explain_instance(
                test.iloc[i].values, predict_fn, num_features=6
            )
            fig = plot_lime_explanation(
                exp,
                pred=class_names[np.argmax(predict_fn(test.iloc[i : i + 1].values))],
                true_label=class_names[labels_test[i]],
            )
            fig.set_size_inches(6.5, 3.5)
            fig.tight_layout()
            if outputF:
                fig.savefig(
                    f"{outputdir}/Lime" + str(i) + ".png",
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                )

    # ## False negatives

    if toCompute:
        if outputF:
            from pathlib import Path

            outputdir = f"{dirO}/lime_out/FN"
            Path(outputdir).mkdir(parents=True, exist_ok=True)
        for i in FN[0:5]:
            exp = lime_explainer.explain_instance(
                test.iloc[i].values, predict_fn, num_features=6
            )
            fig = plot_lime_explanation(
                exp,
                pred=class_names[np.argmax(predict_fn(test.iloc[i : i + 1].values))],
                true_label=class_names[labels_test[i]],
            )
            fig.set_size_inches(6.5, 3.5)
            fig.tight_layout()
            if outputF:
                fig.savefig(
                    f"{outputdir}/Lime" + str(i) + ".png",
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                )

    # ## Correctly classified

    i = correct_prediction[0]
    test.iloc[i]

    if toCompute:
        if outputF:
            from pathlib import Path

            outputdir = f"{dirO}/lime_out/correct"
            Path(outputdir).mkdir(parents=True, exist_ok=True)
        for i in correct_prediction[0:10]:
            exp = lime_explainer.explain_instance(
                test.iloc[i].values, predict_fn, num_features=6
            )
            fig = plot_lime_explanation(
                exp,
                pred=class_names[np.argmax(predict_fn(test.iloc[i : i + 1].values))],
                true_label=class_names[labels_test[i]],
            )
            fig.set_size_inches(6.5, 3.5)
            fig.tight_layout()
            if outputF:
                fig.savefig(
                    f"{outputdir}/Lime" + str(i) + ".png",
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                )


# # SHAP
toComputeShap = False
if toComputeShap:
    if toCompute:
        import shap

        shap_explainer = shap.TreeExplainer(clf)
        # explainer = shap.KernelExplainer(predict_fn, OH_X_train.values, link="logit")

        shap_values = shap_explainer.shap_values(OH_X_test_cols.iloc[0:])
        # shap_values = explainer.shap_values(OH_X_test_cols.iloc[0:5])

    if toCompute:
        shap.initjs()
        shap.force_plot(
            shap_explainer.expected_value[0],
            shap_values[0][0, :],
            OH_X_test_cols.iloc[0:1],
        )

    if toCompute:
        shap.summary_plot(
            shap_values,
            OH_X_test_cols.iloc[0:],
            plot_type="bar",
            class_names=["No Recidivism", "Recidivism"],
        )

    def convertInstance(instance, categorical_features, continuos_features):
        instance = dict(instance.items())
        matching_instance = {
            "_".join(k.split("_")[0:-1]): k.split("_")[-1]
            for k, v in instance.items()
            if "_".join(k.split("_")[0:-1]) in categorical_features and v == 1
        }
        matching_instance.update(
            {k: v for k, v in instance.items() if k in continuos_features}
        )
        return matching_instance

    def plotShapValues(
        shap_values_i, names, fontsize=14, target_class="", pred="", true=""
    ):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        vals = list(shap_values_i)
        vals.reverse()
        names.reverse()
        colors = ["#fa8023" if x > 0 else "#3574b2" for x in vals]
        pos = np.arange(len(vals)) + 0.5
        plt.barh(pos, vals, align="center", color=colors, height=0.4)
        plt.yticks(pos, names, fontsize=fontsize)
        plt.xlabel(f"Class {target_class}", fontsize=fontsize)
        plt.title(f"Predicted={pred} True_class={true}")
        return fig

    # https://github.com/slundberg/shap/issues/397
    def sumCategories(vals, names, categorical_features, matching_instance):
        vals = list(vals)
        d = {
            ("_".join(k.split("_")[0:-1]), k.split("_")[-1])
            if "_".join(k.split("_")[0:-1]) in categorical_features
            else (k, k): v
            for k, v in zip(names, vals)
        }
        # print(d)
        from itertools import groupby

        shap_values_sum_categories = {
            key: sum([g[1] for g in group])
            for key, group in groupby(d.items(), lambda x: x[0][0])
        }
        shap_values_sum_categories = {
            f"{k}={matching_instance[k]}" if k in matching_instance else k: v
            for k, v in shap_values_sum_categories.items()
        }
        return shap_values_sum_categories

    import shap

    # explainer = shap.TreeExplainer(clf)

    # ## FP

    if outputF:
        from pathlib import Path

        outputdir = f"{dirO}/SHAP_out/FP"
        Path(outputdir).mkdir(parents=True, exist_ok=True)
    fontsize = 14
    if toCompute:
        for i in FP[0:5]:
            class_id = 0
            import matplotlib.pyplot as plt

            fig = plt.figure()
            instance = OH_X_test_cols.iloc[i]
            shap_values = shap_explainer.shap_values(instance)
            matching_instance = convertInstance(
                instance, categorical_features, continuos_features
            )

            sum_shap_for_categories = sumCategories(
                shap_values[class_id],
                oh_columns,
                categorical_features,
                matching_instance,
            )
            fig = plotShapValues(
                list(sum_shap_for_categories.values()),
                list(sum_shap_for_categories.keys()),
                target_class=class_names[class_id],
                pred=class_names[clf.predict([instance])[0]],
                true=class_names[labels_test[i]],
            )
            fig.set_size_inches(6.5, 3.5)
            fig.tight_layout()
            if outputF:
                fig.savefig(
                    f"{outputdir}/SHAP" + str(i) + ".png",
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                )

    class_id = 0
    if toCompute:
        for i in FP[0:1]:
            instance = OH_X_test_cols.iloc[i]
            shap_values = shap_explainer.shap_values(instance)
            plotShapValues(
                shap_values[class_id],
                oh_columns,
                target_class=class_names[class_id],
                pred=class_names[clf.predict([instance])[0]],
                true=class_names[labels_test[i]],
            )

    # ## FN

    if outputF:
        from pathlib import Path

        outputdir = f"{dirO}/SHAP_out/FN"
        Path(outputdir).mkdir(parents=True, exist_ok=True)
    fontsize = 14
    if toCompute:
        for i in FN[0:2]:
            class_id = 0
            import matplotlib.pyplot as plt

            fig = plt.figure()
            instance = OH_X_test_cols.iloc[i]
            shap_values = shap_explainer.shap_values(instance)
            print(shap_values)
            matching_instance = convertInstance(
                instance, categorical_features, continuos_features
            )

            sum_shap_for_categories = sumCategories(
                shap_values[class_id],
                oh_columns,
                categorical_features,
                matching_instance,
            )
            print(sum_shap_for_categories)
            fig = plotShapValues(
                list(sum_shap_for_categories.values()),
                list(sum_shap_for_categories.keys()),
                target_class=class_names[class_id],
                pred=class_names[clf.predict([instance])[0]],
                true=class_names[labels_test[i]],
            )
            fig.set_size_inches(6.5, 3.5)
            fig.tight_layout()
            if outputF:
                fig.savefig(
                    f"{outputdir}/SHAP" + str(i) + ".png",
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                )

    class_id = 0
    if toCompute:
        for i in FN[0:1]:
            instance = OH_X_test_cols.iloc[i]
            shap_values = shap_explainer.shap_values(instance)
            plotShapValues(
                shap_values[class_id],
                oh_columns,
                target_class=class_names[class_id],
                pred=class_names[clf.predict([instance])[0]],
                true=class_names[labels_test[i]],
            )

    # ## Correctly classified

    i = correct_prediction[0]
    OH_X_test_cols.iloc[i]

    if outputF:
        from pathlib import Path

        outputdir = f"{dirO}/SHAP_out/correct"
        Path(outputdir).mkdir(parents=True, exist_ok=True)
    fontsize = 14
    if toCompute:
        for i in correct_prediction[0:10]:
            class_id = 1
            import matplotlib.pyplot as plt

            fig = plt.figure()
            instance = OH_X_test_cols.iloc[i]
            shap_values = shap_explainer.shap_values(instance)
            matching_instance = convertInstance(
                instance, categorical_features, continuos_features
            )

            sum_shap_for_categories = sumCategories(
                shap_values[class_id],
                oh_columns,
                categorical_features,
                matching_instance,
            )
            fig = plotShapValues(
                list(sum_shap_for_categories.values()),
                list(sum_shap_for_categories.keys()),
                target_class=class_names[class_id],
                pred=class_names[clf.predict([instance])[0]],
                true=class_names[labels_test[i]],
            )
            fig.set_size_inches(6.5, 3.5)
            fig.tight_layout()
            if outputF:
                fig.savefig(
                    f"{outputdir}/SHAP" + str(i) + ".png",
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                )

    class_id = 0

    if toCompute:
        for i in correct_prediction[0:1]:
            instance = OH_X_test_cols.iloc[i]
            shap_values = shap_explainer.shap_values(instance)
            plotShapValues(
                shap_values[class_id],
                oh_columns,
                target_class=class_names[class_id],
                pred=class_names[clf.predict([instance])[0]],
                true=class_names[labels_test[i]],
            )


# # LACE


from src.import_datasets import getAttributes


cols = [
    "age_cat",
    "c_charge_degree",
    "race",
    "sex",
    "priors_count",
    "length_of_stay",
    "class",
]
df_cols = df  # [cols]


featureMasking = True
attributes = getAttributes(df_cols, featureMasking=featureMasking)


# ## Discretized


data_X_discretized = df[feature_names].copy()
data_X_discretized.reset_index(drop=True, inplace=True)


train_orig = df_cols.iloc[train.index]
test_orig = df_cols.iloc[test.index]


# train_orig.loc[labels_test]


# encoder_t=lambda x: np.hstack((encoder.transform(x[:,categorical_features_pos]),\
#                                         min_max_scaler.transform(x[:,continuos_features_pos])))


# le.classes_
# encoders
# all_encoders
# encoders
# attributes
correct_prediction[0]


from src.dataset import Dataset
from copy import deepcopy

all_encoders = deepcopy(encoders)
all_encoders.update({"class": le})
d_train = Dataset(
    train_orig.values,
    attributes,
    # column_encoders=all_encoders,
    featureMasking=True,
    discreteDataset=train_discretized,
    # encoder_nn=encoder_t
)


d_explain = Dataset(
    test_orig.values,
    attributes,
    # column_encoders=all_encoders,
    featureMasking=True,
    discreteDataset=test_discretized,
    # encoder_nn=encoder_t
)


# ## Explainer


from src.LACE_explainer import LACE_explainer

lace_explainer = LACE_explainer(d_train, predict_fn, dataset_name="COMPAS")


d_train.X()


# ## Explain instance


dataset_name = "COMPAS"
cl = 1

id_i = 3  # FN[3]
for id_i in [0]:
    print(d_explain[id_i])
    instance = d_explain[id_i]
    infos = {"model": "RF"}
    instance_discretized = d_explain.getDiscretizedInstance(id_i)
    explanation_fm = lace_explainer.explain_instance(
        instance,
        cl,
        featureMasking=featureMasking,
        discretizedInstance=instance_discretized,
    )
    explanation_fm.plotExplanation(
        saveFig=True
    )  # , figName=f"{dirO}/expl_{id_i}{featureMasking}_NB_aa")
    print(explanation_fm.diff_single)
    print(explanation_fm.rules_hr)


# a=d_train.discreteDataset.copy()
# a=a.astype(object)
# a.dtypes
attribute = "a"
# [v for k, v in attributes if k==attribute]


d = d_train.X_decoded()


pred = predict_fn(d_explain[id_i][:-1].to_numpy().reshape(1, -1).reshape(1, -1))[0]
target_class_index = list(le.classes_).index(cl)
print(d_explain[id_i])
a = deepcopy(d_explain[id_i][:-1])
diff = {}
print(predict_fn(d_explain[id_i][:-1].values.reshape(1, -1)))
attribute = "a"
c = 0
for e, v in enumerate(list(map(int, dict(attributes)[attribute]))):
    # print(e,v)
    a[attribute] = e
    # print(a)
    print(e, v, "pred:", predict_fn(a.values.reshape(1, -1)), end=" ")
    diff[v] = predict_fn(a.values.reshape(1, -1))
    # print("diff:", diff[v], diff[v][0], end=" ")
    print("diff:", diff[v], end=" ")
    print(diff[v][0][target_class_index], end=" ")
    print("freq:", len(d.loc[d[attribute] == v]) / len(d), end=" ")
    c += diff[v][0][target_class_index] * len(d.loc[d[attribute] == v]) / len(d)
    print(
        "val", diff[v][0][target_class_index] * len(d.loc[d[attribute] == v]) / len(d)
    )
print(c)
print(pred[target_class_index] - c)


# predicted=np.argmax(predict_fn(test.values), axis=1)
predicted = np.argmax(predict_fn(test.values), axis=1)
predicted_explain = np.argmax(predict_fn(d_explain.X_numpy()), axis=1)
# print(test.values==d_explain.X_numpy())
# predicted


import scipy

attrs = [a1 for a1, _ in d_train.attributes()]
P = len(d_train.X_numpy()[0])
print(d_explain.X_decoded().loc[id_i])
print(d_explain[id_i])
x = d_explain[id_i][:-1].to_numpy().reshape(1, -1)
masker_data = d_train.X_numpy()
masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
mask = np.zeros(P, dtype=np.bool)
f = predict_fn
a = masker_data[0:5]
for i in range(P):
    print(attrs[i], end=" ")
    s = []
    weight = 1 / (scipy.special.comb(P - 1, len(s)) * P)
    # print(weight)
    mask[:] = 0
    mask[list(s)] = 1
    f_without_i = f(masker(x, mask)).mean(0)
    # print(masker(x, mask)[0:5])
    a = masker(x, mask)[0:5]
    mask[i] = 1
    f_with_i = f(masker(x, mask)).mean(0)
    # print(masker(x, mask)[0:5])
    diff = f_with_i - f_without_i
    print(diff[target_class_index], "      ", diff)
    print()


# New version with feature masking
def _compute_prediction_difference_single_NEW(
    encoded_instance, predict_fn, class_prob, target_class_index, training_dataset
):

    P = len(training_dataset.attributes())
    instance_i = deepcopy(encoded_instance)
    instance_i = instance_i[:-1].to_numpy().reshape(1, -1)
    masker_data = training_dataset.X_numpy()
    masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
    mask = np.zeros(P, dtype=np.bool)
    f = predict_fn

    class_prob = predict_fn(instance_i)[0]

    import scipy

    diff_f = np.zeros((P, len(training_dataset.class_values())))
    # For each attribute of the instance
    for i in range(P):
        mask[:] = 1
        mask[i] = 0
        avg_remove_f_i = f(masker(instance_i, mask)).mean(0)

        diff_f[i] = f(instance_i.reshape(1, -1))[0] - avg_remove_f_i

    s = diff_f.shape
    attribute_pred_difference = [np.zeros(s[0]) for j in range(s[1])]
    for j in range(s[1]):
        attribute_pred_difference[j] = diff_f[:, j]
    return list(
        attribute_pred_difference[target_class_index]
    )  # TODO --> save overall!!!


print(explanation_fm.diff_single)
print(explanation_fm.map_difference)
# print(d_explain.X_decoded().iloc[i])
# print(explanation_fm.estimateUserRule([1, 2], cl))
print(explanation_fm.checkCorrectness(cl))