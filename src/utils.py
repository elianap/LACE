#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def savePickle(model, dirO, name):
    import pickle

    # createDir(dirO)
    import os

    if not os.path.exists(dirO):
        os.makedirs(dirO)
    with open(dirO + "/" + name + ".pickle", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def openPickle(dirO, name):
    import os.path
    import pickle
    from os import path

    if path.exists(dirO + "/" + name + ".pickle"):
        with open(dirO + "/" + name + ".pickle", "rb") as handle:
            return pickle.load(handle)
    else:
        return False


def get_classifier(classifier_name: str):
    if classifier_name == "Categorical Naive Bayes":
        from sklearn.naive_bayes import CategoricalNB

        skl_clf = CategoricalNB()

        return skl_clf

    if classifier_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        skl_clf = RandomForestClassifier(random_state=42)

        return skl_clf

    if classifier_name == "Neural Network":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        pipe = make_pipeline(
            OneHotEncoder(), MLPClassifier(random_state=42, max_iter=1000)
        )
        skl_clf = pipe

        return skl_clf

    raise ValueError("Classifier not available")


def plotExplanation(e, infos={}):
    import matplotlib.pyplot as plt

    # plt.style.use('seaborn-talk')

    fig, pred_ax = plt.subplots(1, 1)

    attrs = [f"{k}" for k, v in (e["instance"].items())][:-1]
    # values = [f"{k}={v}" for k, v in (e["instance"].items())]
    # nls = ",\n".join(values)
    infos_t = " ".join(
        [f"{k}={infos[k]}" if k in infos else "" for k in ["x", "d", "model"]]
    )
    # " ".join([f"{k}={v}" for k, v in infos.items()])
    title = f"{infos_t}\n p(class={e['target_class']}|x)={e['prob']:.3f} true class={e['instance'].iloc[-1]}"

    pred_ax.set_title(title)
    pred_ax.set_xlabel(f"Δ target class={e['instance'].iloc[-1]}")

    dict_instance = dict(
        enumerate([f"{k}={v}" for k, v in e["instance"].to_dict().items()], start=1)
    )
    rules = [
        ", ".join([dict_instance[int(i)] for i in rule_i.split(",")])
        for rule_i in e["map_difference"]
    ]
    mapping_rules = {f"Rule_{i+1}": rules[i] for i in range(0, len(rules))}

    pred_ax.barh(
        attrs + list(mapping_rules.keys()),
        width=e["diff_single"] + list(e["map_difference"].values()),
        align="center",
        color="#bee2e8",
        linewidth="1",
        edgecolor="black",
    )
    pred_ax.invert_yaxis()
    print([f"{k}={{{v}}}" for k, v in mapping_rules.items()])
    # fig.show()
    # plt.close()
    return fig


def verifyAttributePredictionDifference(
    data, instance_x, attribute, predict_fn, encoders, cl, le
):
    pred = predict_fn(instance_x[:-1].to_numpy().reshape(1, -1).reshape(1, -1))[0]
    target_class_index = list(le.classes_).index(cl)
    print(instance_x)
    # le: label encoding
    from copy import deepcopy

    a = deepcopy(instance_x)
    diff = {}
    print(predict_fn(instance_x.values.reshape(1, -1)))
    c = 0
    for e, v in enumerate(encoders[attribute].classes_):
        # print(e,v)
        a[attribute] = e
        # print(a)
        print(e, v, "pred:", predict_fn(a.values.reshape(1, -1)), end=" ")
        diff[v] = predict_fn(a.values.reshape(1, -1))
        # print("diff:", diff[v], diff[v][0], end=" ")
        print("diff:", diff[v], end=" ")
        print(diff[v][0][target_class_index], end=" ")
        print("freq:", len(data.loc[data[attribute] == v]) / len(data), end=" ")
        c += (
            diff[v][0][target_class_index]
            * len(data.loc[data[attribute] == v])
            / len(data)
        )
        print(
            "val",
            diff[v][0][target_class_index]
            * len(data.loc[data[attribute] == v])
            / len(data),
        )
    print("Total average value", c)
    print("Prediction difference", pred[target_class_index] - c)


def saveJson(json_obj, name, dirO="./"):
    import json

    with open(f"{dirO}/{name}.json", "w") as fp:
        json.dump(json_obj, fp)