def plotShapValues(
    shap_values_i, cols_names, fontsize=14, target_class="", pred="", true=""
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    vals = list(shap_values_i)[::-1]
    names = cols_names[::-1]
    colors = ["#fa8023" if x > 0 else "#3574b2" for x in vals]
    pos = np.arange(len(vals)) + 0.5
    plt.barh(pos, vals, align="center", color=colors, linewidth="1", edgecolor="black")
    plt.yticks(pos, names, fontsize=fontsize)
    plt.xlabel(f"Class {target_class}", fontsize=fontsize)
    plt.title(f"SHAP value\n")  # Predicted={pred} True_class={true}
    return fig


# https://github.com/slundberg/shap/issues/397
def sumCategories(vals, cols_names, categorical_features, matching_instance):
    vals = list(vals)
    d = {
        ("_".join(k.split("_")[0:-1]), k.split("_")[-1])
        if "_".join(k.split("_")[0:-1]) in categorical_features
        else (k, k): v
        for k, v in zip(cols_names, vals)
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


# TODO fixare mapping
def convertInstance(
    instance, categorical_features, continuos_features, min_max_scaler=None
):
    instance_dict = dict(instance.items())
    matching_instance = {
        "_".join(k.split("_")[0:-1]): k.split("_")[-1]
        for k, v in instance_dict.items()
        if "_".join(k.split("_")[0:-1]) in categorical_features and v == 1
    }
    # matching_instance.update(
    #     {k: v for k, v in instance.items() if k in continuos_features}
    # )
    if continuos_features and min_max_scaler:
        scaled_values = min_max_scaler.inverse_transform(
            instance[continuos_features].values.reshape(1, -1)
        )[0]
        for i, k in enumerate(continuos_features):
            matching_instance[k] = scaled_values[i]
    else:
        matching_instance.update(
            {k: v for k, v in instance.items() if k in continuos_features}
        )
    return matching_instance


# TODO fixare mapping
def convertInstanceold(
    instance, categorical_features, continuos_features, min_max_scaler=None
):
    instance_dict = dict(instance.items())
    matching_instance = {
        "_".join(k.split("_")[0:-1]): k.split("_")[-1]
        for k, v in instance_dict.items()
        if "_".join(k.split("_")[0:-1]) in categorical_features and v == 1
    }
    matching_instance.update(
        {k: v for k, v in instance.items() if k in continuos_features}
    )

    return matching_instance


def plot_lime_explanation(
    lime_explanation,
    label=1,
    pred=None,
    true_label=None,
    fontsize=14,
    sortedF=False,
    methodName=False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    exp = dict(lime_explanation.as_list(label=label))
    fig = plt.figure()
    # vals = [x[1] for x in exp]
    # names = [x[0] for x in exp]
    # Interpretable domain (e.g. discretize + order output)
    exp_sorted = {}
    if sortedF:
        names = exp.keys()
        vals = [exp[n] for n in names]
    else:
        names = lime_explanation.domain_mapper.feature_names
        # vals = []
        for n in names:
            if n in exp:
                # remove
                # vals.append(exp[n])

                exp_sorted[n] = exp[n]
            else:
                for k in exp.keys():
                    if n in k:
                        # remove
                        # vals.append(exp[k])

                        exp_sorted[k] = exp[k]
        # Interpretable domain (e.g. discretize + order output)
        names = list(exp_sorted.keys())
        vals = list(exp_sorted.values())
        # vals = [exp[n] for n in names]

    vals = vals[::-1]
    names = names[::-1]
    colors = ["#fa8023" if x > 0 else "#3574b2" for x in vals]
    pos = np.arange(len(exp)) + 0.5
    plt.barh(pos, vals, align="center", color=colors, linewidth="1", edgecolor="black")
    plt.yticks(pos, names, fontsize=fontsize)
    plt.xlabel(f"Class {lime_explanation.class_names[label]}", fontsize=fontsize)
    methodName = "" if methodName is None else "LIME"
    if lime_explanation.mode == "classification":
        title = f"{methodName}\n Local explanation for class {lime_explanation.class_names[label]}"  # \n predicted={pred}  true class={true_label}"
    else:
        title = "Local explanation"
    plt.title(title, fontsize=fontsize)
    return fig


def dictAnchor(instance_id, anchor_explanation, pred_class=None):
    return {
        "instance_id": instance_id,
        "anchor": " AND ".join(anchor_explanation.names()),
        "predicted_class": anchor_explanation.exp_map["prediction"]
        if pred_class is None
        else pred_class,
        "precision": anchor_explanation.precision(),
        "coverage": anchor_explanation.coverage(),
    }


def printAnchor(instance_id, anchor_explanation, pred_class=None):
    print("Id", instance_id)
    print("Anchor: %s" % (" AND ".join(anchor_explanation.names())))
    pred_class = (
        anchor_explanation.exp_map["prediction"] if pred_class is None else pred_class
    )
    print(f"Predicted class:  {pred_class}")
    print("Precision: %.2f" % anchor_explanation.precision())
    print("Coverage: %.2f" % anchor_explanation.coverage())


def showHits(
    hits_dict,
    upperLabel=None,
    saveFig=False,
    nameFig="hits_partial_hits",
    outDir="./",
    ylabel="",
    max_y_tick=10,
    step=1,
    percentage=False,
):
    import numpy as np
    import matplotlib.pyplot as plt

    labelsize = 9.4

    hits = [hits_dict[k]["hit"] for k in hits_dict]
    partial_hits_superset = [hits_dict[k]["partial_hit_superset"] for k in hits_dict]
    partial_hits_subset = [hits_dict[k]["partial_hit_subset"] for k in hits_dict]

    if percentage and upperLabel is not None:

        def computePercentage(values, tot):
            return [100 * (v / tot) for v in values]

        hits = computePercentage(hits, upperLabel)
        partial_hits_superset = computePercentage(partial_hits_superset, upperLabel)
        partial_hits_subset = computePercentage(partial_hits_subset, upperLabel)

    ind = np.arange(len(hits_dict))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, hits, width, color="brown")

    p2 = plt.bar(
        ind, partial_hits_superset, width, bottom=hits, color="lightcoral"
    )  # , hatch='///'
    hit_super_set = list(np.asarray(hits) + np.asarray(partial_hits_superset))
    p3 = plt.bar(
        ind, partial_hits_subset, width, bottom=hit_super_set, color="indianred"
    )  # , hatch='///'
    if upperLabel:
        if percentage:
            eps = 0.01
            upperLabel = 100 + eps
        plt.axhline(y=upperLabel, color="grey", linestyle="--")
    plt.ylabel(ylabel)
    plt.title("Hits and partial hits", fontsize=labelsize)
    plt.xticks(ind, [k for k in hits_dict], fontsize=labelsize)

    plt.legend(
        (p1[0], p2[0], p3[0]),
        ("Hits", "Partial hits - subset", "Partial hits - superset"),
        fontsize=labelsize,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = (4.4, 1.9), 100

    if saveFig:
        plt.savefig(f"./{outDir}/{nameFig}.pdf", bbox_inches="tight")
    plt.show()


def computeHits(attributes_rules, target, hits, verbose=False):
    hit_rule_tuple = None
    superset = False
    subset = False
    for rule_attr in attributes_rules:
        if verbose:
            print(rule_attr)
        # the order does not matter
        if set(rule_attr) == set(target):
            hits["hit"] += 1
            hit_rule_tuple = (frozenset(rule_attr), "hit")
            if verbose:
                print("hit", hit_rule_tuple)
            return hits, hit_rule_tuple
        elif frozenset(rule_attr).issuperset(frozenset(target)):
            superset = True
            hit_rule_tuple = (frozenset(rule_attr), "partial_hit_superset")
            if verbose:
                print("partial_hit_superset")
            continue
        else:
            sunbs = []
            for attribute in target:
                if attribute in rule_attr:
                    subset = True
                    sunbs.append(attribute)
            if subset:
                hit_rule_tuple = (frozenset(rule_attr), "partial_hit_subset")
                if verbose:
                    print("partial_hit_subset", sunbs)
                continue
    if superset:
        hits["partial_hit_superset"] += 1
    elif subset:
        hits["partial_hit_subset"] += 1
    return hits, hit_rule_tuple


def checkRankingTarget(rankings, target):
    targets_hit = [1 for t in target if rankings[t] <= len(target)]
    return sum(targets_hit) / len(target)


def getFeatureRankingList(k_v, feature_names, absValue=False):
    rankings = k_v.copy()
    if absValue:
        for rank, k in enumerate(
            sorted(k_v, key=lambda k: abs(k_v[k]), reverse=True), 1
        ):
            rankings[k] = rank
    else:
        for rank, k in enumerate(sorted(k_v, key=k_v.get, reverse=True), 1):
            rankings[k] = rank
    rankings = {
        feature_names[f]: list(rankings.values())[f] for f in range(len(feature_names))
    }
    return rankings


def evaluateRankingTarget(k_v, feature_names, target, absValue=False):
    rankings = getFeatureRankingList(k_v, feature_names, absValue=absValue)
    return checkRankingTarget(rankings, target)


def lime_explanation_attr_sorted(exp_lime, label):
    exp = dict(exp_lime.as_list(label=label))
    names = exp_lime.domain_mapper.feature_names
    exp_lime_dict = {}
    for n in names:
        if n in exp:
            exp_lime_dict[n] = exp[n]
        else:
            for k in exp.keys():
                if n in k:
                    exp_lime_dict[n] = exp[k]
    return exp_lime_dict