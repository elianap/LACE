# import numpy as np


def feature_importance_similarity(a, b, metric="cosine"):
    from scipy.spatial.distance import cdist

    val = 1.0 - cdist(a.reshape(1, -1), b.reshape(1, -1), metric=metric)[0][0]
    val = max(0.0, min(val, 1.0))
    return val

def explanation_precision_recall_f1_score(a,b):
    e_precision=explanation_precision(a, b)
    e_recall=explanation_recall(a, b)
    e_f1_score=rule_based_similarity(a, b)
    return e_precision, e_recall, e_f1_score

def explanation_precision(a, b):
    from sklearn.metrics import precision_score
    return precision_score(a, b, pos_label=1, average='binary')


def explanation_recall(a, b):
    from sklearn.metrics import recall_score
    return recall_score(a, b, pos_label=1, average='binary')


def rule_based_similarity(a, b):
    from sklearn.metrics import f1_score

    return f1_score(a, b, pos_label=1, average="binary")


def oh_encoding_rule(rule, feature_names):
    return [1 if f in rule else 0 for f in list(feature_names)]


def feature_hit_oh_encoding(feature_importance_values):
    return [1 if round(v, 4) != 0.0 else 0 for v in feature_importance_values]


def feature_hit_oh_encoding_delta(feature_importance_values, delta_p=5 / 100):
    delta = delta_p * abs(max(feature_importance_values))
    return [1 if abs(v) > delta else 0 for v in feature_importance_values]


def computeAverage(explainer_results):
    import numpy as np

    return {
        explainer: np.mean(list(result_values.values()))
        for explainer, result_values in explainer_results.items()
    }


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
    if superset:
        hits["partial_hit_superset"] += 1
    elif subset:
        hits["partial_hit_subset"] += 1
    return hits, hit_rule_tuple
