#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", None)

toCompute = True

classifier_name = "tree"


# # Import dataset
dataset_name = "compas"
for depth in [2, 3, 4, 5, 6]:
    results_output = f"results/results_tree/general/max_depth_{depth}"

    outputDirResults = f"./{results_output}/{dataset_name}/{classifier_name}/"
    saveResults = True

    if saveResults:
        from src.utils import saveJson
        from pathlib import Path

        Path(outputDirResults).mkdir(parents=True, exist_ok=True)

    from src.import_datasets import import_process_compas

    risk_class_type = False
    dfI = import_process_compas(discretize=False, risk_class=risk_class_type)
    dfI.reset_index(drop=True, inplace=True)
    dfI["class"].replace({0: "Not recidivate", 1: "Recidivate"}, inplace=True)
    df = dfI.copy()
    df.head()

    from sklearn import model_selection
    import numpy as np

    np.random.seed(42)
    df_train, df_test = model_selection.train_test_split(df, train_size=0.7)

    # # Process data
    from ProcessedDataset_v2 import ProcessedDatasetTrainTest

    pc = ProcessedDatasetTrainTest(df_train, df_test)

    from sklearn.tree import DecisionTreeClassifier

    clf_init = DecisionTreeClassifier(random_state=7, max_depth=depth)
    pc.processTrainTestDataset(clf_init)  # , dataset_name=dataset_name)

    # In[10]:

    predicted = np.argmax(pc.predict_fn(pc.test.values), axis=1)
    FP = [i for i, p in enumerate(predicted) if p == 1 and pc.labels_test[i] == 0]
    FN = [i for i, p in enumerate(predicted) if p == 0 and pc.labels_test[i] == 1]
    mispredicted = [i for i, p in enumerate(predicted) if p != pc.labels_test[i]]
    correct_prediction = [i for i, p in enumerate(predicted) if p == pc.labels_test[i]]

    # In[11]:

    from sklearn.metrics import accuracy_score

    print("Accuracy: ", round(accuracy_score(pc.labels_test, predicted), 4))

    # In[12]:

    from sklearn import tree

    text_representation = tree.export_text(pc.clf)
    print(text_representation)

    # In[13]:

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(
        pc.clf,
        feature_names=pc.OH_X_test_cols.columns,
        class_names=pc.d_train.class_values(),
        filled=True,
    )

    # In[14]:

    pc.continuos_features  # .index("priors_count")

    # In[15]:

    def decodeValue(instance, column_name, value, pc, threshold_v=False):
        fr = frozenset([f"{column_name}=value"])
        fr_col = frozenset([column_name])
        if column_name in pc.min_max_scaler_cols:
            index = pc.min_max_scaler_cols.index(column_name)
            convert = np.zeros((1, len(pc.min_max_scaler_cols)))
            convert[0, index] = value
            threshold_value = pc.min_max_scaler.inverse_transform(convert)[0, index]
            fr = frozenset([f"{column_name}={threshold_value.round(2)}"])
            return threshold_value.round(2), fr, (fr, fr_col)
        if column_name in pc.getMappingOHColumnsTupleAttrValue():
            attr, attrv = pc.getMappingOHColumnsTupleAttrValue()[column_name]
            if threshold_v == False:
                conv_v = False if value < 0.5 else True
                rel = "!=" if value < 0.5 else "="
                attr_values = dict(pc.getLabelEncodedNames())[attr]
                if len(attr_values) == 2:
                    index = attr_values.index(attrv)
                    index = (index + 1) % 2
                    rel, attrv = "=", attr_values[index]
                df_sel = instance[
                    [col for col in instance if col.startswith(attr)]
                ].astype(bool)
                attrv_equal = pc.getMappingOHColumnsTupleAttrValue()[
                    df_sel.columns[df_sel.values[0]].values[0]
                ][1]
                fr = frozenset([f"{attr}{rel}{attrv}"])
                fr_instance = (frozenset([f"{attr}={attrv_equal}"]), frozenset([attr]))
                return conv_v, fr, fr_instance
        return value, fr, (fr, fr_col)

    # In[16]:

    def getPathDecisionTree(clf, instance, pc, verbose=False):
        path_fr = frozenset([])
        path_fr_instance = frozenset([])
        path_fr_attr = frozenset([])
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold

        X_test = instance.values
        node_indicator = clf.decision_path(X_test)
        leaf_id = clf.apply(X_test)

        sample_id = 0
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            threshold_value, _, _ = decodeValue(
                instance,
                instance.columns[feature[node_id]],
                threshold[node_id],
                pc,
                threshold_v=True,
            )
            column_value, fr, fr_instance_att = decodeValue(
                instance,
                instance.columns[feature[node_id]],
                X_test[sample_id, feature[node_id]],
                pc,
            )
            path_fr = path_fr.union(fr)
            path_fr_instance = path_fr_instance.union(fr_instance_att[0])
            path_fr_attr = path_fr_attr.union(fr_instance_att[1])
            if verbose:
                print(
                    "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
                    "{inequality} {threshold})".format(
                        node=node_id,
                        sample=sample_id,
                        feature=feature[node_id],
                        value=X_test[sample_id, feature[node_id]],
                        inequality=threshold_sign,
                        threshold=threshold[node_id],
                    )
                )

                print(
                    "decision node {node} : ({column_name} = {value}) "
                    "{inequality} {threshold})".format(
                        node=node_id,
                        column_name=instance.columns[feature[node_id]],
                        value=column_value,
                        inequality=threshold_sign,
                        threshold=threshold_value,
                    )
                )
        return path_fr, path_fr_instance, path_fr_attr

    # In[17]:

    sample_id = 2

    instance = pc.OH_X_test_cols.iloc[sample_id : sample_id + 1]
    # display(instance)

    clf = pc.clf
    path_fr, path_fr_instance, path_fr_attr = getPathDecisionTree(
        clf, instance, pc, verbose=False
    )

    print(path_fr)
    print()
    print(path_fr_instance)
    print()
    print(path_fr_attr)

    # # Explainers

    # ## LIME

    # In[18]:

    import lime
    import lime.lime_tabular
    import numpy as np

    np.random.seed(42)

    # In[19]:

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        pc.train.values,
        feature_names=pc.feature_names,
        class_names=pc.class_names,
        categorical_features=pc.categorical_features_pos,
        categorical_names=pc.categorical_names_LE,
        random_state=42,
    )

    # ## SHAP

    # In[20]:

    if toCompute:
        import shap

        if classifier_name in ["RF", "tree"]:
            shap_explainer = shap.TreeExplainer(pc.clf)
        else:
            shap_explainer = shap.KernelExplainer(
                model=pc.predict_fn_OH, data=pc.OH_X_train.values
            )  # , link="logit")

    # ## Anchor

    # In[21]:

    from anchor import utils
    from anchor import anchor_tabular

    # In[22]:

    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        pc.class_names, pc.feature_names, pc.train.values, pc.categorical_names_LE
    )

    # ## LACE

    # In[23]:

    from src.LACE_explainer import LACE_explainer

    lace_explainer = LACE_explainer(pc.d_train, pc.predict_fn, dataset_name="COMPAS")

    # # Comparison

    # In[24]:

    outdir_lace = f"./results/{dataset_name}/lace"
    outdir_lime = f"./results/{dataset_name}/lime"
    outdir_shap = f"./results/{dataset_name}/shap"

    saveFig = False

    if saveFig:
        from pathlib import Path

        Path(outdir_lace).mkdir(parents=True, exist_ok=True)
        Path(outdir_lime).mkdir(parents=True, exist_ok=True)
        Path(outdir_shap).mkdir(parents=True, exist_ok=True)

    # In[25]:

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
            feature_names[f]: list(rankings.values())[f]
            for f in range(len(feature_names))
        }
        return rankings

    def evaluateRankingTarget(k_v, feature_names, target, absValue=False):
        rankings = getFeatureRankingList(k_v, feature_names, absValue=absValue)
        return checkRankingTarget(rankings, target)

    # In[26]:

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
                break
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
                    break
        if superset:
            hits["partial_hit_superset"] += 1
        elif subset:
            hits["partial_hit_subset"] += 1
        return hits, hit_rule_tuple

    # In[27]:

    def saveFigure(fig_lace, outdir, id_i, explainer_name="", w=4, h=3):
        fig_lace.set_size_inches(w, h)
        fig_lace.savefig(f"{outdir}/{explainer_name}_{id_i}.pdf", bbox_inches="tight")

    # In[28]:

    saveFig = False
    fontsize = 14

    # In[29]:

    # target=["X", "Y"]
    lime_targets = {}
    shap_targets = {}
    lace_targets = {}

    # In[30]:

    for id_i in [8]:
        path, true_rule, target = getPathDecisionTree(
            pc.clf, pc.OH_X_test_cols.iloc[id_i : id_i + 1], pc, verbose=False
        )
        print("####################\ntarget")
        print(target)
        print(path)
        print(true_rule)
        from utils_RW import plot_lime_explanation

        predicted_class = pc.predict_fn_class(pc.test.iloc[id_i].values.reshape(1, -1))[
            0
        ]
        print("LIME")
        exp_lime = lime_explainer.explain_instance(
            pc.test.iloc[id_i].values,
            pc.predict_fn,
            num_features=len(pc.feature_names),
            labels=[predicted_class],
        )
        fig_lime = plot_lime_explanation(
            exp_lime,
            label=predicted_class,
            pred=pc.class_names[
                np.argmax(pc.predict_fn(pc.test.iloc[id_i : id_i + 1].values))
            ],
            true_label=pc.class_names[pc.labels_test[id_i]],
            fontsize=fontsize,
        )
        if saveFig:
            saveFigure(fig_lime, outputDirResults, id_i, "lime")
        exp_lime_dict = lime_explanation_attr_sorted(exp_lime, predicted_class)
        lime_targets[id_i] = evaluateRankingTarget(
            exp_lime_dict, list(pc.feature_names), target
        )

        from utils_RW import sumCategories, plotShapValues, convertInstance

        print("SHAP")
        instance = pc.OH_X_test_cols.iloc[id_i]
        shap_values = shap_explainer.shap_values(instance)
        matching_instance = convertInstance(
            instance, pc.categorical_features, pc.continuos_features
        )

        sum_shap_for_categories = sumCategories(
            shap_values[predicted_class],
            pc.oh_columns,
            pc.categorical_features,
            matching_instance,
        )
        fig_shap = plotShapValues(
            list(sum_shap_for_categories.values()),
            list(sum_shap_for_categories.keys()),
            target_class=pc.class_names[predicted_class],
            pred=pc.class_names[pc.clf.predict([instance])[0]],
            true=pc.class_names[pc.labels_test[id_i]],
            fontsize=fontsize,
        )
        if saveFig:
            saveFigure(fig_shap, outputDirResults, id_i, "shap")
        shap_targets[id_i] = evaluateRankingTarget(
            sum_shap_for_categories, list(pc.feature_names), target
        )

        from utils_RW import printAnchor, dictAnchor

        exp_anchor = anchor_explainer.explain_instance(
            pc.test.iloc[id_i].values, pc.predict_fn_class, threshold=0.95
        )
        printAnchor(id_i, exp_anchor)
        dictAnchor(id_i, exp_anchor)
        featureMasking = True
        instance = pc.d_explain[id_i]
        infos = {"model": "RF"}
        instance_discretized = pc.d_explain.getDiscretizedInstance(id_i)
        explanation_fm = lace_explainer.explain_instance(
            instance,
            pc.class_names[predicted_class],
            featureMasking=featureMasking,
            discretizedInstance=instance_discretized,
            verbose=False,
        )
        fig_lace = explanation_fm.plotExplanation(
            showRuleKey=True, retFig=True, fontsize=fontsize
        )
        explanation_fm.local_rules.printLocalRules()
        if saveFig:
            saveFigure(fig_lace, outputDirResults, id_i, "lace")
        prediction_difference_attr = (
            explanation_fm.getPredictionDifferenceDict()
        )  # {attrs_values[i]:explanation_fm.diff_single[i] for i in range(0,len(explanation_fm.diff_single))}
        lace_targets[id_i] = evaluateRankingTarget(
            prediction_difference_attr, list(pc.feature_names), target
        )

        changes = explanation_fm.estimateSingleAttributeChangePrediction()
        if changes:
            print(changes)

    # # Evaluation

    # In[31]:

    showExplanation = False
    n_explanations = 100

    # ## Ranking

    # In[32]:

    lime_targets = {}
    shap_targets = {}
    lace_targets = {}
    lime_targets_abs = {}
    shap_targets_abs = {}
    lace_targets_abs = {}

    # In[33]:

    lace_hits = {"hit": 0, "partial_hit_superset": 0, "partial_hit_subset": 0}
    anchor_hits = {"hit": 0, "partial_hit_superset": 0, "partial_hit_subset": 0}
    lace_rules_hit_all = []
    anchor_rules_hit_all = []

    # In[34]:

    for id_i in range(0, n_explanations):
        path, true_rule, target = getPathDecisionTree(
            pc.clf, pc.OH_X_test_cols.iloc[id_i : id_i + 1], pc, verbose=False
        )

        end_v = "\n" if id_i % 9 == 0 else " "
        print(f"{id_i} ({len(target)})", end=end_v)
        from utils_RW import plot_lime_explanation

        predicted_class = pc.predict_fn_class(pc.test.iloc[id_i].values.reshape(1, -1))[
            0
        ]

        exp_lime = lime_explainer.explain_instance(
            pc.test.iloc[id_i].values,
            pc.predict_fn,
            num_features=len(pc.feature_names),
            labels=[predicted_class],
        )

        if showExplanation:
            print("LIME")
            fig_lime = plot_lime_explanation(
                exp_lime,
                label=predicted_class,
                pred=pc.class_names[
                    np.argmax(pc.predict_fn(pc.test.iloc[id_i : id_i + 1].values))
                ],
                true_label=pc.class_names[pc.labels_test[id_i]],
            )
            if saveFig:
                saveFigure(fig_lime, outdir_lime, id_i, "lime")
        exp_lime_dict = lime_explanation_attr_sorted(exp_lime, predicted_class)
        lime_targets[id_i] = evaluateRankingTarget(
            exp_lime_dict, list(pc.feature_names), target
        )
        lime_targets_abs[id_i] = evaluateRankingTarget(
            exp_lime_dict, list(pc.feature_names), target, absValue=True
        )

        from utils_RW import sumCategories, plotShapValues, convertInstance

        instance = pc.OH_X_test_cols.iloc[id_i]
        shap_values = shap_explainer.shap_values(instance)
        matching_instance = convertInstance(
            instance, pc.categorical_features, pc.continuos_features
        )

        sum_shap_for_categories = sumCategories(
            shap_values[predicted_class],
            pc.oh_columns,
            pc.categorical_features,
            matching_instance,
        )
        if showExplanation:
            print("SHAP")
            fig_shap = plotShapValues(
                list(sum_shap_for_categories.values()),
                list(sum_shap_for_categories.keys()),
                target_class=pc.class_names[predicted_class],
                pred=pc.class_names[pc.clf.predict([instance])[0]],
                true=pc.class_names[pc.labels_test[id_i]],
            )
            if saveFig:
                saveFigure(fig_shap, outdir_shap, id_i, "shap")
        shap_targets[id_i] = evaluateRankingTarget(
            sum_shap_for_categories, list(pc.feature_names), target
        )
        shap_targets_abs[id_i] = evaluateRankingTarget(
            sum_shap_for_categories, list(pc.feature_names), target, absValue=True
        )

        from utils_RW import printAnchor, dictAnchor

        exp_anchor = anchor_explainer.explain_instance(
            pc.test.iloc[id_i].values, pc.predict_fn_class, threshold=0.95
        )

        if showExplanation:
            printAnchor(id_i, exp_anchor)
            dictAnchor(id_i, exp_anchor)
        attributes_anchor_rules = [
            list(pc.feature_names)[attr] for attr in exp_anchor.features()
        ]
        anchor_hits, anchor_rules_hit = computeHits(
            [attributes_anchor_rules], target, anchor_hits, verbose=False
        )
        anchor_rules_hit_all.append(anchor_rules_hit)

        featureMasking = True
        instance = pc.d_explain[id_i]
        infos = {"model": "RF"}
        instance_discretized = pc.d_explain.getDiscretizedInstance(id_i)
        explanation_fm = lace_explainer.explain_instance(
            instance,
            pc.class_names[predicted_class],
            featureMasking=featureMasking,
            discretizedInstance=instance_discretized,
            specialistic_rules=False,
            verbose=False,
        )
        if showExplanation:
            fig_lace = explanation_fm.plotExplanation(showRuleKey=False, retFig=True)
            # explanation_fm.local_rules.printLocalRules()
            if saveFig:
                saveFigure(fig_lace, outdir_lace, id_i, "lace")
            changes = explanation_fm.estimateSingleAttributeChangePrediction()
            if changes:
                print(changes)
        prediction_difference_attr = (
            explanation_fm.getPredictionDifferenceDict()
        )  # {attrs_values[i]:explanation_fm.diff_single[i] for i in range(0,len(explanation_fm.diff_single))}
        lace_targets[id_i] = evaluateRankingTarget(
            prediction_difference_attr, list(pc.feature_names), target
        )
        lace_targets_abs[id_i] = evaluateRankingTarget(
            prediction_difference_attr, list(pc.feature_names), target, absValue=True
        )
        attributes_lace_rules = explanation_fm.getAttributesRules()
        lace_hits, lace_rules_hit = computeHits(
            attributes_lace_rules, target, lace_hits, verbose=False
        )
        lace_rules_hit_all.append(lace_rules_hit)
    print()

    # ### Ranking

    # In[35]:

    print("LIME")
    print(lime_targets)
    print("SHAP")
    print(shap_targets)
    print("LACE")
    print(lace_targets)

    # In[36]:

    print("LIME", np.mean(list(lime_targets.values())))
    print("SHAP", np.mean(list(shap_targets.values())))
    print("LACE", np.mean(list(lace_targets.values())))

    # ### Rule hits

    # In[37]:

    print("LACE")
    print(lace_hits)
    print("Anchor")
    print(anchor_hits)

    # In[38]:

    anchor_rules_hit_all

    # In[39]:

    if saveResults:
        saveJson(lace_hits, "lace_hits", outputDirResults)
        saveJson(anchor_hits, "anchor_hits", outputDirResults)
        saveJson(lime_targets, "lime_targets", outputDirResults)
        saveJson(shap_targets, "shap_targets", outputDirResults)
        saveJson(lace_hits, "lace_targets", outputDirResults)

        mean_hit_targets = {
            "LIME": np.mean(list(lime_targets.values())),
            "SHAP": np.mean(list(shap_targets.values())),
            "LACE": np.mean(list(lace_targets.values())),
        }

        saveJson(mean_hit_targets, "mean_hit_targets", outputDirResults)

    # ### Ranking abs

    # In[40]:

    print("LIME abs")
    print(lime_targets_abs)
    print("SHAP abs")
    print(shap_targets_abs)
    print("LACE abs")
    print(lace_targets_abs)

    # In[41]:

    print("LIME abs", np.mean(list(lime_targets_abs.values())))
    print("SHAP abs", np.mean(list(shap_targets_abs.values())))
    print("LACE abs", np.mean(list(lace_targets_abs.values())))

    # In[42]:

    if saveResults:
        saveJson(lime_targets_abs, "lime_targets_abs", outputDirResults)
        saveJson(shap_targets_abs, "shap_targets_abs", outputDirResults)
        saveJson(lace_targets_abs, "lace_targets_abs", outputDirResults)

        mean_hit_targets_abs = {
            "LIME": np.mean(list(lime_targets_abs.values())),
            "SHAP": np.mean(list(shap_targets_abs.values())),
            "LACE": np.mean(list(lace_targets_abs.values())),
        }

        saveJson(mean_hit_targets_abs, "mean_hit_targets_abs", outputDirResults)

    # ## Rule hit

    # In[43]:

    from utils_RW import showHits

    # In[44]:

    print(anchor_hits)
    print(lace_hits)
    hits_dict_summary = {}
    hits_dict_summary["LACE"] = lace_hits
    hits_dict_summary["anchor"] = anchor_hits

    # In[45]:

    showHits(
        hits_dict_summary,
        upperLabel=n_explanations,
        percentage=True,
        saveFig=saveResults,
        outDir=outputDirResults,
    )
