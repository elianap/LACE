#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", None)

toCompute = True

classifier_name = "tree"

verbose=False

# # Import dataset
dataset_name = "compas"
for depth in [2, 3, 4, 5, 6]: 
    results_output = f"results/quality_evaluation_new/results_tree_new/max_depth_{depth}"

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

    predicted = np.argmax(pc.predict_fn(pc.test.values), axis=1)
    FP = [i for i, p in enumerate(predicted) if p == 1 and pc.labels_test[i] == 0]
    FN = [i for i, p in enumerate(predicted) if p == 0 and pc.labels_test[i] == 1]
    mispredicted = [i for i, p in enumerate(predicted) if p != pc.labels_test[i]]
    correct_prediction = [i for i, p in enumerate(predicted) if p == pc.labels_test[i]]

    from sklearn.metrics import accuracy_score

    print("Accuracy: ", round(accuracy_score(pc.labels_test, predicted), 4))

    from sklearn import tree

    text_representation = tree.export_text(pc.clf)
    if verbose:
        print(text_representation)

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(
        pc.clf,
        feature_names=pc.OH_X_test_cols.columns,
        class_names=pc.d_train.class_values(),
        filled=True,
    )

    pc.continuos_features  # .index("priors_count")

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

    sample_id = 2

    instance = pc.OH_X_test_cols.iloc[sample_id : sample_id + 1]
    # display(instance)

    clf = pc.clf
    path_fr, path_fr_instance, path_fr_attr = getPathDecisionTree(
        clf, instance, pc, verbose=False
    )

    if verbose:
        print(path_fr)
        print()
        print(path_fr_instance)
        print()
        print(path_fr_attr)

    # # Explainers

    # ## LIME

    import lime
    import lime.lime_tabular
    import numpy as np

    np.random.seed(42)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        pc.train.values,
        feature_names=pc.feature_names,
        class_names=pc.class_names,
        categorical_features=pc.categorical_features_pos,
        categorical_names=pc.categorical_names_LE,
        random_state=42,
    )

    # ## SHAP

    if toCompute:
        import shap

        if classifier_name in ["RF", "tree"]:
            shap_explainer = shap.TreeExplainer(pc.clf)
        else:
            shap_explainer = shap.KernelExplainer(
                model=pc.predict_fn_OH, data=pc.OH_X_train.values
            )  # , link="logit")

    # ## Anchor

    from anchor import utils
    from anchor import anchor_tabular

    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        pc.class_names, pc.feature_names, pc.train.values, pc.categorical_names_LE
    )

    # ## LACE

    from src.LACE_explainer import LACE_explainer

    lace_explainer = LACE_explainer(pc.d_train, pc.predict_fn, dataset_name="COMPAS")

    # # Comparison

    outdir_lace = f"./results/{dataset_name}/lace"
    outdir_lime = f"./results/{dataset_name}/lime"
    outdir_shap = f"./results/{dataset_name}/shap"

    saveFig = False

    if saveFig:
        from pathlib import Path

        Path(outdir_lace).mkdir(parents=True, exist_ok=True)
        Path(outdir_lime).mkdir(parents=True, exist_ok=True)
        Path(outdir_shap).mkdir(parents=True, exist_ok=True)

    from quality_evaluation import (
        evaluateRankingTarget,
        feature_hit_oh_encoding,
        feature_hit_oh_encoding_delta,
        lime_explanation_attr_sorted,
        feature_importance_similarity,
        rule_based_similarity,
        computeHits,
        oh_encoding_rule,
        computeAverage,
        explanation_precision_recall_f1_score, 
        explanation_precision, explanation_recall
    )

    def saveFigure(fig_lace, outdir, id_i, explainer_name="", w=4, h=3):
        fig_lace.set_size_inches(w, h)
        fig_lace.savefig(f"{outdir}/{explainer_name}_{id_i}.pdf", bbox_inches="tight")

    showExplanation = False
    n_explanations = 100

    # ## Ranking

    lime_targets = {}
    shap_targets = {}
    lace_targets = {}
    lime_targets_abs = {}
    shap_targets_abs = {}
    lace_targets_abs = {}

    lace_hits = {"hit": 0, "partial_hit_superset": 0, "partial_hit_subset": 0}
    anchor_hits = {"hit": 0, "partial_hit_superset": 0, "partial_hit_subset": 0}
    lace_rules_hit_all = []
    anchor_rules_hit_all = []

    fi_explainers = ["LIME", "SHAP", "LACE"]
    feature_cosine = {e: {} for e in fi_explainers}
    feature_f1_score = {e: {} for e in fi_explainers}
    feature_f1_score_delta = {e: {} for e in fi_explainers}
    feature_recall = {e: {} for e in fi_explainers}
    feature_precision = {e: {} for e in fi_explainers}
    rule_explainers = ["anchor", "LACE"]
    rule_f1_score = {e: {} for e in rule_explainers}
    rule_precision = {e: {} for e in rule_explainers}
    rule_recall = {e: {} for e in rule_explainers}

    for id_i in range(0, n_explanations):
        path, true_rule, target = getPathDecisionTree(
            pc.clf, pc.OH_X_test_cols.iloc[id_i : id_i + 1], pc, verbose=False
        )
        target_vector = np.asarray(
            [1.0 / float(len(target)) if i in target else 0 for i in pc.feature_names]
        )
        print(target_vector)
        true_rule_oh = [1 if f in target else 0 for f in list(pc.feature_names)]

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

        expl_lime_fi = np.asarray((list(exp_lime_dict.values())))
        feature_cosine["LIME"][id_i] = feature_importance_similarity(
                target_vector, expl_lime_fi
            )

        expl_lime_feature_vector=np.asarray(feature_hit_oh_encoding(exp_lime_dict.values()))

        feature_precision["LIME"][id_i], feature_recall["LIME"][id_i], feature_f1_score["LIME"][id_i], = explanation_precision_recall_f1_score(
            np.asarray(true_rule_oh), expl_lime_feature_vector
        )
        
        feature_f1_score_delta["LIME"][id_i] = rule_based_similarity(
            np.asarray(true_rule_oh),
            np.asarray(feature_hit_oh_encoding_delta(exp_lime_dict.values())),
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
        expl_shap_fi = np.asarray((list(sum_shap_for_categories.values())))
        
        feature_cosine["SHAP"][id_i] = feature_importance_similarity(
                target_vector, expl_shap_fi
        )

        feature_f1_score["SHAP"][id_i] = rule_based_similarity(
            np.asarray(true_rule_oh),
            np.asarray(feature_hit_oh_encoding(sum_shap_for_categories.values())),
        )
        feature_f1_score_delta["SHAP"][id_i] = rule_based_similarity(
            np.asarray(true_rule_oh),
            np.asarray(feature_hit_oh_encoding_delta(sum_shap_for_categories.values())),
        )

        expl_shap_feature_vector=np.asarray(feature_hit_oh_encoding(sum_shap_for_categories.values()))

        feature_precision["SHAP"][id_i], feature_recall["SHAP"][id_i], feature_f1_score["SHAP"][id_i], = explanation_precision_recall_f1_score(
                np.asarray(true_rule_oh), expl_shap_feature_vector
        )

        from utils_RW import printAnchor, dictAnchor

        exp_anchor = anchor_explainer.explain_instance(
            pc.test.iloc[id_i].values, pc.predict_fn_class, threshold=0.95
        )

        if showExplanation:
            printAnchor(id_i, exp_anchor)
            dictAnchor(id_i, exp_anchor)
        attributes_anchor_rules = [[
            list(pc.feature_names)[attr] for attr in exp_anchor.features()
        ]]
        anchor_hits, anchor_rules_hit = computeHits(
            attributes_anchor_rules, target, anchor_hits, verbose=False
        )
        anchor_rules_hit_all.append(anchor_rules_hit)

        rules_anchor_oh = [
            oh_encoding_rule(rule, pc.feature_names) for rule in attributes_anchor_rules
        ]
        if rules_anchor_oh ==[]:
            rules_anchor_oh = [oh_encoding_rule([], pc.feature_names)]

        rule_f1_score["anchor"][id_i] = max(
            [
                rule_based_similarity(np.asarray(r), np.asarray(true_rule_oh))
                for r in rules_anchor_oh
            ]
        )
        rule_recall["anchor"][id_i] = max(
            [
                explanation_recall(np.asarray(r), np.asarray(true_rule_oh))
                for r in rules_anchor_oh
            ]
        )

        rule_precision["anchor"][id_i] = max(
            [
                explanation_precision(np.asarray(r), np.asarray(true_rule_oh))
                for r in rules_anchor_oh
            ]
        )
        featureMasking = True
        instance = pc.d_explain[id_i]
        infos = {"model": classifier_name}
        instance_discretized = pc.d_explain.getDiscretizedInstance(id_i)
        explanation_fm = lace_explainer.explain_instance(
            instance,
            pc.class_names[predicted_class],
            featureMasking=featureMasking,
            discretizedInstance=instance_discretized,
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

        expl_lace_fi = np.asarray((list(prediction_difference_attr.values())))

        feature_cosine["LACE"][id_i] = feature_importance_similarity(
                target_vector, expl_lace_fi
            )


        expl_lace_feature_vector=np.asarray(feature_hit_oh_encoding(prediction_difference_attr.values()))
        

        feature_precision["LACE"][id_i], feature_recall["LACE"][id_i], feature_f1_score["LACE"][id_i], = explanation_precision_recall_f1_score(
                np.asarray(true_rule_oh), expl_lace_feature_vector
            )
            
        feature_f1_score_delta["LACE"][id_i] = rule_based_similarity(
            np.asarray(true_rule_oh),
            np.asarray(
                feature_hit_oh_encoding_delta(prediction_difference_attr.values())
            ),
        )
        rules_lace_oh = [
            oh_encoding_rule(rule, pc.feature_names) for rule in attributes_lace_rules
        ]

        if rules_lace_oh == []:
            rules_lace_oh = [oh_encoding_rule([], pc.feature_names)]
        rule_f1_score["LACE"][id_i] = max(
            [
                rule_based_similarity(np.asarray(r), np.asarray(true_rule_oh))
                for r in rules_lace_oh
            ]
        )
        rule_precision["LACE"][id_i] = max(
            [
                explanation_precision(np.asarray(r), np.asarray(true_rule_oh))
                for r in rules_lace_oh
            ]
        )
        rule_recall["LACE"][id_i] = max(
            [
                explanation_recall(np.asarray(r), np.asarray(true_rule_oh))
                for r in rules_lace_oh
            ]
        )
    print()

    # # Quality evaluation

    if verbose:
        print("feature_cosine")
        print(feature_cosine)

        print("feature_f1_score")
        print(feature_f1_score)

        print("feature_f1_score_delta")
        print(feature_f1_score_delta)

        print("rule_f1_score")
        print(rule_f1_score)

    feature_f1_score_mean = computeAverage(feature_f1_score)
    print("feature_f1_score_mean")
    print(feature_f1_score_mean)

    feature_f1_score_delta_mean = computeAverage(feature_f1_score_delta)
    print("feature_f1_score_delta_mean")
    print(feature_f1_score_delta_mean)

    rule_f1_score_mean = computeAverage(rule_f1_score)
    print("rule_f1_score_mean")
    print(rule_f1_score_mean)

    if saveResults:
        saveJson(feature_cosine, "feature_cosine_results", outputDirResults)
        saveJson(
            computeAverage(feature_cosine),
            "feature_cosine_mean",
            outputDirResults,
        )
        saveJson(feature_f1_score, "feature_f1_score_results", outputDirResults)
        saveJson(
            feature_f1_score_delta,
            "feature_f1_score_delta_results",
            outputDirResults,
        )

        saveJson(feature_precision, "feature_precision_results", outputDirResults)
        saveJson(
            computeAverage(feature_precision),
            "feature_precision_mean",
            outputDirResults,
        )
        saveJson(feature_recall, "feature_recall_results", outputDirResults)
        saveJson(
            computeAverage(feature_recall),
            "feature_recall_mean",
            outputDirResults,
        )

        saveJson(rule_f1_score, "rule_f1_score", outputDirResults)

        saveJson(
            computeAverage(feature_f1_score),
            "feature_f1_score_mean",
            outputDirResults,
        )
        saveJson(
            computeAverage(feature_f1_score_delta),
            "feature_f1_score_delta",
            outputDirResults,
        )
        saveJson(computeAverage(rule_f1_score), "rule_f1_score_mean", outputDirResults)

        saveJson(rule_precision, "rule_precision_results", outputDirResults)
        saveJson(
            computeAverage(rule_precision),
            "rule_precision_mean",
            outputDirResults,
        )
        saveJson(rule_recall, "rule_recall_results", outputDirResults)
        saveJson(
            computeAverage(rule_recall),
            "rule_recall_mean",
            outputDirResults,
        )

    # ### Ranking
    if verbose:
        print("LIME targets")
        print(lime_targets)
        print("SHAP targets")
        print(shap_targets)
        print("LACE targets")
        print(lace_targets)

    print("LIME targets mean", np.mean(list(lime_targets.values())))
    print("SHAP targets mean", np.mean(list(shap_targets.values())))
    print("LACE targets mean", np.mean(list(lace_targets.values())))

    # ### Rule hits
    if verbose:
        print("LACE rule hits")
        print(lace_hits)
        print("Anchor rule hits")
        print(anchor_hits)

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
    if verbose:
        print("LIME abs targets")
        print(lime_targets_abs)
        print("SHAP abs targets")
        print(shap_targets_abs)
        print("LACE abs targets")
        print(lace_targets_abs)

    print("LIME abs targets mean", np.mean(list(lime_targets_abs.values())))
    print("SHAP abs targets mean", np.mean(list(shap_targets_abs.values())))
    print("LACE abs targets mean", np.mean(list(lace_targets_abs.values())))

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

    from utils_RW import showHits

    print(anchor_hits)
    print(lace_hits)
    hits_dict_summary = {}
    hits_dict_summary["LACE"] = lace_hits
    hits_dict_summary["anchor"] = anchor_hits

    showHits(
        hits_dict_summary,
        upperLabel=n_explanations,
        percentage=True,
        saveFig=saveResults,
        outDir=outputDirResults,
    )
