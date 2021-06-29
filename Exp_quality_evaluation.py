#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", None)

toCompute = True

verbose=False
# # Import dataset

# for dataset_name, dataset_filename in [("cross_discrete", "cross_dataset_discretized"), ("chess_discrete", "chess_dataset_discretized") , ("groups_discrete", "groups_dataset_discretized")]:
for classifier_name in ["RF", "NN"]:  # ["RF", "NN"]:
    print("\n\n------------------------------------------------------------------")
    print(classifier_name)
    for dataset_name, dataset_filename in [
        ("cross_discrete", "cross_dataset_discretized"),
        ("chess_discrete", "chess_dataset_discretized"),
        ("groups_discrete", "groups_dataset_discretized"),
        ("groups_10_discrete", "groups_10_dataset_discretized"),
        # ("monks", ""),
        # ("adult", ""),
        # ("compas", ""),
    ]:
        print(
            "\n#########################################################################"
        )
        print(dataset_name)
        results_output = "results/quality_evaluation"

        outputDirResults = f"./{results_output}/{dataset_name}/{classifier_name}/"
        saveResults = True

        if saveResults:
            from src.utils import saveJson
            from pathlib import Path

            Path(outputDirResults).mkdir(parents=True, exist_ok=True)

        import pandas as pd

        if dataset_name == "adult":
            from src.import_datasets import import_process_adult

            df = import_process_adult()
            df.reset_index(drop=True, inplace=True)
        elif dataset_name == "compas":
            from src.import_datasets import import_process_compas

            risk_class_type = False
            dfI = import_process_compas(discretize=False, risk_class=risk_class_type)
            dfI.reset_index(drop=True, inplace=True)
            dfI["class"].replace({0: "Not recidivate", 1: "Recidivate"}, inplace=True)
            df = dfI.copy()
        elif dataset_name == "monks":
            from src.import_datasets import importArff

            data = importArff("./datasets/monks.arff")
            data.rename(columns={"y": "class"}, inplace=True)
            df = data.copy()
        else:

            df = pd.read_csv(f"./datasets/{dataset_filename}.csv")
        df.head()

        from ProcessedDataset import ProcessedDataset

        all_data = False
        pc = ProcessedDataset(df)

        if classifier_name == "RF":

            from sklearn.ensemble import RandomForestClassifier

            clf_init = RandomForestClassifier(random_state=42)

        elif classifier_name == "NN":

            from sklearn.neural_network import MLPClassifier

            clf_init = MLPClassifier(random_state=True)

        elif classifier_name == "NB":

            from sklearn.naive_bayes import MultinomialNB

            clf_init = MultinomialNB()

        else:
            raise ValueError()
        # pc.processDataset(
        #     clf_init, all_data=all_data, dataset_name=dataset_name, round_v=3, bins=3
        # )
        from sklearn import model_selection
        import numpy as np

        np.random.seed(42)
        df_train, df_test = model_selection.train_test_split(df, train_size=0.7)

        # # Process data
        from ProcessedDataset_v2 import ProcessedDatasetTrainTest

        pc = ProcessedDatasetTrainTest(df_train, df_test)

        pc.processTrainTestDataset(clf_init)  # , dataset_name=dataset_name)

        predicted = np.argmax(pc.predict_fn(pc.test.values), axis=1)
        # FP=[i for i,p in enumerate(predicted) if p==1 and labels_test[i]==0]
        # FN=[i for i,p in enumerate(predicted) if p==0 and labels_test[i]==1]
        mispredicted = [i for i, p in enumerate(predicted) if p != pc.labels_test[i]]
        correct_prediction = [
            i for i, p in enumerate(predicted) if p == pc.labels_test[i]
        ]

        from sklearn.metrics import accuracy_score

        print("Accuracy: ", round(accuracy_score(pc.labels_test, predicted), 4))

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

            if classifier_name == "RF":
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

        lace_explainer = LACE_explainer(
            pc.d_train, pc.predict_fn, dataset_name=dataset_name
        )

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

        def saveFigure(fig_lace, outdir, id_i, explainer_name="", w=4, h=3):
            fig_lace.set_size_inches(w, h)
            fig_lace.savefig(
                f"{outdir}/{explainer_name}_{id_i}.pdf", bbox_inches="tight"
            )

        # # Evaluation
        showExplanation = False
        n_explanations = 100

        # ## Ranking
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
        )

        target = ["X", "Y"]

        target_vector = np.asarray(
            [1.0 / float(len(target)) if i in target else 0 for i in pc.feature_names]
        )
        print(target_vector)

        true_rule_oh = [1 if f in target else 0 for f in list(pc.feature_names)]
        true_rule_oh

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
        rule_explainers = ["anchor", "LACE"]
        rule_f1_score = {e: {} for e in rule_explainers}

        for id_i in range(0, n_explanations):
            end_v = "\n" if id_i % 9 == 0 else " "
            print(id_i, end=end_v)
            from utils_RW import plot_lime_explanation

            predicted_class = pc.predict_fn_class(
                pc.test.iloc[id_i].values.reshape(1, -1)
            )[0]

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

            feature_f1_score["LIME"][id_i] = rule_based_similarity(
                np.asarray(true_rule_oh),
                np.asarray(feature_hit_oh_encoding(exp_lime_dict.values())),
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
                np.asarray(
                    feature_hit_oh_encoding_delta(sum_shap_for_categories.values())
                ),
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
                oh_encoding_rule(rule, pc.feature_names)
                for rule in attributes_anchor_rules
            ]
            if rules_anchor_oh ==[]:
                rules_anchor_oh = [oh_encoding_rule([], pc.feature_names)]

            rule_f1_score["anchor"][id_i] = max(
                [
                    rule_based_similarity(np.asarray(r), np.asarray(true_rule_oh))
                    for r in rules_anchor_oh
                ]
            )

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
            if showExplanation:
                fig_lace = explanation_fm.plotExplanation(
                    showRuleKey=False, retFig=True
                )
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
                prediction_difference_attr,
                list(pc.feature_names),
                target,
                absValue=True,
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
            feature_hit_oh = {
                k: 1 if v != 0.0 else 0 for k, v in prediction_difference_attr.items()
            }

            feature_f1_score["LACE"][id_i] = rule_based_similarity(
                np.asarray(true_rule_oh),
                np.asarray(
                    feature_hit_oh_encoding(prediction_difference_attr.values())
                ),
            )
            feature_f1_score_delta["LACE"][id_i] = rule_based_similarity(
                np.asarray(true_rule_oh),
                np.asarray(
                    feature_hit_oh_encoding_delta(prediction_difference_attr.values())
                ),
            )

            rules_lace_oh = [
                oh_encoding_rule(rule, pc.feature_names)
                for rule in attributes_lace_rules
            ]
            if rules_lace_oh == []:
                rules_lace_oh = [oh_encoding_rule([], pc.feature_names)]
            rule_f1_score["LACE"][id_i] = max(
                [
                    rule_based_similarity(np.asarray(r), np.asarray(true_rule_oh))
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

        feature_cosine_mean = computeAverage(feature_cosine)
        print("feature_cosine_mean")
        print(feature_cosine_mean)

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
            saveJson(feature_f1_score, "feature_f1_score_results", outputDirResults)
            saveJson(
                feature_f1_score_delta,
                "feature_f1_score_delta_results",
                outputDirResults,
            )
            saveJson(rule_f1_score, "rule_f1_score", outputDirResults)

            saveJson(
                computeAverage(feature_cosine), "feature_cosine_mean", outputDirResults
            )
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
            saveJson(
                computeAverage(rule_f1_score), "rule_f1_score_mean", outputDirResults
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

        # ## Rule hits
        from utils_RW import showHits

        print("anchor hits")
        print(anchor_hits)
        print("LACE hits")
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
