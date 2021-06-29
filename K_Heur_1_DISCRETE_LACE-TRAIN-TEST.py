#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", None)

toCompute = True

classifier_name = "RF"


# # Import dataset
dataset_name, dataset_filename = "cross_discrete", "cross_dataset_discretized"

# for dataset_name, dataset_filename in [("cross_discrete", "cross_dataset_discretized"), ("chess_discrete", "chess_dataset_discretized") , ("groups_discrete", "groups_dataset_discretized")]:
for classifier_name in ["RF", "NN", "NB"]:
    print("\n\n------------------------------------------------------------------")
    print(classifier_name)
    for dataset_name, dataset_filename in [
        ("cross_discrete", "cross_dataset_discretized"),
        ("chess_discrete", "chess_dataset_discretized"),
        ("groups_discrete", "groups_dataset_discretized"),
        ("groups_10_discrete", "groups_10_dataset_discretized"),
        ("monks", ""),
        ("adult", ""),
        ("compas", ""),
    ]:
        print(
            "\n#########################################################################"
        )
        print(dataset_name)
        results_output = "results/results_k_heuristic_multilple_classifiers"

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

        # ## LACE
        from src.LACE_explainer import LACE_explainer

        lace_explainer = LACE_explainer(
            pc.d_train, pc.predict_fn, dataset_name=dataset_name
        )

        # # Compute explanation
        outdir_lace = f"./results/{dataset_name}/lace/heuristic"

        saveFig = False

        if saveFig:
            from pathlib import Path

            Path(outdir_lace).mkdir(parents=True, exist_ok=True)

        def saveFigure(fig_lace, outdir, id_i, explainer_name="", w=4, h=3):
            fig_lace.set_size_inches(w, h)
            fig_lace.savefig(
                f"{outdir}/{explainer_name}_{id_i}.pdf", bbox_inches="tight"
            )

        def numberIterations(errors):
            return len(errors)

        def getFirstLastApproxError(errors):
            import operator

            error_last_iter = max(errors.items(), key=operator.itemgetter(0))[1]
            error_first_iter = min(errors.items(), key=operator.itemgetter(0))[1]
            return error_first_iter, error_last_iter

        # # Evaluation
        showExplanation = False
        n_explanations = 100

        iterations = {}
        approx_errors = {}
        print(pc.test.shape)
        
        # ## Heuristic
        for id_i in range(0, n_explanations):
            end_v = "\n" if id_i % 9 == 0 else " "
            print(id_i, end=end_v)
            predicted_class = pc.predict_fn_class(
                pc.test.iloc[id_i].values.reshape(1, -1)
            )[0]

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
                fig_lace = explanation_fm.plotExplanation(
                    showRuleKey=False, retFig=True
                )
                # explanation_fm.local_rules.printLocalRules()
                if saveFig:
                    saveFigure(fig_lace, outdir_lace, id_i, "lace")
                changes = explanation_fm.estimateSingleAttributeChangePrediction()
                if changes:
                    print(changes)
            iterations[id_i] = numberIterations(explanation_fm.errors)
            approx_errors[id_i] = getFirstLastApproxError(explanation_fm.errors)

        print()

        # ### Heuristic recap
        from statistics import mean, stdev

        print(iterations)
        n_iterations = list(iterations.values())

        stats_iterations = {
            "min": min(n_iterations),
            "max": max(n_iterations),
            "mean": mean(n_iterations),
            "stdev": stdev(n_iterations),
        }
        print(stats_iterations)
        if saveResults:
            saveJson(stats_iterations, "stats_iterations", outputDirResults)

        print(approx_errors)
        if saveResults:
            saveJson(approx_errors, "approx_errors", outputDirResults)

        approx_errors_v = list(approx_errors.values())
        from statistics import mean, stdev

        delta_approx = [v[0] - v[1] for v in approx_errors_v]

        stats_approx_error = {
            "min": min(delta_approx),
            "max": max(delta_approx),
            "mean": mean(delta_approx),
            "stdev": stdev(delta_approx),
        }

        print(stats_approx_error)
        if saveResults:
            saveJson(stats_approx_error, "stats_approx_error", outputDirResults)
