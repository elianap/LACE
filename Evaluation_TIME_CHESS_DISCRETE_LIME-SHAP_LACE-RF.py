#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

toCompute=True

classifier_name="RF"

results_output="results_output_with_time"

# # Import dataset
#dataset_name, dataset_filename="cross_discrete", "cross_dataset_discretized"

for dataset_name, dataset_filename in [("cross_discrete", "cross_dataset_discretized"), ("chess_discrete", "chess_dataset_discretized"), ("groups_discrete", "groups_dataset_discretized"), ("groups_10_discrete", "groups_10_dataset_discretized") ]:
    print("\n#########################################################################")
    print(dataset_name)
    outputDirResults=f"./{results_output}/{dataset_name}/{classifier_name}/"
 
    saveResults=True

    if saveResults:
        from src.utils import saveJson
        from pathlib import Path
        Path(outputDirResults).mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df=pd.read_csv(f"./datasets/{dataset_filename}.csv")
    df.head()

    from ProcessedDataset import ProcessedDataset

    all_data=False
    pc = ProcessedDataset(df)

    if classifier_name=="RF":

        from sklearn.ensemble import RandomForestClassifier
        clf_init = RandomForestClassifier(random_state=42)
        
    elif classifier_name=="NN":
        
        from sklearn.neural_network import MLPClassifier
        clf_init = MLPClassifier(random_state=True)
        
    else:
        raise ValueError()
    pc.processDataset(clf_init, all_data=all_data, dataset_name=dataset_name,  round_v=3, bins=3)

    predicted=np.argmax(pc.predict_fn(pc.test.values), axis=1)
    #FP=[i for i,p in enumerate(predicted) if p==1 and labels_test[i]==0]
    #FN=[i for i,p in enumerate(predicted) if p==0 and labels_test[i]==1]
    mispredicted=[i for i,p in enumerate(predicted) if p!=pc.labels_test[i]]
    correct_prediction=[i for i,p in enumerate(predicted) if p==pc.labels_test[i]]

    from sklearn.metrics import accuracy_score
    print("Accuracy: ", round(accuracy_score(pc.labels_test, predicted),4))


    # # Explainers

    # ## LIME
    import lime
    import lime.lime_tabular
    import numpy as np
    np.random.seed(42)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(pc.train.values ,feature_names = pc.feature_names,class_names=pc.class_names,
                                                       categorical_features=pc.categorical_features_pos, 
                                                       categorical_names=pc.categorical_names_LE, random_state=42)


    # ## SHAP
    if toCompute:
        import shap
        if classifier_name=="RF":
            shap_explainer = shap.TreeExplainer(pc.clf)
        else:
            shap_explainer = shap.KernelExplainer(model=pc.predict_fn_OH, data=pc.OH_X_train.values)#, link="logit")
        


    # ## Anchor
    from anchor import utils
    from anchor import anchor_tabular

    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        pc.class_names,
        pc.feature_names,
        pc.train.values,
        pc.categorical_names_LE)


    # ## LACE
    from src.LACE_explainer import LACE_explainer

    lace_explainer = LACE_explainer(pc.d_train,pc.predict_fn, dataset_name="COMPAS")


    # # Comparison
    outdir_lace=f"./results/{dataset_name}/lace"
    outdir_lime=f"./results/{dataset_name}/lime"
    outdir_shap=f"./results/{dataset_name}/shap"

    saveFig=False

    if saveFig:
        from pathlib import Path
        Path(outdir_lace).mkdir(parents=True, exist_ok=True)
        Path(outdir_lime).mkdir(parents=True, exist_ok=True)
        Path(outdir_shap).mkdir(parents=True, exist_ok=True)

    def lime_explanation_attr_sorted(exp_lime, label):
        exp = dict(exp_lime.as_list(label=label))
        names = exp_lime.domain_mapper.feature_names
        exp_lime_dict = {}
        for n in names:
            if n in exp:
                exp_lime_dict[n]=exp[n]
            else:
                for k in exp.keys():
                    if n in k:
                        exp_lime_dict[n]=exp[k]
        return exp_lime_dict


    def checkRankingTarget(rankings, target):
        targets_hit=[1 for t in target if rankings[t]<=len(target)]
        return sum(targets_hit)/len(target)

    def getFeatureRankingList(k_v, feature_names, absValue=False):
        rankings=k_v.copy()
        if absValue:
            for rank, k in enumerate(sorted(k_v, key=lambda k: abs(k_v[k]), reverse=True), 1):
                rankings[k]=rank
        else:
            for rank, k in enumerate(sorted(k_v, key=k_v.get, reverse=True), 1):
                rankings[k]=rank
        rankings={feature_names[f]:list(rankings.values())[f] for f in range(len(feature_names))}
        return rankings

    def evaluateRankingTarget(k_v,feature_names, target, absValue=False):
        rankings=getFeatureRankingList(k_v, feature_names, absValue=absValue)
        return checkRankingTarget(rankings, target)

    def computeHits(attributes_rules, target, hits, verbose=False):
        hit_rule_tuple=None
        superset=False
        subset=False
        for rule_attr in attributes_rules:
            if verbose:
                print(rule_attr)
            # the order does not matter
            if set(rule_attr)==set(target):
                hits["hit"]+=1
                hit_rule_tuple=(frozenset(rule_attr), "hit")
                if verbose:
                    print("hit", hit_rule_tuple)
                return hits, hit_rule_tuple
            elif frozenset(rule_attr).issuperset(frozenset(target)):
                superset=True
                hit_rule_tuple=(frozenset(rule_attr), "partial_hit_superset")
                if verbose:
                    print("partial_hit_superset")
            else:
                sunbs=[]
                for attribute in target:
                    if attribute in rule_attr:                    
                        subset=True
                        sunbs.append(attribute)
                if subset:
                    hit_rule_tuple=(frozenset(rule_attr), "partial_hit_subset")
                    if verbose:
                        print("partial_hit_subset", sunbs)
        if superset:
            hits["partial_hit_superset"]+=1
        elif subset:
            hits["partial_hit_subset"]+=1
        return hits, hit_rule_tuple

    def saveFigure(fig_lace, outdir, id_i, explainer_name="", w=4,h=3):
        fig_lace.set_size_inches(w,h)
        fig_lace.savefig(f"{outdir}/{explainer_name}_{id_i}.pdf", bbox_inches="tight")

    target=["X", "Y"]
    lime_targets={}
    shap_targets={}
    lace_targets={}

    for id_i in [42]:
        from utils_RW import plot_lime_explanation
        predicted_class=pc.predict_fn_class(pc.test.iloc[id_i].values.reshape(1,-1))[0]
        print("LIME")
        exp_lime = lime_explainer.explain_instance(pc.test.iloc[id_i].values, pc.predict_fn, num_features=len(pc.feature_names), labels=[predicted_class])#, num_samples=20000)
        fig_lime = plot_lime_explanation(exp_lime, label=predicted_class, pred=pc.class_names[np.argmax(pc.predict_fn(pc.test.iloc[id_i:id_i+1].values))], true_label=pc.class_names[pc.labels_test[id_i]],                                    fontsize=20)
        if saveFig:
            saveFigure(fig_lime, outputDirResults, id_i, "lime")
        exp_lime_dict=lime_explanation_attr_sorted(exp_lime, predicted_class)
        lime_targets[id_i]=evaluateRankingTarget(exp_lime_dict,list(pc.feature_names), target)
        
        
        from utils_RW import sumCategories, plotShapValues, convertInstance
        print("SHAP")
        instance=pc.OH_X_test_cols.iloc[id_i]
        shap_values = shap_explainer.shap_values(instance)
        matching_instance=convertInstance(instance, pc.categorical_features, pc.continuos_features)

        sum_shap_for_categories=sumCategories(shap_values[predicted_class], pc.oh_columns, pc.categorical_features, matching_instance)
        fig_shap=plotShapValues(list(sum_shap_for_categories.values()), list(sum_shap_for_categories.keys()), target_class=pc.class_names[predicted_class],                    pred=pc.class_names[pc.clf.predict([instance])[0]], true=pc.class_names[pc.labels_test[id_i]],                           fontsize=20)
        if saveFig:
            saveFigure(fig_shap, outputDirResults, id_i, "shap")
        shap_targets[id_i]=evaluateRankingTarget(sum_shap_for_categories,list(pc.feature_names), target)

     
        from utils_RW import printAnchor, dictAnchor
        exp_anchor = anchor_explainer.explain_instance(pc.test.iloc[id_i].values, pc.predict_fn_class, threshold=0.95)
        printAnchor(id_i, exp_anchor)
        dictAnchor(id_i, exp_anchor)
        featureMasking=True
        instance = pc.d_explain[id_i]
        infos = {"model": "RF"}
        instance_discretized = pc.d_explain.getDiscretizedInstance(id_i)
        explanation_fm = lace_explainer.explain_instance(
            instance,
            pc.class_names[predicted_class],
            featureMasking=featureMasking,
            discretizedInstance=instance_discretized,
            verbose=True)
        fig_lace =explanation_fm.plotExplanation(showRuleKey=False, retFig=True, fontsize=20)
        explanation_fm.local_rules.printLocalRules()
        if saveFig:
            saveFigure(fig_lace, outputDirResults, id_i, "lace")
        prediction_difference_attr=explanation_fm.getPredictionDifferenceDict()#{attrs_values[i]:explanation_fm.diff_single[i] for i in range(0,len(explanation_fm.diff_single))}
        lace_targets[id_i]=evaluateRankingTarget(prediction_difference_attr,list(pc.feature_names), target)


        
        changes=explanation_fm.estimateSingleAttributeChangePrediction()
        if changes:
            print(changes)


    # # Evaluation
    showExplanation=False
    n_explanations=100


    # ## Ranking
    target=["X", "Y"]
    lime_targets={}
    shap_targets={}
    lace_targets={}
    lime_targets_abs={}
    shap_targets_abs={}
    lace_targets_abs={}

    exec_times_dict={"LACE":{}, "anchor":{}, "SHAP":{}, "LIME":{}}

    lace_hits={"hit":0, "partial_hit_superset":0, "partial_hit_subset":0}
    anchor_hits={"hit":0, "partial_hit_superset":0, "partial_hit_subset":0}
    lace_rules_hit_all=[]
    anchor_rules_hit_all=[]

    import time
    for id_i in range(0,n_explanations):
        end_v="\n" if id_i%9==0 else " "
        print(id_i, end=end_v)
        from utils_RW import plot_lime_explanation
        predicted_class=pc.predict_fn_class(pc.test.iloc[id_i].values.reshape(1,-1))[0]
        
        start_time = time.time()
        exp_lime = lime_explainer.explain_instance(pc.test.iloc[id_i].values, pc.predict_fn, num_features=len(pc.feature_names), labels=[predicted_class])
        exec_time=time.time() - start_time
        exec_times_dict["LIME"][id_i]=exec_time
        
        if showExplanation:
            print("LIME")
            fig_lime = plot_lime_explanation(exp_lime, label=predicted_class, pred=pc.class_names[np.argmax(pc.predict_fn(pc.test.iloc[id_i:id_i+1].values))], true_label=pc.class_names[pc.labels_test[id_i]])
            if saveFig:
                saveFigure(fig_lime, outdir_lime, id_i, "lime")
        exp_lime_dict=lime_explanation_attr_sorted(exp_lime, predicted_class)
        lime_targets[id_i]=evaluateRankingTarget(exp_lime_dict,list(pc.feature_names), target)
        lime_targets_abs[id_i]=evaluateRankingTarget(exp_lime_dict,list(pc.feature_names), target, absValue=True)
        
        from utils_RW import sumCategories, plotShapValues, convertInstance
        
        instance=pc.OH_X_test_cols.iloc[id_i]
        start_time = time.time()
        shap_values = shap_explainer.shap_values(instance)
        exec_time=time.time() - start_time
        exec_times_dict["SHAP"][id_i]=exec_time
        matching_instance=convertInstance(instance, pc.categorical_features, pc.continuos_features)

        sum_shap_for_categories=sumCategories(shap_values[predicted_class], pc.oh_columns, pc.categorical_features, matching_instance)
        if showExplanation:
            print("SHAP")
            fig_shap=plotShapValues(list(sum_shap_for_categories.values()), list(sum_shap_for_categories.keys()), target_class=pc.class_names[predicted_class],                        pred=pc.class_names[pc.clf.predict([instance])[0]], true=pc.class_names[pc.labels_test[id_i]])
            if saveFig:
                saveFigure(fig_shap, outdir_shap, id_i, "shap")
        shap_targets[id_i]=evaluateRankingTarget(sum_shap_for_categories,list(pc.feature_names), target)
        shap_targets_abs[id_i]=evaluateRankingTarget(sum_shap_for_categories,list(pc.feature_names), target, absValue=True)
     
        from utils_RW import printAnchor, dictAnchor
        start_time = time.time()
        exp_anchor = anchor_explainer.explain_instance(pc.test.iloc[id_i].values, pc.predict_fn_class, threshold=0.95)
        exec_time=time.time() - start_time
        exec_times_dict["anchor"][id_i]=exec_time
        
        
        if showExplanation:
            printAnchor(id_i, exp_anchor)
            dictAnchor(id_i, exp_anchor)
        attributes_anchor_rules=[list(pc.feature_names)[attr] for attr in exp_anchor.features()]
        anchor_hits, anchor_rules_hit=computeHits([attributes_anchor_rules], target, anchor_hits, verbose=False)
        anchor_rules_hit_all.append(anchor_rules_hit)
        
        featureMasking=True
        instance = pc.d_explain[id_i]
        infos = {"model": "RF"}
        instance_discretized = pc.d_explain.getDiscretizedInstance(id_i)
        start_time = time.time()
        explanation_fm = lace_explainer.explain_instance(
            instance,
            pc.class_names[predicted_class],
            featureMasking=featureMasking,
            discretizedInstance=instance_discretized,
            verbose=False)
        exec_time=time.time() - start_time
        exec_times_dict["LACE"][id_i]=exec_time
        if showExplanation:
            fig_lace =explanation_fm.plotExplanation(showRuleKey=False, retFig=True)
            #explanation_fm.local_rules.printLocalRules()
            if saveFig:
                saveFigure(fig_lace, outdir_lace, id_i, "lace")
            changes=explanation_fm.estimateSingleAttributeChangePrediction()
            if changes:
                print(changes)
        prediction_difference_attr=explanation_fm.getPredictionDifferenceDict()#{attrs_values[i]:explanation_fm.diff_single[i] for i in range(0,len(explanation_fm.diff_single))}
        lace_targets[id_i]=evaluateRankingTarget(prediction_difference_attr,list(pc.feature_names), target)    
        lace_targets_abs[id_i]=evaluateRankingTarget(prediction_difference_attr,list(pc.feature_names), target, absValue=True)
        attributes_lace_rules=explanation_fm.getAttributesRules()
        lace_hits, lace_rules_hit=computeHits(attributes_lace_rules, target, lace_hits, verbose=False)
        lace_rules_hit_all.append(lace_rules_hit)
    print()


    # # ### Ranking
    # print("LIME")
    # print(lime_targets)
    # print("SHAP")
    # print(shap_targets)
    # print("LACE")
    # print(lace_targets)

    # print("LIME", np.mean(list(lime_targets.values())))
    # print("SHAP", np.mean(list(shap_targets.values())))
    # print("LACE", np.mean(list(lace_targets.values())))


    # # ### Rule hits
    # print("LACE")
    # print(lace_hits)
    # print("Anchor")
    # print(anchor_hits)

    # if saveResults:
    #     saveJson(lace_hits, "lace_hits", outputDirResults)
    #     saveJson(anchor_hits, "anchor_hits", outputDirResults)
    #     saveJson(lime_targets, "lime_targets", outputDirResults)
    #     saveJson(shap_targets, "shap_targets", outputDirResults)
    #     saveJson(lace_hits, "lace_targets", outputDirResults)


    #     mean_hit_targets={"LIME": np.mean(list(lime_targets.values())), 
    #     "SHAP":np.mean(list(shap_targets.values())),
    #     "LACE":np.mean(list(lace_targets.values()))}

    #     saveJson(mean_hit_targets, "mean_hit_targets", outputDirResults)


    # # ### Ranking abs
    # print("LIME abs")
    # print(lime_targets_abs)
    # print("SHAP abs")
    # print(shap_targets_abs)
    # print("LACE abs")
    # print(lace_targets_abs)

    # print("LIME abs", np.mean(list(lime_targets_abs.values())))
    # print("SHAP abs", np.mean(list(shap_targets_abs.values())))
    # print("LACE abs", np.mean(list(lace_targets_abs.values())))

    # if saveResults:
    #     saveJson(lime_targets_abs, "lime_targets_abs", outputDirResults)
    #     saveJson(shap_targets_abs, "shap_targets_abs", outputDirResults)
    #     saveJson(lace_targets_abs, "lace_targets_abs", outputDirResults)


    #     mean_hit_targets_abs={"LIME": np.mean(list(lime_targets_abs.values())), 
    #     "SHAP":np.mean(list(shap_targets_abs.values())),
    #     "LACE":np.mean(list(lace_targets_abs.values()))}

    #     saveJson(mean_hit_targets_abs, "mean_hit_targets_abs", outputDirResults)


    # # ## Rule hits
    # from utils_RW import showHits

    # print(anchor_hits)
    # print(lace_hits)
    # hits_dict_summary={}
    # hits_dict_summary["LACE"]=lace_hits
    # hits_dict_summary["anchor"]=anchor_hits

    # showHits(hits_dict_summary, upperLabel=n_explanations, percentage=True, saveFig=saveResults, outDir=outputDirResults)


    # # Time performance
    for explainer, time_perf in exec_times_dict.items():
        print(explainer)
        print(time_perf)
        if saveResults:
            saveJson(time_perf, f"time_performance_{explainer}", outputDirResults)

    from statistics import mean, stdev
    for explainer, time_perf in exec_times_dict.items():
        print(explainer)
        exec_times_v=list(time_perf.values())

        stats_time={"min":min(exec_times_v), "max":max(exec_times_v), "mean":mean(exec_times_v), "stdev":stdev(exec_times_v)}
        print(stats_time)
        if saveResults:
            saveJson(stats_time, f"stats_time_{explainer}", outputDirResults)

