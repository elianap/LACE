def encodeInstance(instance, encoders):
    import pandas as pd

    instance_en = pd.DataFrame(instance).T
    for attr in instance.keys():
        if attr in encoders:
            instance_en[attr] = (
                encoders[attr]
                .transform([instance[attr]])
                .astype(instance_en[attr].dtype)
            )
        else:
            instance_en[attr] = instance[attr]
    return instance_en.loc[0]


def decodeInstance(instance, encoders):
    instance_d = {}
    for attr in instance.keys():
        if attr in encoders:
            instance_d[attr] = encoders[attr].inverse_transform([instance[attr]])[0]
        else:
            instance_d[attr] = instance[attr]
    import pandas as pd

    instance_d = pd.Series(instance_d)
    return instance_d


def DiscretizeInstance(instance_dec, continuos_features, dataset_name=""):
    import pandas as pd

    instance_discr_df = pd.DataFrame(instance_dec).T
    from src.import_datasets import discretize

    instance_discr_df[continuos_features] = discretize(
        instance_discr_df,
        attributes=continuos_features,
        dataset_name=dataset_name,
    )
    instance_discr = instance_discr_df.loc[0]
    return instance_discr


def evaluateAllAttributeValues(instance, attr, pc, target_class_id):
    prob_fn_instance = pc.predict_fn(instance.values.reshape(1, -1))[0]
    # target_class = pc.class_names[target_class_id]
    tot = 0
    instance_pred_prob = prob_fn_instance[target_class_id]
    distr_attr = {
        k: v / len(pc.d_train.discreteDataset)
        for k, v in pc.d_train.discreteDataset[attr].value_counts().items()
    }[attr] = {
        k: v / len(pc.d_train.discreteDataset)
        for k, v in pc.d_train.discreteDataset[attr].value_counts().items()
    }

    for v in list(pc.encoders[attr].classes_):
        # from src.utils_analysis import decodeInstance, encodeInstance
        print(attr, v)
        from copy import deepcopy

        instance_dec = decodeInstance(deepcopy(instance), pc.encoders)
        instance_dec[attr] = v

        instance_en = encodeInstance(instance_dec, pc.encoders)
        # print(dict(instance_en))
        # print(dict(instance_dec))
        prob = pc.predict_fn(instance_en[pc.feature_names].values.reshape(1, -1))[0]
        print(prob)
        # probs_attr = prob
        print(
            f"v: {v} prob:{prob[target_class_id]} freq:{distr_attr[v]} prod: {prob[target_class_id]*distr_attr[v]}"
        )
        tot = prob[target_class_id] * distr_attr[v] + tot

    print(f"{instance_pred_prob-tot} = {instance_pred_prob} - {tot}")


def decode_rule(r_, l3clf_):
    r_class = l3clf_._class_dict[r_.class_id]
    r_attr_ixs_and_values = sorted([l3clf_._item_id_to_item[i] for i in r_.item_ids])
    r_attrs_and_values = [
        (l3clf_._column_id_to_name[c], v) for c, v in r_attr_ixs_and_values
    ]
    return {"body": r_attrs_and_values, "class": r_class}


def showSaveRules(
    l3clf,
    discretizedInstance,
    instance_predClass,
    id_expl,
    outputdir="./rules",
    saveFile=False,
    showLevel_2=False,
):

    if saveFile:
        from pathlib import Path

        Path(outputdir).mkdir(parents=True, exist_ok=True)

    def convertToItems(body):
        return frozenset([f"{a}={v}" for a, v in body])

    print(id_expl)
    # liv_2_rules = []

    for level, level_rules in {
        "lev_1": l3clf.lvl1_rules_,
        "lev_2": l3clf.lvl2_rules_,
    }.items():
        liv_i_rules_match = []
        # liv_i_rules_not_match = []
        for r in level_rules:
            decoded_rule = decode_rule(r, l3clf)
            for a, v in decoded_rule["body"]:
                if discretizedInstance[a] != v:
                    # Exit the inner loop, not entering the else clause, therefore not adding the rule
                    break
            # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
            else:

                info = (
                    list(r.item_ids),
                    convertToItems(decoded_rule["body"]),
                    decoded_rule["class"],
                    r.support,
                    r.confidence,
                    len(list(r.item_ids)),
                )

                if decode_rule(r, l3clf)["class"] == str(instance_predClass):
                    liv_i_rules_match.append(info)
                # else:
                #     liv_i_rules_not_match.append(info)

        if liv_i_rules_match:  # or liv_i_rules_not_match:
            import pandas as pd

            cols = ["ids", "body", "class", "sup", "conf", "len"]

            if liv_i_rules_match:
                # print("match")
                if level == "lev_1" or showLevel_2:
                    print(f"{level} - ({len(liv_i_rules_match)} rules)")
                    df_match = pd.DataFrame(liv_i_rules_match, columns=cols)
                    from IPython.display import display

                    display(df_match)

            # if liv_i_rules_not_match:
            #     print("not")
            #     df_not_match = pd.DataFrame(liv_i_rules_not_match, columns=cols)
            #     from IPython.display import display

            #     display(df_not_match)

            if level == "lev_2" and saveFile:
                df_match.to_csv(f"{outputdir}/level_2_{id_expl}.csv")
