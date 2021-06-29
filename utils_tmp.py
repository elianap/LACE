def recap_errors_investigate(explanation_fm, target_class):
    union_rule = explanation_fm.local_rules.getMaximalRule()
    check_exp = explanation_fm.checkCorrectness(target_class)

    from copy import deepcopy

    r_tmp = deepcopy(check_exp)
    r_tmp["error"] = explanation_fm.error
    r_tmp["pi"] = explanation_fm.pi
    r_tmp["max_rule"] = union_rule.rule_key if union_rule is not None else None
    r_tmp["max_rule_diff"] = (
        union_rule.prediction_difference if union_rule is not None else None
    )
    return r_tmp


def saveRecap(recap, dataset_name="", output_dir="./tmp", verbose=False):
    from pathlib import Path

    outDir = f"{output_dir}_{dataset_name}"
    Path(outDir).mkdir(parents=True, exist_ok=True)
    for t in recap:
        import pandas as pd

        df_rec = pd.DataFrame.from_dict(recap[t]).T

        df_rec.to_csv(f"./{outDir}/{t}.csv", index=False)
        if verbose:
            print(t)
            print(df_rec)