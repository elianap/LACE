class LACE_explanation_old:
    """The LACE_explainer class, through the `explain_instance` method allows
    to obtain a rule-based model-agnostic local explanation for an instance.

    Parameters
    ----------
    clf : sklearn classifier
        Any sklearn-like classifier can be passed. It must have the methods
        `predict` and `predict_proba`.
    train_dataset : Dataset
        The dataset from which the locality of the explained instance is created.
    min_sup : float
        L^3 Classifier's Minimum support parameter.
    """

    def __init__(
        self,
        LACE_explainer_o,
        diff_single,
        map_difference,
        k,
        error,
        instance,
        encoded_instance,
        target_class,
        errors,
        instance_class_index,
        prob,
        metas=None,
        discretizedInstance=None,
    ):
        self.LACE_explainer_o = LACE_explainer_o
        self.diff_single = diff_single
        self.map_difference = map_difference
        self.k = k
        self.error = error
        self.instance = instance
        self.encoded_instance = encoded_instance
        self.target_class = target_class
        self.errors = errors
        self.instance_class_index = instance_class_index
        self.prob = prob
        self.metas = metas
        self.infos = {
            "d": self.LACE_explainer_o.dataset_name,
            "model": self.LACE_explainer_o.model,
            "x": self.metas,
        }  # d_explain.metas[i]}
        self.discretizedInstance = (
            instance if discretizedInstance is None else discretizedInstance
        )

    def estimateUserRule(self, rule, decoded_target_class, featureMasking=False):
        # Rule = [1, 2] --> 1rst and 2nd attribute value
        if (
            min(rule) <= 0
            or max(rule) > self.LACE_explainer_o.train_dataset.lenAttributes()
        ):
            raise ValueError("Not a valid rule. Rule min=1, max=len(attributes)")

        return self.LACE_explainer_o.getEstimationUserRule(
            rule,
            self.encoded_instance,
            decoded_target_class,
            featureMasking=featureMasking,
        )

    def checkCorrectness(self, decoded_target_class, featureMasking=True):
        return self.LACE_explainer_o._checkCorrectness(
            self.encoded_instance, decoded_target_class, featureMasking=featureMasking
        )

    def plotExplanation(self, saveFig=False, dirName=".", figName="expl"):
        import matplotlib.pyplot as plt

        # plt.style.use('seaborn-talk')

        fig, pred_ax = plt.subplots(1, 1)

        attrs_values = [f"{k}={v}" for k, v in (self.instance.items())][:-1]
        infos_t = " ".join(
            [
                f"{k}={self.infos[k]}"
                if k in self.infos and self.infos[k] is not None
                else ""
                for k in ["x", "d", "model"]
            ]
        )
        # " ".join([f"{k}={v}" for k, v in infos.items()])
        title = f"{infos_t}\n p(class={self.target_class}|x)={self.prob:.2f} true class={self.instance.iloc[-1]}"

        pred_ax.set_title(title)
        pred_ax.set_xlabel(f"Δ target class={self.target_class}")

        dict_instance = dict(
            enumerate([f"{k}={v}" for k, v in self.instance.to_dict().items()], start=1)
        )
        rules = {
            ", ".join([dict_instance[int(i)] for i in rule_i.split(",")]): v
            for rule_i, v in self.map_difference.items()
        }
        mapping_rules = {
            list(rules.keys())[i]: f"Rule_{i+1}" for i in range(0, len(rules))
        }
        # Do not plot rules of 1 item
        rules_plot = {
            mapping_rules[k]: v for k, v in rules.items() if len(k.split(", ")) > 1
        }
        pred_ax.barh(
            attrs_values + list(rules_plot.keys()),
            width=self.diff_single + list(rules_plot.values()),
            align="center",
            color="#bee2e8",
            linewidth="1",
            edgecolor="black",
        )
        pred_ax.invert_yaxis()
        print([f"{v}={{{k}}}" for k, v in mapping_rules.items()])
        if saveFig:
            import os

            fig.savefig(os.path.join(dirName, f"{figName}.pdf"), bbox_inches="tight")
        # fig.show()
        # plt.close()
