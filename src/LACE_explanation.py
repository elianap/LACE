class LACE_explanation:
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
        local_rules=None,
        pi=None,
    ):
        self.LACE_explainer_o = LACE_explainer_o
        self.diff_single = diff_single
        self.map_difference = map_difference
        self.k = k
        self.pi = pi
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

        self.local_rules = local_rules
        # self.rules_hr = None
        # self.rule_id_difference = None
        # self.rules_hr_class = None

        dict_instance = dict(
            enumerate([f"{k}={v}" for k, v in self.instance.to_dict().items()], start=1)
        )

        self.local_rules.setRulesHR(dict_instance)

        self.rules_hr = {
            local_rule.rule_id: local_rule.rule_hr
            for local_rule in self.local_rules.rules
        }

        # TODO Eliana --> gestire la classe della union rule
        # self.rules_hr_class = {
        #     local_rule.rule_id: (local_rule.rule_hr, "-->", local_rule.rule_class)
        #     for local_rule in self.local_rules.rules
        # }

        # Id: local rule hr --> class
        self.rules_hr_class = self.local_rules.getHRLocalRule()

        self.rules_id_rule_key = {
            local_rule.rule_id: local_rule.rule_key
            for local_rule in self.local_rules.rules
        }

        self.rule_id_difference = {
            (local_rule.rule_id): {
                "rule": local_rule.rule_hr,
                "rule_class": local_rule.rule_class,
                "pred_diff": local_rule.prediction_difference,
            }
            for local_rule in self.local_rules.rules
        }

    def _attributes(self):
        # TODO -1
        return list(self.instance.keys()[:-1])

    def _attribute_values(self):
        # TODO -1
        return [f"{k}={v}" for k, v in (self.instance.items())][:-1]

    def estimateUserRule(self, rule, decoded_target_class, featureMasking=True):
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

    def getPredictionDifferenceDict(self):
        attrs_values = [f"{k}={v}" for k, v in (self.instance.items())][:-1]
        prediction_difference_single = {
            attrs_values[i]: self.diff_single[i]
            for i in range(0, len(self.diff_single))
        }
        return prediction_difference_single

    def getAttributesRules(self):
        rule_attributes = []
        for rule in self.local_rules.getListRules():
            rule_attributes.append([self._attributes()[i - 1] for i in rule])
        return rule_attributes

    def print_rules(self):
        self.local_rules.printLocalRules()

    def plotUserRule(
        self,
        rule,
        decoded_target_class,
        featureMasking=True,
        saveFig=False,
        dirName=".",
        figName="expl_user",
    ):
        if len(rule) <= 1:
            print(
                f"Rule should have len greater than one, len of {rule} is {len(rule)} instead"
            )
            return -1
        from src.local_rules import UserRule

        user_rule_dict = self.estimateUserRule(
            rule, decoded_target_class, featureMasking=featureMasking
        )

        id_rule = list(user_rule_dict.keys())[0]
        user_rule_list = list(map(int, id_rule.split(",")))

        dict_instance = dict(
            enumerate([f"{k}={v}" for k, v in self.instance.to_dict().items()], start=1)
        )
        user_rule = UserRule(user_rule_list)
        user_rule.setPredictionDifference(user_rule_dict[id_rule])
        user_rule.setRuleHR(dict_instance)
        user_rule.setRuleId("1")

        fig, pred_ax = self.plotExplanation(saveFig=False, retFig=True)
        pred_ax.barh(
            user_rule.rule_id,
            user_rule.prediction_difference,
            linewidth="1",
            edgecolor="black",
        )

        print(self.rules_hr_class)
        user_rule.printHRRule()

        if saveFig:
            import os

            fig.savefig(os.path.join(dirName, f"{figName}.pdf"), bbox_inches="tight")

    def checkCorrectness(self, decoded_target_class, featureMasking=True):
        return self.LACE_explainer_o._checkCorrectness(
            self.encoded_instance,
            decoded_target_class,
            featureMasking=featureMasking,
            pi=self.pi,
        )

    def plotExplanation(
        self,
        saveFig=False,
        dirName=".",
        figName="expl",
        retFig=False,
        saveRule=False,
        showRuleKey=False,
        interactive=False,
        c1="#e6f2ff",
        c2="#cce6ff",
        fontsize=14,
    ):

        attrs_values = [f"{k}={v}" for k, v in (self.instance.items())][:-1]
        infos_t = " ".join(
            [
                f"{k}={self.infos[k]}"
                if k in self.infos and self.infos[k] is not None
                else ""
                for k in ["x", "d", "model"]
            ]
        )
        predicted_class = self.LACE_explainer_o.getDecodedPredictedClass(
            self.encoded_instance
        )
        title = f"{infos_t} p(class={self.target_class}|x)={self.prob:.2f}\n$y_{{pred}}$={predicted_class} $y_{{true}}$={self.instance.iloc[-1]}"

        xaxis_title = f"δ target class={self.target_class}"

        # Do not plot rules of 1 item
        rules_plot = {
            f"{local_rule.rule_id} - {self.rules_id_rule_key[local_rule.rule_id]}"
            if showRuleKey
            else local_rule.rule_id: local_rule.prediction_difference
            for local_rule in self.local_rules.rules
            if len(local_rule.rule) > 1
        }

        x = self.diff_single + list(rules_plot.values())
        y = attrs_values + list(rules_plot.keys())
        print(self.rules_hr_class)
        colors = [c1] * len(attrs_values) + [c2] * len(rules_plot.keys())
        if interactive:
            from src.utils_plot import barPlotPlotly

            fig = barPlotPlotly(x, y, title, xaxis_title=xaxis_title)
        else:
            from src.utils_plot import barPlotMPL

            fig = barPlotMPL(
                x, y, title, xaxis_title=xaxis_title, colors=colors, fontsize=fontsize
            )

        if saveFig:
            import os

            fig.savefig(os.path.join(dirName, f"{figName}.pdf"), bbox_inches="tight")

        if saveRule:
            out_data = {
                "rules": self.rule_id_difference,
                "diff_single": self.diff_single,
            }
            import json

            with open(os.path.join(dirName, f"{figName}.json"), "w") as f:
                json.dump(out_data, f)

        if retFig:
            return fig  # , pred_ax

    def plotExplanation_v1(
        self,
        saveFig=False,
        dirName=".",
        figName="expl",
        retFig=False,
        saveRule=False,
        showRuleKey=False,
    ):
        import matplotlib.pyplot as plt

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

        title = f"{infos_t}\n p(class={self.target_class}|x)={self.prob:.2f} true class={self.instance.iloc[-1]}"

        pred_ax.set_title(title)
        pred_ax.set_xlabel(f"δ target class={self.target_class}")

        # Do not plot rules of 1 item
        rules_plot = {
            f"{local_rule.rule_id} - {self.rules_id_rule_key[local_rule.rule_id]}"
            if showRuleKey
            else local_rule.rule_id: local_rule.prediction_difference
            for local_rule in self.local_rules.rules
            if len(local_rule.rule) > 1
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

        print(self.rules_hr_class)

        if saveFig:
            import os

            fig.savefig(os.path.join(dirName, f"{figName}.pdf"), bbox_inches="tight")

        if saveRule:
            out_data = {
                "rules": self.rule_id_difference,
                "diff_single": self.diff_single,
            }
            import json

            with open(os.path.join(dirName, f"{figName}.json"), "w") as f:
                json.dump(out_data, f)

        if retFig:
            return fig, pred_ax

    # Conterfactual change
    def estimateSingleAttributeChangePrediction(self, verbose=False):
        import numpy as np

        encoded_instance = self.encoded_instance
        predict_fn = self.LACE_explainer_o.predict_fn
        training_dataset = self.LACE_explainer_o.train_dataset
        classes_list = self.LACE_explainer_o.train_dataset.class_values()

        attribute_values = self._attribute_values()
        attributes = self._attributes()
        ##### Single
        from copy import deepcopy

        P = len(training_dataset.attributes())
        instance_i = deepcopy(encoded_instance)
        instance_i = instance_i[:-1].to_numpy().reshape(1, -1)
        masker_data = training_dataset.X_numpy()
        masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
        mask = np.zeros(P, dtype=np.bool)
        f = predict_fn

        pred_class_id = np.argmax(predict_fn(instance_i)[0])
        pred_class = classes_list[pred_class_id]
        changes = {}
        import scipy

        diff_f = np.zeros((P, len(training_dataset.class_values())))
        # For each attribute of the instance
        diff_av_dict = {}
        for i in range(P):
            mask[:] = 1
            mask[i] = 0

            masked_data = masker(instance_i, mask)

            f_masked = f(masked_data)
            preds = f_masked.argmax(1)

            id_diff = np.where(preds != pred_class_id)
            distinct_preds = np.unique(preds[np.where(preds != pred_class_id)])
            diff_av = np.unique(masked_data[id_diff][:, i]).astype(int)
            diff_av = training_dataset.decodeAttribute(diff_av, attributes[i])
            if len(distinct_preds) == 1:
                class_val = classes_list[distinct_preds[0]]
            else:
                class_val = None

            diff_av_dict.update({f"{attributes[i]}={v}": class_val for v in diff_av})
            avg_remove_f_i = f_masked.mean(0)

            diff_f[i] = f(instance_i.reshape(1, -1))[0] - avg_remove_f_i
            diff_f[i] = avg_remove_f_i
            new_class = classes_list[np.argmax(diff_f[i])]

            if new_class != pred_class:
                changes[attribute_values[i]] = diff_av_dict
                if verbose:
                    print(attribute_values[i], "changed")

        return changes, diff_av_dict

    def estimateSingleAttributeChangePrediction_V1(self, verbose=False):
        import numpy as np

        encoded_instance = self.encoded_instance
        predict_fn = self.LACE_explainer_o.predict_fn
        training_dataset = self.LACE_explainer_o.train_dataset
        classes_list = self.LACE_explainer_o.train_dataset.class_values()

        attribute_values = self._attribute_values()
        ##### Single
        from copy import deepcopy

        P = len(training_dataset.attributes())
        instance_i = deepcopy(encoded_instance)
        instance_i = instance_i[:-1].to_numpy().reshape(1, -1)
        masker_data = training_dataset.X_numpy()
        masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
        mask = np.zeros(P, dtype=np.bool)
        f = predict_fn

        pred_class = classes_list[np.argmax(predict_fn(instance_i)[0])]
        changes = {}
        import scipy

        diff_f = np.zeros((P, len(training_dataset.class_values())))
        # For each attribute of the instance
        for i in range(P):
            mask[:] = 1
            mask[i] = 0
            avg_remove_f_i = f(masker(instance_i, mask)).mean(0)

            diff_f[i] = f(instance_i.reshape(1, -1))[0] - avg_remove_f_i
            diff_f[i] = avg_remove_f_i
            new_class = classes_list[np.argmax(diff_f[i])]
            if new_class != pred_class:
                changes[attribute_values[i]] = new_class
                if verbose:
                    print(i, "changed")
        return changes

    def interactiveAttributeLevelInspectionExp(self, target_class):
        instance_dec = self.instance
        from interactive import interactiveAttributeLevelInspection

        interactiveAttributeLevelInspection(
            self.LACE_explainer_o,
            instance_dec,
            target_class,
            self.LACE_explainer_o.train_dataset._column_encoders,
        )


# New version with feature masking
def _compute_prediction_difference_single_conterfact(
    encoded_instance, predict_fn, training_dataset, classes_list, verbose=False
):
    from copy import deepcopy
    import numpy as np

    P = len(training_dataset.attributes())
    instance_i = deepcopy(encoded_instance)
    instance_i = instance_i[:-1].to_numpy().reshape(1, -1)
    masker_data = training_dataset.X_numpy()
    masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
    mask = np.zeros(P, dtype=np.bool)
    f = predict_fn

    pred_class = classes_list[np.argmax(predict_fn(instance_i)[0])]
    changes = []
    import scipy

    diff_f = np.zeros((P, len(training_dataset.class_values())))
    # For each attribute of the instance
    for i in range(P):
        mask[:] = 1
        mask[i] = 0
        avg_remove_f_i = f(masker(instance_i, mask)).mean(0)

        diff_f[i] = f(instance_i.reshape(1, -1))[0] - avg_remove_f_i
        diff_f[i] = avg_remove_f_i
        new_class = classes_list[np.argmax(diff_f[i])]
        if new_class != pred_class:
            changes.append(i)
            if verbose:
                print(i, "changed")
    return changes
