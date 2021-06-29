"""The module provides the LACE_explainer class, which is used to perform the
$name analysis on a dataset.
"""
import itertools
from collections import Counter
from copy import deepcopy
import numpy as np

import pandas as pd
import sklearn.neighbors
from l3wrapper.l3wrapper import L3Classifier

from src.dataset import Dataset
from src.LACE_explanation import LACE_explanation

ERROR_DIFFERENCE_THRESHOLD = 0.01
ERROR_THRESHOLD = 0.02
MINIMUM_SUPPORT = 0.01
SMALL_DATASET_LEN = 150

clf_names = {"<class 'sklearn.ensemble.forest.RandomForestClassifier'>": "RF"}


class LACE_explainer:
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
        # clf,
        train_dataset,
        predict_fn,
        min_sup=MINIMUM_SUPPORT,
        dataset_name=None,
        # predict_fn=None,
        clf_type="",
    ):
        # self.clf = clf

        self.predict_fn = predict_fn

        self.train_dataset = train_dataset

        self.K, self.max_K = _get_KNN_threshold_max(len(self.train_dataset))

        self.starting_K = self.K

        self.nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=len(self.train_dataset.Z_encoded),  # len(self.train_dataset),
            metric="euclidean",
            algorithm="auto",
            metric_params=None,
        ).fit(
            self.train_dataset.Z_encoded
        )  # self.train_dataset.X_numpy_NN()

        self.min_sup = min_sup

        self.decoded_class_frequencies = self.train_dataset.Y_decoded().value_counts(
            normalize=True
        )
        self.dataset_name = dataset_name
        # self.model = (
        #     clf_names[str(type(self.clf))] if str(type(self.clf)) in clf_names else None
        # )
        self.model = clf_names[str(clf_type)] if str(clf_type) in clf_names else None

    def getTargetClassIndex(self, decoded_target_class):
        return self.train_dataset.class_values().index(decoded_target_class)

    def predict_class(self, input):
        import numpy as np

        return np.argmax(self.predict_fn(input), axis=1)

    def getDecodedPredictedClass(self, instance):
        class_column_name = self.train_dataset.class_column_name()
        if class_column_name in instance:
            instance_values = instance.drop(class_column_name).values
        predicted_class = self.predict_class(instance_values.reshape(1, -1))[0]
        return self.train_dataset.class_values()[predicted_class]

    def explain_instance(
        self,
        encoded_instance: pd.Series,
        decoded_target_class,
        retObj=True,
        metas=None,
        featureMasking=False,
        discretizedInstance=None,
        verbose=False,
        all_rules=False,
        starting_K_input=None,  # TODO
    ):
        """
        Explain the classifer's prediction on the instance with respect to a
        target class.

        Parameters
        ----------

        encoded_instance :
            The instance whose prediction needs to be explained. It may come
            from a :class:`~src.dataset.Dataset`. In that case, the `encoded`
            form an instance must be used i.e. :code:`my_dataset.X()[42]`, as
            opposed to the `decoded` form i.e. :code:`my_dataset.X_decoded()[42]`

        decoded_target_class : str
            The name of the class for which the prediction is explained. It may
            come from a :class:`~src.dataset.Dataset`. In that case, the
            `decoded` form must be used i.e. :code:`my_dataset.class_values()[3]`.

        Returns
        -------
        explanation : dict
            The explanation

            ::

                {
                    'LACE_explainer_o': <src.LACE_explainer.LACE_explainer object at 0x7fde70c7a828>,
                    'diff_single': [0.11024691358024685, 0.02308641975308645, 0.19728395061728388, 0.31407407407407417, 0.004938271604938316, 0.006913580246913575, 0.0, 0.07111111111111112, 0.00864197530864197, 0.03358024691358019, 0.0007407407407408195, 0.0, 0.005185185185185182, 0.0, 0.0, 0.0],
                    'map_difference': {'1,2,3,4,8,9,10,11': 0.5839506172839506},
                    'k': 33,
                    'error': 0.00864197530864197,
                    'instance':
                        hair             1
                        feathers         0
                        eggs             0
                        milk             1
                        airborne         0
                        aquatic          0
                        predator         0
                        toothed          1
                        backbone         1
                        breathes         1
                        venomous         0
                        fins             0
                        legs             4
                        tail             1
                        domestic         0
                        catsize          1
                        type        mammal
                        dtype: object,
                    'target_class': 'mammal',
                    'errors': [0.00864197530864197],
                    'instance_class_index': 5,
                    'prob': 1.0
                }
            Field description:

            * ``"LACE_explainer_o"``: A reference to the explainer used to
              generate this explanation.
            * ``diff_single``: The prediction difference of every single
              attribute of the instance to explain.
              Thus, it has the same len as the instance attributes.
            * ``map_difference``: A dict where each key is a subset of the
              attributes and each key is prediction difference of such subset.
            * ``k``: The size of the neighborhood used in the last iteration of
              the algorithm.
            * ``error``: The approximation error :math:`epsilon` of the last
              iteration of the algorithm.
            * ``instance``: A reference to the instance whose prediction was
              explained.
            * ``target_class``: The name of the class for which the prediction
              is explained.
            * ``errors``: A list of all the approximation errors calculated
              at every step of the algorithm.
            * ``instance_class_index``: ``target_class`` after being processed
              with sklearn's LabelEncoder.
            * ``prob``: The probability (given by the classifier) of the
              instance belonging to the ``target_class``.

        """
        discretizedInstance = (
            self.train_dataset.inverse_transform_instance(
                encoded_instance[:-1], featureMasking=featureMasking
            )
            if discretizedInstance is None
            else discretizedInstance
        )
        target_class_index = self.train_dataset.class_values().index(
            decoded_target_class
        )

        encoded_instance_x = encoded_instance[:-1].to_numpy().reshape(1, -1)

        # Problem with very small training dataset. The starting k is low, very few examples:
        # difficult to capture the locality.
        # Risks: examples too similar, only 1 class. Starting k: proportional to the class frequence
        training_dataset_len = len(self.train_dataset)
        if training_dataset_len < SMALL_DATASET_LEN:
            decoded_pred_class = self.train_dataset.class_values()[
                self.predict_class(encoded_instance_x)[0].astype(int)
            ]
            self.starting_K = max(
                int(
                    self.decoded_class_frequencies[decoded_pred_class]
                    * training_dataset_len
                ),
                self.starting_K,
            )

        if starting_K_input is not None:
            self.starting_K = starting_K_input

        # Initialize k and error to be defined in case the for loop is not entered
        k = self.starting_K

        old_error = 10.0
        error = 1e9
        class_prob = self.predict_fn(encoded_instance_x)[0][target_class_index]

        single_attribute_differences = _compute_prediction_difference_single(
            encoded_instance,
            self.predict_fn,
            class_prob,
            target_class_index,
            self.train_dataset,
            featureMasking=featureMasking,
        )
        difference_map = {}

        first_iteration = True

        # errors = []
        errors = {}

        # Euristically search for the best k to use to approximate the local model
        for k in range(self.starting_K, self.max_K, self.K):
            if verbose:
                print(f"\n\ncompute_lace_step k={k}")
            local_rules, error, pi = self._compute_lace_step(
                encoded_instance,
                k,
                self.decoded_class_frequencies[decoded_target_class],
                target_class_index,
                class_prob,
                single_attribute_differences,
                featureMasking=featureMasking,
                discretizedInstance=discretizedInstance,
                verbose=verbose,
                all_rules=all_rules,
            )

            difference_map = local_rules.getPredictionDifferenceMap()

            if verbose:
                print(difference_map)

            errors[k] = error
            if verbose:
                if error != -1:
                    print("Error", error)
            # TODO: fix this, it does not correspond exactly to the paper, it should return the past
            #       error I believe
            # If we have reached the minimum or we are stuck in a local minimum
            if (abs(error) < ERROR_THRESHOLD) or (
                (abs(error) - abs(old_error)) > ERROR_DIFFERENCE_THRESHOLD
                and not first_iteration
            ):
                if (
                    abs(error) - abs(old_error)
                ) > ERROR_DIFFERENCE_THRESHOLD and not first_iteration:
                    k = k - self.K  # NEW
                    if verbose:
                        print("Rollback")
                break
            else:
                first_iteration = False
                old_error = error

        # print("explain_instance errors:", ", ".join([f"{err:.3E}" for err in errors]))
        # TODO --> save previous difference_map and error
        local_rules, error, pi = self._compute_lace_step(
            encoded_instance,
            k,
            self.decoded_class_frequencies[decoded_target_class],
            target_class_index,
            class_prob,
            single_attribute_differences,
            featureMasking=featureMasking,
            discretizedInstance=discretizedInstance,
        )
        difference_map = local_rules.getPredictionDifferenceMap()
        local_rules.setRulesId()
        local_rules.setPi(pi)

        if retObj:
            xp = LACE_explanation(
                self,
                single_attribute_differences,
                deepcopy(difference_map),
                k,
                error,
                self.train_dataset.inverse_transform_instance(
                    encoded_instance, featureMasking=featureMasking
                ),
                encoded_instance,
                decoded_target_class,
                errors,
                target_class_index,
                self.predict_fn(encoded_instance_x)[0][target_class_index],
                metas=metas,
                local_rules=local_rules,
                pi=pi,
            )
        else:
            xp = {
                "LACE_explainer_o": self,
                "diff_single": single_attribute_differences,
                "map_difference": deepcopy(difference_map),
                "k": k,
                "error": error,
                "instance": self.train_dataset.inverse_transform_instance(
                    encoded_instance, featureMasking=featureMasking
                ),
                "target_class": decoded_target_class,
                "errors": errors,
                "instance_class_index": target_class_index,
                "prob": self.predict_fn(encoded_instance_x)[0][target_class_index],
                "local_rules": local_rules,
            }
        return xp

    def _checkCorrectness(
        self,
        encoded_instance,
        decoded_target_class,
        featureMasking=True,
        pi=None,
        verbose=False,
    ):

        # TODO sitemare
        target_class_index = self.train_dataset.class_values().index(
            decoded_target_class
        )
        # TODO
        encoded_instance_x = encoded_instance[:-1].to_numpy().reshape(1, -1)
        class_prob = self.predict_fn(encoded_instance_x)[0][target_class_index]
        allRuleKey = list(range(1, self.train_dataset.lenAttributes() + 1))
        allRule = self.getEstimationUserRule(
            allRuleKey,
            encoded_instance,
            decoded_target_class,
            featureMasking=featureMasking,
        )

        prior_prob = {
            k: v / len(self.train_dataset.Y_decoded())
            for k, v in self.train_dataset.Y_decoded().value_counts().items()
        }
        # TODO
        if (class_prob - prior_prob[decoded_target_class]) != pi:
            raise Warning("Different pi!")

        if verbose:
            print(prior_prob)
            print(class_prob - prior_prob[decoded_target_class])
            print(allRule)
            # print(pi - allRule[",".join(map(str, allRuleKey))])
            # Equal
            print(
                (class_prob - prior_prob[decoded_target_class])
                - allRule[",".join(map(str, allRuleKey))]
            )
        eps_all_instance = (class_prob - prior_prob[decoded_target_class]) - allRule[
            ",".join(map(str, allRuleKey))
        ]
        return {
            "class_prob": class_prob,
            "prior_for_class": prior_prob[decoded_target_class],
            "pred_diff_all_instance": allRule[",".join(map(str, allRuleKey))],
            "eps_all_instance": eps_all_instance,
        }

    def _compute_lace_step(
        self,
        encoded_instance,
        k,
        target_class_frequency,
        target_class_index,
        class_prob,
        single_attribute_differences,
        featureMasking=False,
        discretizedInstance=None,
        verbose=False,
        all_rules=False,
    ):
        # Generate the neighborhood of the instance, classify it, and return the rules created by L3
        l3clf = L3Classifier(min_sup=self.min_sup)
        if featureMasking:
            local_rules = _create_locality_and_get_rules(
                self.train_dataset,
                self.nbrs,
                encoded_instance,
                k,
                self.predict_fn,
                self.train_dataset.encoder_nn,
                l3clf,
                featureMasking,
                discretizedDataset=self.train_dataset.discreteDataset,
                discretizedInstance=discretizedInstance,
                verbose=verbose,
                all_rules=all_rules,
            )
        else:
            local_rules = _create_locality_and_get_rules(
                self.train_dataset,
                self.nbrs,
                encoded_instance,
                k,
                self.predict_fn,
                self.train_dataset.encoder_nn,
                l3clf,
                featureMasking=featureMasking,
                verbose=verbose,
                all_rules=all_rules,
            )

        # For each rule, calculate the prediction difference for the its attributes
        # difference_map = {}
        for local_rule in local_rules.rules:
            rule = local_rule.rule

            pred_difference_rule = _compute_prediction_difference_subset(
                self.train_dataset,
                encoded_instance,
                rule,
                self.predict_fn,
                target_class_index,
                featureMasking=featureMasking,
            )
            # difference_map[rule_key] = pred_difference_rule
            local_rule.setPredictionDifference(pred_difference_rule)
            # local_rule.setRuleKey(rule_key)

        rules = local_rules.getListRules()
        difference_map = local_rules.getPredictionDifferenceMap()
        # Compute the approximation error
        pi, _, error, _ = _compute_approximation_error(
            target_class_frequency,
            class_prob,
            single_attribute_differences,
            rules,
            difference_map,
            verbose=verbose,
        )

        return local_rules, error, pi

    # def _getGlobalExplanationRules(self, d_explain):
    #     # noinspection PyUnresolvedReferences
    #     from src.global_explanation_old import GlobalExplanation

    #     global_expl = GlobalExplanation(self, d_explain)
    #     global_expl = global_expl.getGlobalExplanation()
    #     return global_expl

    def getEstimationUserRule(
        self, rule, encoded_instance, decoded_target_class, featureMasking=False
    ):
        target_class_index = self.getTargetClassIndex(decoded_target_class)

        return {
            ",".join(map(str, rule)): _compute_prediction_difference_subset(
                self.train_dataset,
                encoded_instance,
                rule,
                self.predict_fn,
                target_class_index,
                featureMasking=featureMasking,
            )
        }


def _create_locality_and_get_rules(
    training_dataset: Dataset,
    nbrs,
    encoded_instance: pd.Series,
    k: int,
    predict_fn,
    encoder_nn,
    l3clf,
    featureMasking=False,
    discretizedDataset=None,
    discretizedInstance=None,
    verbose=False,
    all_rules=False,
):
    if featureMasking:
        return _create_locality_and_get_rules_NEW(
            training_dataset,
            nbrs,
            encoded_instance,
            k,
            predict_fn,
            encoder_nn,
            l3clf,
            discretizedDataset,
            discretizedInstance,
            verbose=verbose,
            all_rules=all_rules,
        )
    else:
        return _create_locality_and_get_rules_OLD(
            training_dataset,
            nbrs,
            encoded_instance,
            k,
            predict_fn,
            encoder_nn,
            l3clf,
            verbose=verbose,
            all_rules=all_rules,
        )


def _create_locality_and_get_rules_NEW(
    training_dataset: Dataset,
    nbrs,
    encoded_instance: pd.Series,
    k: int,
    predict_fn,
    encoder_nn,
    l3clf,
    discretizedDataset,
    discretizedInstance,
    verbose=False,
    all_rules=False,
):

    from src.local_rules import LocalRule, LocalRules, UnionRule

    instance_x = encoded_instance[:-1].to_numpy()
    # Neigbors in X
    # X_encoder_nn = encoder_nn(instance_x.reshape(1, -1)) if encoder_nn else [instance_x]
    # nearest_neighbors_ixs = nbrs.kneighbors(X_encoder_nn, k, return_distance=False)[0]

    instance_discretized_encoded = training_dataset.encodeInstance_Z(
        discretizedInstance, discretizedInstance.keys()
    )

    nearest_neighbors_ixs = nbrs.kneighbors(
        [instance_discretized_encoded], k, return_distance=False
    )[0]

    # TMP TODO

    instance_predClass = training_dataset.decodeAttribute(
        np.asarray(
            [np.argmax(predict_fn(instance_x.reshape(1, -1)), axis=1)[0]],
            dtype=int,
        ),
        training_dataset.class_column_name(),
    )[0]

    instanceDf = pd.DataFrame(discretizedInstance).T
    X_neighbors = pd.concat([instanceDf, discretizedDataset.loc[nearest_neighbors_ixs]])

    y_pred_neighbors = [
        np.argmax(predict_fn(instance_x.reshape(1, -1)), axis=1)[0]
    ] + list(
        np.argmax(
            predict_fn(training_dataset.X().iloc[nearest_neighbors_ixs].values), axis=1
        )
    )
    y_pred_neighbors = training_dataset.decodeAttribute(
        np.asarray(y_pred_neighbors, dtype=int), training_dataset.class_column_name()
    )
    columns_n = X_neighbors.columns.to_list()

    l3clf.fit(
        X_neighbors.values,
        y_pred_neighbors,
        column_names=columns_n,
    )
    # tmp = X_neighbors.copy()
    # tmp["class"] = y_pred_neighbors
    # tmp.to_csv(f"tmp_{id_instance}_fm.csv")

    # Drop rules which use values not in the decoded instance
    decoded_instance = training_dataset.inverse_transform_instance(
        encoded_instance, featureMasking=True
    )

    encoded_rules = l3clf.lvl1_rules_

    def decode_rule(r_, l3clf_):
        r_class = l3clf_._class_dict[r_.class_id]
        r_attr_ixs_and_values = sorted(
            [l3clf_._item_id_to_item[i] for i in r_.item_ids]
        )
        r_attrs_and_values = [
            (l3clf_._column_id_to_name[c], v) for c, v in r_attr_ixs_and_values
        ]
        return {"body": r_attrs_and_values, "class": r_class}

    if verbose:
        from src.utils_analysis import showSaveRules

        showSaveRules(
            l3clf,
            discretizedInstance,
            instance_predClass,
            f"{k}",
        )

    local_rules = LocalRules()
    for r in encoded_rules:
        # For each of its attributes and values
        for a, v in decode_rule(r, l3clf)["body"]:
            # If rule uses an attribute's value different from the instance's
            if discretizedInstance[a] != v:
                # Exit the inner loop, not entering the else clause, therefore not adding the rule
                break
        # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
        else:
            # If the inner loop has completed normally without break-ing, then all of the rule's
            # attribute values are in the instance as well, so we will use this rule
            # Get the instance attribute index from the rule's item_ids

            # if decode_rule(r, l3clf)["class"] == instance_predClass:
            # TODO: SAVE PREDICTED CLASS - GENERALIZE FOR ALL CLASSES
            # if rulesAllPredictedClasses:
            # if verbose:
            #     print(
            #         "--- Rule", r, decode_rule(r, l3clf), decode_rule(r, l3clf)["class"]
            #     )

            # if decode_rule(r, l3clf)["class"] == str(instance_predClass):
            if all_rules or decode_rule(r, l3clf)["class"] == str(instance_predClass):

                di = decoded_instance.index
                r_name = list(
                    sorted(
                        [di.get_loc(a) + 1 for a, v in decode_rule(r, l3clf)["body"]]
                    )
                )
                # rules.append(r_name)
                if len(r_name) != len(training_dataset.attributes()):
                    # if verbose:
                    #     print("--- OK")
                    local_rule = LocalRule(
                        r_name,
                        support_count=r.support,
                        support=r.support / k,
                        confidence=r.confidence,
                        rule_class=decode_rule(r, l3clf)["class"],
                    )

                    local_rules.addRule(local_rule)
                    # if verbose:
                    #     local_rule.printRule()
                # else:
                #     if verbose:
                #         print("Rule class", decode_rule(r, l3clf)["class"])
    # Get the union rule
    rules = local_rules.getListRules()
    union_rule_ids = list(sorted(set(itertools.chain.from_iterable(rules))))

    # TODO Add not entire instance!!!
    if (
        union_rule_ids not in rules
        and len(union_rule_ids) > 0
        and len(union_rule_ids) != len(training_dataset.attributes())
    ):
        union_rule = UnionRule(union_rule_ids)
        local_rules.addRule(union_rule)
        if verbose:
            union_rule.printRule()

    return local_rules


# Problema: class --> last


def _create_locality_and_get_rules_OLD(
    training_dataset: Dataset,
    nbrs,
    encoded_instance: pd.Series,
    k: int,
    predict_fn,
    encoder_nn,
    l3clf,
    verbose=False,
    all_rules=False,
):
    from src.local_rules import LocalRule, LocalRules, UnionRule

    cc = training_dataset.class_column_name()
    instance_x = encoded_instance[:-1].to_numpy()
    X_encoder_nn = encoder_nn(instance_x.reshape(1, -1)) if encoder_nn else [instance_x]
    nearest_neighbors_ixs = nbrs.kneighbors(X_encoder_nn, k, return_distance=False)[0]

    classified_instance = deepcopy(encoded_instance)
    import numpy as np

    classified_instance[cc] = np.argmax(predict_fn(instance_x.reshape(1, -1)), axis=1)[
        0
    ]
    classified_instances = [classified_instance]

    for neigh_ix in nearest_neighbors_ixs:
        neigh = training_dataset[neigh_ix]
        neigh_x = neigh[:-1].to_numpy()

        classified_neigh = deepcopy(neigh)

        classified_neigh[cc] = np.argmax(predict_fn(neigh_x.reshape(1, -1)), axis=1)[0]

        classified_instances.append(classified_neigh)

    classified_instances_dataset = Dataset(
        [training_dataset.inverse_transform_instance(c) for c in classified_instances],
        training_dataset.columns,
    )

    l3clf.fit(
        classified_instances_dataset.X_decoded(),
        classified_instances_dataset.Y_decoded(),
        column_names=classified_instances_dataset.X_decoded().columns.to_list(),
    )

    # tmp = pd.DataFrame(classified_instances_dataset.X_decoded())
    # tmp["class"] = classified_instances_dataset.Y_decoded()
    # tmp.to_csv("tmp_old")
    # Drop rules which use values not in the decoded instance
    decoded_instance = training_dataset.inverse_transform_instance(encoded_instance)
    encoded_rules = l3clf.lvl1_rules_

    def decode_rule(r_, clf_):
        r_class = clf_._class_dict[r_.class_id]
        r_attr_ixs_and_values = sorted([clf_._item_id_to_item[i] for i in r_.item_ids])
        r_attrs_and_values = [
            (clf_._column_id_to_name[c], v) for c, v in r_attr_ixs_and_values
        ]
        return {"body": r_attrs_and_values, "class": r_class}

    rules = []

    # Perform matching: remove all rules that use an attibute value not present in the instance to
    # explain

    local_rules = LocalRules()
    # For each rule
    for r in encoded_rules:
        # For each of its attributes and values
        for a, v in decode_rule(r, l3clf)["body"]:
            # If rule uses an attribute's value different from the instance's
            if decoded_instance[a] != v:
                # Exit the inner loop, not entering the else clause, therefore not adding the rule
                break
        # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
        else:
            # If the inner loop has completed normally without break-ing, then all of the rule's
            # attribute values are in the instance as well, so we will use this rule

            # Get the instance attribute index from the rule's item_ids
            # TODO - NEW
            # if decode_rule(r, l3clf)["class"] == decoded_instance.iloc[-1]:
            if all_rules or decode_rule(r, l3clf)["class"] == decoded_instance.iloc[-1]:
                di = decoded_instance.index
                r_name = list(
                    sorted(
                        [di.get_loc(a) + 1 for a, v in decode_rule(r, l3clf)["body"]]
                    )
                )

                local_rule = LocalRule(
                    r_name,
                    support_count=r.support,
                    support=r.support / k,
                    confidence=r.confidence,
                    rule_class=decode_rule(r, l3clf)["class"],
                )
                local_rules.addRule(local_rule)

                if verbose:
                    local_rule.printRule()

    # Get the union rule
    rules = local_rules.getListRules()
    union_rule_ids = list(sorted(set(itertools.chain.from_iterable(rules))))
    if (
        union_rule_ids not in rules
        and len(union_rule_ids) > 0
        and len(union_rule_ids) != len(training_dataset.attributes())
    ):
        union_rule = UnionRule(union_rule_ids)
        local_rules.addRule(union_rule)
        if verbose:
            union_rule.printRule()

    return local_rules


def _compute_perturbed_difference(
    item,
    predict_fn,
    encoded_instance,
    instance_class_index,
    rule_attributes,
    training_dataset,
):
    (attribute_set, occurrences) = item

    perturbed_instance = deepcopy(encoded_instance)
    for i in range(len(rule_attributes)):
        perturbed_instance[rule_attributes[i]] = attribute_set[i]

    prob = predict_fn(perturbed_instance[:-1].to_numpy().reshape(1, -1))[0][
        instance_class_index
    ]

    # Compute the prediction difference using the weighted average of the
    # probability over the frequency of this attribute set in the
    # dataset
    difference = prob * occurrences / len(training_dataset)
    return difference


def _compute_prediction_difference_single(
    encoded_instance,
    predict_fn,
    class_prob,
    target_class_index,
    training_dataset,
    featureMasking=False,
):
    if featureMasking == False:
        return _compute_prediction_difference_single_OLD(
            encoded_instance,
            predict_fn,
            class_prob,
            target_class_index,
            training_dataset,
        )
    else:
        return _compute_prediction_difference_single_NEW(
            encoded_instance,
            predict_fn,
            class_prob,
            target_class_index,
            training_dataset,
        )


# Previous version with discretization
def _compute_prediction_difference_single_OLD(
    encoded_instance,
    predict_fn,
    class_prob_instance,
    target_class_index,
    training_dataset,
):
    attribute_pred_difference = [0] * len(training_dataset.attributes())

    # For each attribute of the instance
    for (attr_ix, (attr, _)) in enumerate(training_dataset.attributes()):
        # Create a dataset containing only the column of the attribute
        filtered_dataset = training_dataset.X()[attr]
        # Count how many times each value of that attribute appears in the dataset
        attr_occurrences = dict(Counter(filtered_dataset).items())
        # For each value of the attribute
        for attr_val in attr_occurrences:
            # Create an instance whose attribute `attr` has that value (`attr_val`)
            perturbed_encoded_instance = deepcopy(encoded_instance)
            perturbed_encoded_instance[attr] = attr_val
            perturbed_encoded_instance_x = perturbed_encoded_instance[:-1].to_numpy()

            # See how the prediction changes
            class_prob = predict_fn(perturbed_encoded_instance_x.reshape(1, -1))[0][
                target_class_index
            ]

            # Update the attribute difference weighting the prediction by the value frequency
            weight = attr_occurrences[attr_val] / len(training_dataset)
            difference = class_prob * weight
            attribute_pred_difference[attr_ix] += difference

    for i in range(len(attribute_pred_difference)):
        attribute_pred_difference[i] = (
            class_prob_instance - attribute_pred_difference[i]
        )
    return attribute_pred_difference


# New version with feature masking
def _compute_prediction_difference_single_NEW(
    encoded_instance, predict_fn, class_prob, target_class_index, training_dataset
):

    P = len(training_dataset.attributes())
    instance_i = deepcopy(encoded_instance)
    instance_i = instance_i[:-1].to_numpy().reshape(1, -1)
    masker_data = training_dataset.X_numpy()
    masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
    mask = np.zeros(P, dtype=np.bool)
    f = predict_fn

    class_prob = predict_fn(instance_i)[0]

    import scipy

    diff_f = np.zeros((P, len(training_dataset.class_values())))
    # For each attribute of the instance
    for i in range(P):
        mask[:] = 1
        mask[i] = 0
        avg_remove_f_i = f(masker(instance_i, mask)).mean(0)

        diff_f[i] = f(instance_i.reshape(1, -1))[0] - avg_remove_f_i

    s = diff_f.shape
    attribute_pred_difference = [np.zeros(s[0]) for j in range(s[1])]
    for j in range(s[1]):
        attribute_pred_difference[j] = diff_f[:, j]

    return list(
        attribute_pred_difference[target_class_index]
    )  # TODO --> save overall!!!


def _compute_prediction_difference_subset(
    training_dataset: Dataset,
    encoded_instance: pd.Series,
    rule_body_indices,
    predict_fn,
    instance_class_index,
    featureMasking=False,
):
    if featureMasking == False:
        return _compute_prediction_difference_subset_OLD(
            training_dataset,
            encoded_instance,
            rule_body_indices,
            predict_fn,
            instance_class_index,
        )
    else:
        return _compute_prediction_difference_subset_NEW(
            training_dataset,
            encoded_instance,
            rule_body_indices,
            predict_fn,
            instance_class_index,
        )


# Previous version with discretization
def _compute_prediction_difference_subset_OLD(
    training_dataset: Dataset,
    encoded_instance: pd.Series,
    rule_body_indices,
    predict_fn,
    instance_class_index,
):
    encoded_instance_x = encoded_instance[:-1].to_numpy()

    rule_attributes = [
        list(training_dataset.attributes())[rule_body_index - 1][0]
        for rule_body_index in rule_body_indices
    ]

    # Take only the considered attributes from the dataset
    filtered_dataset = training_dataset.X()[rule_attributes]

    # Count how many times a set of attribute values appears in the dataset
    attribute_sets_occurrences = dict(
        Counter(map(tuple, filtered_dataset.values.tolist())).items()
    )

    # For each set of attributes
    differences = [
        _compute_perturbed_difference(
            item,
            predict_fn,
            encoded_instance,
            instance_class_index,
            rule_attributes,
            training_dataset,
        )
        for item in attribute_sets_occurrences.items()
    ]

    prediction_difference = sum(differences)

    # p(y=c|x) i.e. Probability that instance x belongs to class c
    p = predict_fn(encoded_instance_x.reshape(1, -1))[0][instance_class_index]
    prediction_differences = p - prediction_difference

    return prediction_differences


# New version with feature masking
def _compute_prediction_difference_subset_NEW(
    training_dataset: Dataset,
    encoded_instance: pd.Series,
    rule_body_indices,
    predict_fn,
    instance_class_index,
):

    P = len(training_dataset.attributes())
    instance_i = deepcopy(encoded_instance)
    instance_i = instance_i[:-1].to_numpy().reshape(1, -1)
    masker_data = training_dataset.X_numpy()
    masker = lambda x, mask: x * mask + masker_data * np.invert(mask)
    mask = np.zeros(P, dtype=np.bool)
    f = predict_fn

    # class_prob=clf.predict_proba(instance_i)[0]

    import scipy

    # prediction_differences={}
    rule = [int(i) - 1 for i in rule_body_indices]
    mask[:] = 1
    mask[rule] = 0
    f_with_i = f(masker(instance_i, mask)).mean(0)

    prediction_difference = f(instance_i.reshape(1, -1))[0] - f_with_i

    return prediction_difference[instance_class_index]


def _compute_approximation_error(
    class_frequency,
    class_prob,
    single_attribute_differences,
    impo_rules_complete,
    difference_map,
    verbose=False,
):
    PI = class_prob - class_frequency
    Sum_Deltas = sum(single_attribute_differences)
    # UPDATED_EP
    # impo_rules_complete : list of all local rule, [[1,2], [5], [1,2,5]]
    # the longest is the union rule --> to do, separate

    union_rule_ids = []
    if len(impo_rules_complete) > 0:
        # get the union rule
        union_rule_ids = list(max(impo_rules_complete, key=len))

    approx_single_d = abs(PI - Sum_Deltas)
    approx_single_rel = approx_single_d / abs(PI)

    approx_eps = 1
    PI_approx = None
    approx_eps_rel = None
    # If the union rule exists, i.e. more than one rule
    if union_rule_ids:
        if len(union_rule_ids) > 1:
            union_rule_ids_str = ",".join(map(str, union_rule_ids))
            Delta_impo_rules_completeC = difference_map[union_rule_ids_str]
            PI_approx = Delta_impo_rules_completeC
            Sum_Deltas_not_in = 0.0
            # Sum of delta_i for each attribute not included
            for id_i_single, diff_i_single in enumerate(single_attribute_differences):

                if (id_i_single + 1) not in union_rule_ids:
                    Sum_Deltas_not_in += diff_i_single

        else:
            index = union_rule_ids[0] - 1
            PI_approx = single_attribute_differences[index]
        approx_eps = abs(PI - PI_approx)
        approx_eps_rel = approx_eps / abs(PI)

    if verbose:
        print("class_prob - class_frequency = PI", class_prob, class_frequency, PI)
        print("PI_approx", PI_approx)

    return PI, approx_single_rel, approx_eps, approx_eps_rel


def _compute_approximation_error_old(
    class_frequency,
    class_prob,
    single_attribute_differences,
    impo_rules_complete,
    difference_map,
):
    PI = class_prob - class_frequency
    Sum_Deltas = sum(single_attribute_differences)
    # UPDATED_EP
    if len(impo_rules_complete) > 0:
        impo_rules_completeC = ", ".join(
            map(str, list(max(impo_rules_complete, key=len)))
        )
    else:
        impo_rules_completeC = ""

    approx_single_d = abs(PI - Sum_Deltas)
    approx_single_rel = approx_single_d / abs(PI)

    if impo_rules_completeC != "":
        if len(impo_rules_completeC.replace(" ", "").split(",")) > 1:
            Delta_impo_rules_completeC = difference_map[
                impo_rules_completeC.replace(" ", "")
            ]
            PI_approx2 = Delta_impo_rules_completeC
            Sum_Deltas_not_in = 0.0
            # Sum of delta_i for each attribute not included
            for i_out_data in range(0, len(single_attribute_differences)):
                if str(i_out_data + 1) not in impo_rules_completeC.replace(
                    " ", ""
                ).split(","):
                    Sum_Deltas_not_in = (
                        Sum_Deltas_not_in + single_attribute_differences[i_out_data]
                    )
        else:
            index = int(impo_rules_completeC.replace(" ", "").split(",")[0]) - 1
            PI_approx2 = single_attribute_differences[index]
        approx2 = abs(PI - PI_approx2)
        approx_rel2 = approx2 / abs(PI)
    else:
        PI_approx2 = 0.0
        approx_rel2 = 1

    approx2 = (
        abs(PI - PI_approx2) if impo_rules_completeC != "" else -1
    )  # Updated 18/11

    return approx_single_rel, approx2, approx_rel2


def _get_KNN_threshold_max(len_dataset):
    import math

    k = int(round(math.sqrt(len_dataset)))

    if len_dataset < 150:
        max_n = len_dataset
    elif len_dataset < 1000:
        max_n = int(len_dataset / 2)
    elif len_dataset < 10000:
        max_n = int(len_dataset / 10)
    else:
        max_n = int(len_dataset * 5 / 100)

    return k, max_n
