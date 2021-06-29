import pickle

# from src.LACE_explainer import *


def savePickle(model, dirO, name):
    import os

    os.makedirs(dirO)
    with open(dirO + "/" + name + ".pickle", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def openPickle(dirO, name):
    from os import path

    if path.exists(dirO + "/" + name + ".pickle"):
        with open(dirO + "/" + name + ".pickle", "rb") as handle:
            return pickle.load(handle)
    else:
        return False


class GlobalExplanation:
    def __init__(self, xplain_obj, explain_dataset):
        import numpy as np

        self.XPLAIN_obj = xplain_obj
        # self.attr_list = getAttrList(self.XPLAIN_obj.explain_dataset[0])
        self.attr_list = self.XPLAIN_obj.train_dataset.attributes_list()
        self.explain_dataset = explain_dataset
        self.predict_fn = self.XPLAIN_obj.predict_fn
        self.global_expl_attribute = None
        self.global_expl_attribute_abs = None
        self.global_expl_info = {
            k: {
                "attribute": np.zeros(len(self.attr_list)),
                "attribute_abs": np.zeros(len(self.attr_list)),
                "attribute_value": {},
                "attribute_value_abs": {},
                "local_rules": {},
                "local_rules_abs": {},
            }
            for k in ["global"] + self.explain_dataset.class_values()
        }

    def predict_class(self, input):
        import numpy as np

        return np.argmax(self.predict_fn(input), axis=1)

    def getGlobalExplanation(self, featureMasking=True, verbose=False, all_rules=False):
        from copy import deepcopy

        import numpy as np

        attr_list = self.attr_list
        cnt = 0
        tot_tb_explained = 5

        total_diff_attribute = {
            k: {
                "attribute": np.zeros(len(self.attr_list)),
                "attribute_abs": np.zeros(len(self.attr_list)),
                "count": 0,
            }
            for k in ["global"] + self.explain_dataset.class_values()
        }

        for id_i in range(0, tot_tb_explained):  # TODO
            instance = self.explain_dataset[id_i]
            instance_discretized = self.explain_dataset.getDiscretizedInstance(id_i)
            instance_x = instance[:-1].to_numpy().reshape(1, -1)
            c = self.predict_class(instance_x)
            print(c)
            target_class = self.explain_dataset.class_values()[c[0].astype(int)]
            print(target_class)
            # e = self.XPLAIN_obj.explain_instance(instance, target_class)
            e = self.XPLAIN_obj.explain_instance(
                instance,
                target_class,
                featureMasking=featureMasking,
                discretizedInstance=instance_discretized,
                verbose=verbose,
                all_rules=all_rules,
            )

            # TODO: for each class!!!

            for type_info in ["global", target_class]:
                total_diff_attribute[type_info]["attribute"] += np.asarray(
                    e.diff_single, dtype=np.float32
                )
                total_diff_attribute[type_info]["attribute_abs"] += np.abs(
                    np.asarray(e.diff_single, dtype=np.float32)
                )
                total_diff_attribute[type_info]["count"] += 1

            # TODO
            ## Version 1:
            # Only for categorical values
            decoded_instance = e.instance

            dict_instance_categorical = {
                i: f"{k}={v}"
                for i, (k, v) in enumerate(decoded_instance[:-1].to_dict().items())
                if k in self.explain_dataset.discrete_attributes
            }
            for type_info in ["global", target_class]:
                for id_attr, attr_value in dict_instance_categorical.items():
                    self.global_expl_info[type_info]["attribute_value"][
                        attr_value
                    ] = self.global_expl_info[type_info]["attribute_value"].get(
                        attr_value, np.zeros(2)
                    ) + [
                        e.diff_single[id_attr],
                        1,
                    ]

                    self.global_expl_info[type_info]["attribute_value_abs"][
                        attr_value
                    ] = self.global_expl_info[type_info]["attribute_value_abs"].get(
                        attr_value, np.zeros(2)
                    ) + [
                        abs(e.diff_single[id_attr]),
                        1,
                    ]

                for local_rule in e.local_rules.rules:

                    self.global_expl_info[type_info]["local_rules"][
                        local_rule.rule_key
                    ] = self.global_expl_info[type_info]["local_rules"].get(
                        attr_value, np.zeros(2)
                    ) + [
                        local_rule.prediction_difference,
                        1,
                    ]

                    self.global_expl_info[type_info]["local_rules_abs"][
                        local_rule.rule_key
                    ] = self.global_expl_info[type_info]["local_rules_abs"].get(
                        attr_value, np.zeros(2)
                    ) + [
                        abs(local_rule.prediction_difference),
                        1,
                    ]

            cnt = cnt + 1

            # TMP - ELIANA 100 INSTANCES
            if cnt == 100:
                break
        # TODO
        for type_info in ["global"] + self.explain_dataset.class_values():
            for type_info_2 in ["attribute", "attribute_abs"]:
                total_diff_attribute[type_info][type_info_2] = (
                    total_diff_attribute[type_info][type_info_2]
                    / total_diff_attribute[type_info]["count"]
                )  # TODO

                self.global_expl_info[type_info][type_info_2] = {
                    attr_list[i]: total_diff_attribute[type_info][type_info_2][i]
                    for i in range(len(self.attr_list))
                }
                self.global_expl_info[type_info][type_info_2] = {
                    attr_list[i]: total_diff_attribute[type_info][type_info_2][i]
                    for i in range(len(self.attr_list))
                }

        for type_info in ["global"] + self.explain_dataset.class_values():
            for type_info_2 in [
                "attribute_value",
                "attribute_value_abs",
                "local_rules",
                "local_rules_abs",
            ]:
                for k, v in self.global_expl_info[type_info][type_info_2].items():
                    self.global_expl_info[type_info][type_info_2][k] = v[0] / v[1]

        return self.global_expl_info

    # def computeGlobalInfo(self):
    #     for key in self.global_expl_info.keys():
    #         dict1 = [self.map_predicted_class[k][key] for k in self.map_predicted_class]
    #         all_keys = set().union(*(d.keys() for d in dict1))
    #         self.global_expl_info[key] = {
    #             k: {
    #                 "d": sum(d[k]["d"] for d in dict1 if k in d),
    #                 "cnt": sum(d[k]["cnt"] for d in dict1 if k in d),
    #             }
    #             for k in all_keys
    #         }
    #         self.global_expl_info[key] = self.computeAvgD_Cnt(
    #             self.global_expl_info[key]
    #         )
    #     return self.global_expl_info

    # def computeAvgD_Cnt(self, mapS):
    #     map_avg = sortByValue(
    #         {
    #             k: (mapS[k]["d"] / mapS[k]["cnt"] if mapS[k]["cnt"] != 0 else 0)
    #             for k in mapS.keys()
    #         }
    #     )
    #     return map_avg

    #     # input: imporules complete, impo_rules, separatore

    # def getRM(self, r, instT, sep=", "):
    #     if type(r) == str:
    #         rule1 = ""
    #         for k in r.split(sep):
    #             rule1 = (
    #                 rule1
    #                 + instT[int(k) - 1].variable.name
    #                 + "="
    #                 + instT[int(k) - 1].value
    #                 + ", "
    #             )
    #         rule1 = rule1[:-2]
    #     return rule1

    # def initD_Cnt(self, r, map_contr):
    #     if r not in map_contr:
    #         map_contr[r] = {"d": 0, "cnt": 0}
    #     return map_contr

    # def updateRuleMapping(
    #     self, map_rules, map_rules_abs, e_map_difference, single_diff, e_instance
    # ):
    #     for k in e_map_difference:
    #         r = self.getRM(k, e_instance, sep=",")
    #         map_rules = self.initD_Cnt(r, map_rules)
    #         map_rules_abs = self.initD_Cnt(r, map_rules_abs)

    #         map_rules[r] = {
    #             "d": map_rules[r]["d"] + e_map_difference[k],
    #             "cnt": map_rules[r]["cnt"] + 1,
    #         }
    #         map_rules_abs[r] = {
    #             "d": map_rules_abs[r]["d"] + abs(e_map_difference[k]),
    #             "cnt": map_rules_abs[r]["cnt"] + 1,
    #         }
    #     return map_rules, map_rules_abs

    # def computeGlobalExplanation_map(self):
    #     self.map_global_info = self.map_predicted_class
    #     self.map_global_info["global"] = self.global_expl_info
    #     return self.map_global_info

    def getGlobalAttributeContributionAbs(self, target_class="global"):
        return self.global_expl_info[target_class]["attribute"]

    def getGlobalAttributeValueContributionAbs(self, k=False, target_class="global"):
        # if k:
        #     return getKMap(self.map_global_info[target_class]["map_avM"], k)
        # return self.map_global_info[target_class]["map_avM"]
        return self.global_expl_info[target_class]["attribute_abs"]

    def getGlobalRuleContributionAbs(self, target_class="global"):
        return self.global_expl_info[target_class]["local_rules_abs"]

    def getRuleMapping(self, map_i, k=False):
        map_i = getKMap(map_i, k) if k else map_i
        mapping_rule = {
            "Rule_" + str(i + 1): key
            for (key, i) in zip(map_i.keys(), range(len(map_i)))
        }
        return mapping_rule


def getAttrList(e_instance):
    attributes_list = []
    for i in e_instance.domain.attributes:
        attributes_list.append(str(i.name))
    return attributes_list


def getAttrValueList(e_instance):
    attributes_list = []
    for i in e_instance.domain.attributes:
        attributes_list.append(str(i.name + "=" + str(e_instance[i])))
    return attributes_list


def sortByValue(map_i):
    return {
        k: v for k, v in sorted(map_i.items(), key=lambda item: item[1], reverse=True)
    }


def getKMap(map_i, k):
    return dict(list(map_i.items())[:k])


# def updateSum_Cnt(map_i, sum_value, absV=False):
#     if absV:
#         return {"d": map_i["d"] + abs(sum_value), "cnt": map_i["cnt"] + 1}
#     else:
#         return {"d": map_i["d"] + sum_value, "cnt": map_i["cnt"] + 1}
