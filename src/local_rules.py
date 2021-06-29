class LocalRules:
    def __init__(self, rules=[]):
        self.rules = rules if rules else []
        self.rules_df = None
        self.pi = None

    # target difference
    def setPi(self, pi):
        self.pi = pi

    def addRule(self, local_rule):
        self.rules.append(local_rule)

    def getListRules(self):
        return [local_rule.rule for local_rule in self.rules]

    def getDictHRRules(self):
        return {
            frozenset(local_rule.rule_hr): local_rule.prediction_difference
            for local_rule in self.rules
        }

    def getDataframeRule(self):
        if self.rules_df is None:
            rules_app = []
            for local_rule in self.rules:
                rules_app.append(
                    [
                        frozenset(local_rule.rule_hr),
                        local_rule.rule_class,
                        local_rule.prediction_difference,
                        local_rule.support_count,
                        local_rule.support,
                        local_rule.confidence,
                        local_rule.rule_key,
                        self.pi - local_rule.prediction_difference,
                    ]
                )
            import pandas as pd

            self.rules_df = pd.DataFrame(
                rules_app,
                columns=[
                    "rule",
                    "rule_class",
                    "prediction_difference",
                    "support_count",
                    "support",
                    "confidence",
                    "rule_id",
                    "eps_error",  # pi_approx_rule
                ],
            )
        return self.rules_df

    def getPredictionDifferenceMap(self):
        return {
            local_rule.rule_key: local_rule.prediction_difference
            for local_rule in self.rules
        }

    # TODO Why sorted?
    def printLocalRules(self):
        # TODO TMP z
        self.rules.sort(
            key=lambda x: x.rule_key if isinstance(x, UnionRule) == False else "z"
        )
        for local_rule in self.rules:
            local_rule.printRule()

    def getHRLocalRule(self):
        return {local_rule.rule_id: local_rule.getHRRule() for local_rule in self.rules}

    # TODO Why sorted?
    def setRulesId(self):
        # TODO TMP z
        self.rules.sort(
            key=lambda x: x.rule_key if isinstance(x, UnionRule) == False else "z"
        )
        for i, local_rule in enumerate(self.rules):
            if isinstance(local_rule, UnionRule):
                local_rule.setRuleId("U")
            else:
                local_rule.setRuleId(i + 1)

    def setRulesHR(self, dict_instance_x):
        for local_rule in self.rules:
            local_rule.setRuleHR(dict_instance_x)

    def getUnionRule(self):
        for local_rule in self.rules:
            if type(local_rule) is UnionRule:
                return local_rule
        return None

    def getMaximalRule(self):
        if self.rules is None:
            return None
        rule_ids_dict = {local_rule: local_rule.rule for local_rule in self.rules}
        maximal_rule = max(rule_ids_dict, key=lambda k: len(rule_ids_dict[k]))

        union_rule = self.getUnionRule()
        if union_rule is not None:
            # TODO REMOVE TMP
            if union_rule != maximal_rule:
                raise Warning("Union and maximal are different")
        return maximal_rule


class LACERule:
    # TODO Add prediction difference in the constructor
    def __init__(
        self,
        rule,
    ):
        self.rule = rule
        self.rule_key = ",".join(map(str, rule))

        self.prediction_difference = None
        self.rule_id = None
        # List of attributes of the rule
        self.rule_hr = None
        self.rule_class = None
        self.support_count = None
        self.support = None
        self.confidence = None
        self.epsilon = None

    def __len__(self):
        return len(self.rule)

    def setPredictionDifference(self, prediction_difference):
        self.prediction_difference = prediction_difference

    def setRuleHR(self, dict_instance_x):
        self.rule_hr = [dict_instance_x[id_attr] for id_attr in self.rule]


class LocalRule(LACERule):
    def __init__(
        self, rule, support_count=None, support=None, confidence=None, rule_class=None
    ):
        super().__init__(rule)

        self.support_count = support_count
        self.support = support
        self.confidence = confidence
        self.rule_class = rule_class

    def getHRRule(self):
        return (self.rule_hr, "-->", self.rule_class)

    def printRule(self):

        print(
            self.rule_hr,
            "-->",
            self.rule_class,
            self.support_count,
            "support:",
            self.support,
            "confidence:",
            self.confidence,
            self.rule,
        )

    def setRuleId(self, rule_id):
        self.rule_id = f"Rule_{rule_id}"


class UnionRule(LACERule):
    def __init__(self, rule):
        super().__init__(rule)

    def printRule(self):
        print("Union rule", self.rule)

    def setRuleId(self, rule_id):
        self.rule_id = f"Rule_{rule_id}"

    def getHRRule(self):
        return (self.rule_hr, "U")


class UserRule(LACERule):
    def __init__(self, rule):
        super().__init__(rule)

    def printRule(self):
        print("User rule", self.rule)

    def printHRRule(self):
        print("User rule", self.rule_id, self.rule_hr)

    def setRuleId(self, rule_id):
        self.rule_id = f"User_Rule_{rule_id}"

    def getHRRule(self):
        return (self.rule_hr, "User")
