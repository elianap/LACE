def updateKeyListDictionary(keys_mapping, values, dictionaryByClass):
    for i, k in enumerate(keys_mapping):
        if k not in dictionaryByClass:
            dictionaryByClass[k] = {"sum": 0, "count": 0}
        dictionaryByClass[k]["sum"] += values[i]
        dictionaryByClass[k]["count"] += 1


def computeGlobalAverage(summaryByClass):
    averagesByClass = {}
    for class_name, dict_diff in summaryByClass.items():
        averagesByClass[class_name] = {
            key: list_diff["sum"] / list_diff["count"]
            for key, list_diff in dict_diff.items()
        }
    return averagesByClass


def addTotalColumns(summaryByClass):
    summaryByClass["total"] = {}
    for class_name, dict_diff in summaryByClass.items():
        if class_name != "total":
            for key, summary_diff in dict_diff.items():
                if key not in summaryByClass["total"]:
                    summaryByClass["total"][key] = {"sum": 0, "count": 0}
                summaryByClass["total"][key]["sum"] += summary_diff["sum"]
                summaryByClass["total"][key]["count"] += summary_diff["count"]
