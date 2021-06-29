#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings

CONSIDER_DISCRETE = 20

DATASET_DIR = "./datasets"
warnings.simplefilter("ignore")
from src.dataset import Dataset


def getAttributes(df, ignoreCols=[], featureMasking=True):
    if featureMasking == False:
        return getAttributes_OLD(df, ignoreCols=ignoreCols)
    else:
        return getAttributes_NEW(df, ignoreCols=ignoreCols)


def getAttributes_OLD(df, ignoreCols=[], new=True):
    columns = df.columns.drop(ignoreCols)
    attributes = [(c, list(df[c].unique().astype(str))) for c in columns]
    return attributes


def getAttributes_NEW(df, ignoreCols=[]):
    columns = df.columns.drop(ignoreCols)
    attributes = [
        (c, list(df[c].unique().astype(str)))
        if len(df[c].unique()) <= CONSIDER_DISCRETE
        else (c, [])
        for c in columns
    ]
    return attributes


def importZooDataset(inputDir=DATASET_DIR):
    meta_col = "name"
    import pandas as pd

    df = pd.read_csv(f"{inputDir}/zoo.tab", sep="\t", dtype=str)
    import random

    random.seed = 7
    explain_indices = list(random.sample(range(len(df)), 100))
    df_train = df
    df_explain = df
    attributes = getAttributes(df, ignoreCols=[meta_col])
    d_train = Dataset(
        df_train.drop(columns=meta_col).values, attributes, df_train[[meta_col]].values
    )
    d_explain = Dataset(
        df_explain.drop(columns=meta_col).values,
        attributes,
        df_explain[meta_col].values,
    )
    return d_train, d_explain, [str(i) for i in explain_indices]


def importArff(filename):
    import pandas as pd
    from scipy.io.arff import loadarff

    dataset = pd.DataFrame(loadarff(filename)[0])
    dataset = dataset.stack().str.decode("utf-8").unstack()
    return dataset


# COMPAS
# https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
# Quantize priors count between 0, 1-3, and >3
def quantizePrior(x, n_splits_prior=3):
    if n_splits_prior == 3:
        if x <= 0:
            return "0"
        elif 1 <= x <= 3:
            return "[1-3]"
        else:
            return ">3"
    else:
        if x < 3:
            return "<3"
        elif 3 <= x <= 8:
            return "[3-8]"
        elif 9 <= x <= 15:
            return "[9-15]"
        else:
            return ">15"


# Quantize length of stay
def quantizeLOS(x):
    if x <= 7:
        return "<week"
    if 8 < x <= 93:
        return "1w-3M"
    else:
        return ">3Months"


def get_decile_score_class(x):
    if x >= 8:
        return "High"
    else:
        return "Medium-Low"


def import_process_compas(
    risk_class=True, inputDir=DATASET_DIR, discretize=True, ageTmp=False
):
    import pandas as pd

    df_raw = pd.read_csv(f"{inputDir}/compas-scores-two-years.csv")
    cols_propb = [
        "c_charge_degree",
        "race",
        "age",  # tmp
        "age_cat",
        "sex",
        "priors_count",
        "days_b_screening_arrest",
        "two_year_recid",
    ]  # , "is_recid"]
    if ageTmp == False:
        cols_propb.remove("age")
    cols_propb.sort()
    # df_raw[["days_b_screening_arrest"]].describe()
    df = df_raw[cols_propb]
    # Warning
    df["length_of_stay"] = (
        (
            pd.to_datetime(df_raw["c_jail_out"]).dt.date
            - pd.to_datetime(df_raw["c_jail_in"]).dt.date
        ).dt.days
    ).copy()

    df = df.loc[
        abs(df["days_b_screening_arrest"]) <= 30
    ]  # .sort_values("days_b_screening_arrest")
    # df=df.loc[df["is_recid"]!=-1]
    df = df.loc[df["c_charge_degree"] != "O"]  # F: felony, M: misconduct
    discrete = [
        "age_cat",
        "c_charge_degree",
        "race",
        "sex",
        "two_year_recid",
    ]  # , "is_recid"]
    # continuous = ["days_b_screening_arrest", "priors_count", "length_of_stay"]
    continuous = ["age"] if ageTmp else []
    if discretize:
        df["priors_count"] = df["priors_count"].apply(lambda x: quantizePrior(x))
        df["length_of_stay"] = df["length_of_stay"].apply(lambda x: quantizeLOS(x))
    df = df[discrete + continuous + ["priors_count", "length_of_stay"]]

    if risk_class:
        df["class"] = df_raw["decile_score"].apply(get_decile_score_class)
        df.drop(columns="two_year_recid", inplace=True)
    else:
        df.rename(columns={"two_year_recid": "class"}, inplace=True)

    return df


def KBinsDiscretizer_continuos(
    dt, attributes=None, bins=3, strategy="quantile", round_v=0
):
    import numpy as np

    attributes = dt.columns if attributes is None else attributes
    continuous_attributes = [a for a in attributes if dt.dtypes[a] != np.object]
    X_discretize = dt[attributes].copy()

    for col in continuous_attributes:
        if len(dt[col].value_counts()) >= 10:
            from sklearn.preprocessing import KBinsDiscretizer

            est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
            est.fit(dt[[col]])
            edges = [i.round(round_v) for i in est.bin_edges_][0]
            if round_v == 0:
                edges = [int(i) for i in edges][1:-1]
            else:
                edges = edges[1:-1]
            if len(set(edges)) != len(edges):
                edges = [
                    edges[i]
                    for i in range(0, len(edges))
                    if len(edges) - 1 == i or edges[i] != edges[i + 1]
                ]
            for i in range(0, len(edges)):
                if i == 0:
                    data_idx = dt.loc[dt[col] <= edges[i]].index
                    X_discretize.loc[data_idx, col] = f"<={edges[i]}"
                if i == len(edges) - 1:
                    data_idx = dt.loc[dt[col] > edges[i]].index
                    X_discretize.loc[data_idx, col] = f">{edges[i]}"

                data_idx = dt.loc[
                    (dt[col] > edges[i - 1]) & (dt[col] <= edges[i])
                ].index
                X_discretize.loc[data_idx, col] = f"({edges[i-1]}-{edges[i]}]"
        else:
            X_discretize[col] = X_discretize[col].astype("object")
    return X_discretize


def cap_gains_fn(x):
    import numpy as np

    x = x.astype(float)
    d = np.digitize(
        x, [0, np.median(x[x > 0]), float("inf")], right=True
    )  # .astype('|S128')
    return d.copy()


def discretize(
    dfI,
    bins=4,
    dataset_name=None,
    attributes=None,
    indexes_FP=None,
    n_splits_prior=3,
    strategy="quantile",
    round_v=0,
):

    indexes_validation = dfI.index if indexes_FP is None else indexes_FP
    attributes = dfI.columns if attributes is None else attributes
    if dataset_name == "compas":
        X_discretized = dfI[attributes].copy()
        X_discretized["priors_count"] = X_discretized["priors_count"].apply(
            lambda x: quantizePrior(x, n_splits_prior)
        )
        X_discretized["length_of_stay"] = X_discretized["length_of_stay"].apply(
            lambda x: quantizeLOS(x)
        )
    elif dataset_name == "adult":
        X_discretized = dfI[attributes].copy()
        X_discretized["capital-gain"] = cap_gains_fn(
            X_discretized["capital-gain"].values
        )
        X_discretized["capital-gain"] = X_discretized["capital-gain"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )
        X_discretized["capital-loss"] = cap_gains_fn(
            X_discretized["capital-loss"].values
        )
        X_discretized["capital-loss"] = X_discretized["capital-loss"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )
        X_discretized = KBinsDiscretizer_continuos(
            X_discretized, attributes, bins=bins, strategy=strategy
        )
    else:
        X_discretized = KBinsDiscretizer_continuos(
            dfI, attributes, bins=bins, strategy=strategy, round_v=round_v
        )
    # TODO why reset index??
    return X_discretized.loc[indexes_validation]  # .reset_index(drop=True)


def import_process_adult(discretize=False, bins=3, inputDir=DATASET_DIR):
    education_map = {
        "10th": "Dropout",
        "11th": "Dropout",
        "12th": "Dropout",
        "1st-4th": "Dropout",
        "5th-6th": "Dropout",
        "7th-8th": "Dropout",
        "9th": "Dropout",
        "Preschool": "Dropout",
        "HS-grad": "High School grad",
        "Some-college": "High School grad",
        "Masters": "Masters",
        "Prof-school": "Prof-School",
        "Assoc-acdm": "Associates",
        "Assoc-voc": "Associates",
    }
    occupation_map = {
        "Adm-clerical": "Admin",
        "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar",
        "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Prof-specialty": "Professional",
        "Protective-serv": "Other",
        "Sales": "Sales",
        "Tech-support": "Other",
        "Transport-moving": "Blue-Collar",
    }
    married_map = {
        "Never-married": "Never-Married",
        "Married-AF-spouse": "Married",
        "Married-civ-spouse": "Married",
        "Married-spouse-absent": "Separated",
        "Separated": "Separated",
        "Divorced": "Separated",
        "Widowed": "Widowed",
    }

    country_map = {
        "Cambodia": "SE-Asia",
        "Canada": "British-Commonwealth",
        "China": "China",
        "Columbia": "South-America",
        "Cuba": "Other",
        "Dominican-Republic": "Latin-America",
        "Ecuador": "South-America",
        "El-Salvador": "South-America",
        "England": "British-Commonwealth",
        "France": "Euro_1",
        "Germany": "Euro_1",
        "Greece": "Euro_2",
        "Guatemala": "Latin-America",
        "Haiti": "Latin-America",
        "Holand-Netherlands": "Euro_1",
        "Honduras": "Latin-America",
        "Hong": "China",
        "Hungary": "Euro_2",
        "India": "British-Commonwealth",
        "Iran": "Other",
        "Ireland": "British-Commonwealth",
        "Italy": "Euro_1",
        "Jamaica": "Latin-America",
        "Japan": "Other",
        "Laos": "SE-Asia",
        "Mexico": "Latin-America",
        "Nicaragua": "Latin-America",
        "Outlying-US(Guam-USVI-etc)": "Latin-America",
        "Peru": "South-America",
        "Philippines": "SE-Asia",
        "Poland": "Euro_2",
        "Portugal": "Euro_2",
        "Puerto-Rico": "Latin-America",
        "Scotland": "British-Commonwealth",
        "South": "Euro_2",
        "Taiwan": "China",
        "Thailand": "SE-Asia",
        "Trinadad&Tobago": "Latin-America",
        "United-States": "United-States",
        "Vietnam": "SE-Asia",
    }
    # as given by adult.names
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income-per-year",
    ]
    import pandas as pd
    import os

    train = pd.read_csv(
        os.path.join(inputDir, "adult.data"),
        header=None,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )

    test = pd.read_csv(
        os.path.join(inputDir, "adult.test"),
        header=0,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )
    dt = pd.concat([test, train], ignore_index=True)
    dt["education"] = dt["education"].replace(education_map)
    dt.drop(columns=["education-num", "fnlwgt"], inplace=True)
    dt["occupation"] = dt["occupation"].replace(occupation_map)
    dt["marital-status"] = dt["marital-status"].replace(married_map)
    dt["native-country"] = dt["native-country"].replace(country_map)

    dt.rename(columns={"income-per-year": "class"}, inplace=True)
    dt["class"] = (
        dt["class"].astype("str").replace({">50K.": ">50K", "<=50K.": "<=50K"})
    )
    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)
    if discretize:
        dt = KBinsDiscretizer_continuos(dt, bins=bins)
    dt.drop(columns=["native-country"], inplace=True)
    return dt
