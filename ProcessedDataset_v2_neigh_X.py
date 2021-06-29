# TODO Problem with int a string
class ProcessedDatasetTrainTest:
    def __init__(self, df_train, df_test, meta_col=None, dataset_name=""):
        self.df_train = df_train
        self.df_test = df_test
        self.meta_col = meta_col
        self.dataset_name = dataset_name

        self.OH_X_test_cols = None
        self.feature_names = None
        self.class_names = None
        self.categorical_features_pos = None
        self.categorical_names_LE = None
        self.train = None
        self.test = None
        self.labels_test = None

        self.d_train = None
        self.d_explain = None

        self.predict_fn = None
        self.predict_fn_class = None
        self.clf = None

        self.OH_X_train = None

    def processTrainTestDataset(
        self,
        clf,
        discretized_dataset=None,
        featureMasking=True,
        dataset_name="",
        class_col_name="class",
        discr_par={"round_v": 0, "bins": 4, "strategy": "quantile"},
        verbose=False,
    ):
        df_train = self.df_train
        df_test = self.df_test
        meta_col = self.meta_col
        cols_to_drop = (
            [class_col_name] if meta_col is None else [class_col_name, meta_col]
        )

        feature_names = df_train.columns.drop(cols_to_drop)
        (
            categorical_features,
            categorical_features_pos,
            continuos_features,
            continuos_features_pos,
        ) = getCategoricalAndContinuous(df_train, feature_names)
        labels_train, class_names, le = getClassLabelEncoding(
            df_train[class_col_name].values
        )
        data_LE_train, categorical_names_LE, encoders = getLabelEncodingMapping(
            df_train, feature_names, categorical_features_pos
        )

        train = data_LE_train[feature_names].copy()
        from copy import deepcopy

        data_LE_test = encodeAttributes(deepcopy(df_test[feature_names]), encoders)

        test = data_LE_test[feature_names].copy()
        labels_test = le.transform(df_test[class_col_name].values)

        OH_X_train, encoder, min_max_scaler = oneHotEncodingNormalize(
            train, categorical_features
        )

        clf.fit(OH_X_train.values, labels_train)

        predict_fn = getPredictFunction(
            clf,
            categorical_features_pos,
            continuos_features_pos,
            encoder=encoder,
            min_max_scaler=min_max_scaler,
        )
        if verbose:
            print("feature_names", feature_names)
            print("categorical_features", categorical_features)
            print("continuos_features", continuos_features)
            print("encoder", encoder)

        OH_X_test, _, _ = oneHotEncodingNormalize(
            test,
            categorical_features,
            encoder=encoder,
            min_max_scaler=min_max_scaler,
        )
        categorical_names_map = {
            feature_names[k]: val for k, val in categorical_names_LE.items()
        }
        oh_columns = list(OH_X_test.columns)
        # oh_columns_categorical = [
        #     f'{"_".join(c.split("_")[0:-1])}_{categorical_names_map["_".join(c.split("_")[0:-1])][int(float(c.split("_")[-1]))]}'
        #     for c in oh_columns
        #     if "_".join(c.split("_")[0:-1]) in categorical_features
        # ]
        oh_columns = [
            f'{"_".join(c.split("_")[0:-1])}_{categorical_names_map["_".join(c.split("_")[0:-1])][int(float(c.split("_")[-1]))]}'
            if "_".join(c.split("_")[0:-1]) in categorical_features
            else c
            for c in oh_columns
        ]

        OH_X_test_cols = OH_X_test.copy()
        OH_X_test_cols.columns = oh_columns

        predict_fn_class = getPredictClassFunction(
            clf,
            categorical_features_pos,
            continuos_features_pos,
            encoder=encoder,
            min_max_scaler=min_max_scaler,
        )
        # Label encoding for categorical + discretized for continuos (result as a string type)
        # TODO when used?
        test_discretized = test.copy()
        test_discretized.reset_index(drop=True, inplace=True)
        if continuos_features:
            from src.import_datasets import discretize

            test_discretized[continuos_features] = discretize(
                test_discretized,
                attributes=continuos_features,
                dataset_name=self.dataset_name,
                round_v=discr_par["round_v"],
                bins=discr_par["bins"],
                strategy=discr_par["strategy"],
            )
            test_discretized.head()

        #### LACE
        from copy import deepcopy

        # TODO: only train or both??????
        # TODO separate for train and test
        # data_X_discretized = deepcopy(df[feature_names])
        data_X_discretized = deepcopy(df_train[feature_names]).append(
            deepcopy(df_test[feature_names])
        )
        if continuos_features:
            data_X_discretized[continuos_features] = discretize(
                data_X_discretized[continuos_features],
                attributes=continuos_features,
                dataset_name=dataset_name,
                round_v=discr_par["round_v"],
                bins=discr_par["bins"],
                strategy=discr_par["strategy"],
            )
        # data_X_discretized.reset_index(drop=True, inplace=True)

        encoder_t = getEncoder(
            categorical_features_pos,
            continuos_features_pos,
            encoder=encoder,
            min_max_scaler=min_max_scaler,
        )

        if verbose:
            print("encoder_t", encoder_t)

        from src.import_datasets import getAttributes

        ignoreCols = [] if meta_col is None else [meta_col]
        attributes = getAttributes(
            df_train, featureMasking=featureMasking, ignoreCols=ignoreCols
        )
        from src.dataset_neigh_X import Dataset
        from copy import deepcopy

        # train_orig = df_train.iloc[train.index].copy()
        # test_orig = df_train.iloc[test.index].copy()
        train_orig = df_train.copy()
        test_orig = df_test.copy()

        all_encoders = deepcopy(encoders)
        all_encoders.update({"class": le})
        self.d_train = Dataset(
            train_orig.drop(columns=meta_col).values
            if meta_col != None
            else train_orig.values,
            deepcopy(attributes),
            column_encoders=all_encoders,
            featureMasking=True,
            discreteDataset=data_X_discretized.loc[train.index],
            encoder_nn=encoder_t,
        )
        self.d_explain = Dataset(
            test_orig.drop(columns=meta_col).values
            if meta_col != None
            else test_orig.values,
            deepcopy(attributes),
            column_encoders=all_encoders,
            featureMasking=True,
            discreteDataset=data_X_discretized.loc[test.index],
            encoder_nn=encoder_t,
        )

        self.predict_fn_class = predict_fn_class
        self.OH_X_train = OH_X_train
        self.OH_X_test_cols = OH_X_test_cols
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_features_pos = categorical_features_pos
        self.categorical_names_LE = categorical_names_LE
        self.train = train
        self.test = test
        self.labels_test = labels_test
        self.categorical_features = categorical_features
        self.continuos_features = continuos_features

        self.clf = clf
        self.predict_fn = predict_fn
        self.oh_columns = oh_columns

        self.encoder_t = encoder_t
        self.encoder = encoder
        self.encoders = encoders

    def getExplainDataLE(self):
        return self.test

    def getExplainLabels(self):
        return self.labels_test

    def getPredict_fn(self):
        return self.predict_fn

    def getPredict_fn_class(self):
        return self.predict_fn_class

    def getExplainOneHotNormalized(self):
        return self.OH_X_test_cols


def SplitTrainTest(data_LE, feature_names, labels, all_data=False):
    if all_data:
        train = data_LE[feature_names].copy()
        test = data_LE[feature_names].copy()
        labels_train = labels
        labels_test = labels
    else:
        from sklearn import model_selection
        import numpy as np

        np.random.seed(1)
        train, test, labels_train, labels_test = model_selection.train_test_split(
            data_LE[feature_names], labels, train_size=0.8
        )
    return train, test, labels_train, labels_test


def getCategoricalAndContinuous(df, feature_names):
    categorical_features = list(
        df[feature_names].select_dtypes(include=["object", "category"]).columns
    )
    categorical_features_pos = [
        i for i, k in enumerate(feature_names) if k in categorical_features
    ]
    continuos_features = list(set(feature_names) - set(categorical_features))
    continuos_features_pos = [
        i for i, k in enumerate(feature_names) if k in continuos_features
    ]

    return (
        categorical_features,
        categorical_features_pos,
        continuos_features,
        continuos_features_pos,
    )


def getClassLabelEncoding(labels):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    class_names = le.classes_
    return labels, class_names, le


def getLabelEncodingMapping(df, feature_names, categorical_features_pos):
    from sklearn.preprocessing import LabelEncoder

    data_LE = df.copy()
    categorical_names = {}
    encoders = {}
    for i in categorical_features_pos:
        feature = feature_names[i]
        le = LabelEncoder()
        le.fit(data_LE[feature].values)
        data_LE[feature] = le.transform(data_LE[feature].values).astype(
            df[feature].dtype
        )
        categorical_names[i] = le.classes_
        encoders[feature] = le
    return data_LE, categorical_names, encoders


def oneHotEncodingNormalize(
    data, categorical_features=[], encoder=None, min_max_scaler=None
):
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

    if encoder is None:
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(data[categorical_features])

    df_X_encoded = pd.DataFrame(
        encoder.transform(data[categorical_features]), index=data.index
    )
    df_X_encoded.columns = encoder.get_feature_names(categorical_features)
    if df_X_encoded.shape[1] != 0:
        OH_X_data = df_X_encoded
    df_X_continuos = data.drop(categorical_features, axis=1)

    if df_X_continuos.shape[1] != 0:
        x_cont = df_X_continuos.copy()

        if min_max_scaler is None:
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(x_cont.values)

        x_cont_scaled = min_max_scaler.transform(x_cont.values)
        df_X_continuos_scaled = pd.DataFrame(
            x_cont_scaled, columns=df_X_continuos.columns, index=data.index
        )
        if df_X_encoded.shape[1] != 0:
            OH_X_data = pd.concat([df_X_encoded, df_X_continuos_scaled], axis=1)
        else:
            OH_X_data = df_X_continuos_scaled

    return OH_X_data, encoder, min_max_scaler


def getPredictFunction(
    clf,
    categorical_features_pos,
    continuos_features_pos,
    encoder=None,
    min_max_scaler=None,
):
    import numpy as np

    if continuos_features_pos and categorical_features_pos:
        predict_fn = lambda x: clf.predict_proba(
            np.hstack(
                (
                    encoder.transform(x[:, categorical_features_pos]),
                    min_max_scaler.transform(x[:, continuos_features_pos]),
                )
            )
        )
        # x[:,continuos_features_pos].astype(float)))    )
    elif categorical_features_pos:
        predict_fn = lambda x: clf.predict_proba(
            encoder.transform(x[:, categorical_features_pos])
        )
    elif continuos_features_pos:
        predict_fn = lambda x: clf.predict_proba(
            min_max_scaler.transform(x[:, continuos_features_pos])
        )
    else:
        raise ValueError("Empty dataset")
    return predict_fn


def getPredictClassFunction(
    clf,
    categorical_features_pos,
    continuos_features_pos,
    encoder=None,
    min_max_scaler=None,
):
    import numpy as np

    if continuos_features_pos and categorical_features_pos:
        predict_fn_class = lambda x: clf.predict(
            np.hstack(
                (
                    encoder.transform(x[:, categorical_features_pos]),
                    min_max_scaler.transform(x[:, continuos_features_pos]),
                )
            )
        )
        # x[:,continuos_features_pos].astype(float)))    )
    elif categorical_features_pos:
        predict_fn_class = lambda x: clf.predict(
            encoder.transform(x[:, categorical_features_pos])
        )
    elif continuos_features_pos:
        predict_fn_class = lambda x: clf.predict(
            min_max_scaler.transform(x[:, continuos_features_pos])
        )
    else:
        raise ValueError("Empty dataset")
    return predict_fn_class


def getEncoder(
    categorical_features_pos, continuos_features_pos, encoder=None, min_max_scaler=None
):
    if continuos_features_pos and categorical_features_pos:
        import numpy as np

        encoder_t = lambda x: np.hstack(
            (
                encoder.transform(x[:, categorical_features_pos]),
                min_max_scaler.transform(x[:, continuos_features_pos]),
            )
        )
        # x[:,continuos_features_pos].astype(float)))    )
    elif categorical_features_pos:
        encoder_t = lambda x: encoder.transform(x[:, categorical_features_pos])
    elif continuos_features_pos:
        encoder_t = lambda x: min_max_scaler.transform(x[:, continuos_features_pos])
    else:
        raise ValueError("Empty dataset")
    return encoder_t


def encodeAttributes(df_t, encoders):
    for attr in encoders:
        df_t[attr] = encoders[attr].transform(df_t[attr])
    return df_t