"""This module provides the Dataset class, which is used to describe your
dataset in a format suitable for the $name analysis.

"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Dataset:
    """The Dataset class contains information on a dataset and all of the
    possible values of every one of each columns.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Anything that can be converted to a pandas DataFrame.

        This data, after being converted into a DataFrame, has every row processed by
        a scikit-learn LabelEncoder.

        An example:
        ::
            [
                ['1','0','0','1','0','0','1','1','1','1','0','0','4','0','0','1','mammal'],
                ['1','0','0','1','0','0','0','1','1','1','0','0','4','1','0','1','mammal']
            ]
    columns : list
        A list of tuples. Each tuple corresponds to one of the columns of `data`
        and contains its name and a list of **all** its possible values.

        An example (all of the examples will use the UCI ML Zoo dataset):
        ::

            [
                ('hair', ['0', '1']),
                ('feathers', ['0', '1']),
                ('eggs', ['0', '1']),
                ('milk', ['0', '1']),
                ('airborne', ['0', '1']),
                ('aquatic', ['0', '1']),
                ('predator', ['0', '1']),
                ('toothed', ['0', '1']),
                ('backbone', ['0', '1']),
                ('breathes', ['0', '1']),
                ('venomous', ['0', '1']),
                ('fins', ['0', '1']),
                ('legs', ['0', '2', '4', '5', '6', '8']),
                ('tail', ['0', '1']),
                ('domestic', ['0', '1']),
                ('catsize', ['0', '1']),
                ('type', ['amphibian','bird','fish','insect','invertebrate','mammal','reptile'])
            ]

    Attributes
    ----------
    columns : list
        The same `columns` object passed in the constructor


    """

    @classmethod
    def from_indices(cls, indices: [int], other):
        return cls(other._decoded_df.copy().iloc[indices], other.columns)

    # TODO preserve indexes!!!!
    def __init__(
        self,
        data,
        columns,
        metas=None,
        column_encoders=None,
        encoder_nn=None,
        featureMasking=False,
        discreteDataset=None,
        class_col_name="class",
        verbose=False,
    ):
        fitEncoders = True if column_encoders is None else False

        # self._decoded_df = data  # modified Eliana 12/4 pd.DataFrame(data)

        self._decoded_df = pd.DataFrame(data)  # , index=data.index)
        # self._decoded_df = data

        self.continuos_attributes = [k for k, v in columns if v == []]
        self.discrete_attributes = [k for k, v in columns if v != []]

        # Rename columns from 0,1,... to the attributes[0,1,...][0]
        columns_mapper = {i: a for (i, a) in enumerate([a for (a, _) in columns])}
        # Note done.. (revers v,k)
        self._decoded_df = self._decoded_df.rename(columns=columns_mapper)

        if list(columns_mapper.keys())[-1] != class_col_name:
            cols = list(self._decoded_df.columns)  # columns_mapper
            cols.remove(class_col_name)
            cols.append(class_col_name)

            self._decoded_df = self._decoded_df[cols]

            class_name_values = [(c, vs) for c, vs in columns if c == class_col_name][0]
            columns.remove(class_name_values)
            columns.append(class_name_values)

        # dict_columns = {k: v for (k, v) in self.columns}

        # Encode categorical columns with value between 0 and n_classes-1
        # Keep the columns encoders used to perform the inverse transformation
        # https://stackoverflow.com/a/31939145
        def funcX(x, columns, featureMasking=False, verbose=False):

            if featureMasking:
                if x.name in self.continuos_attributes:
                    return x
            if fitEncoders:
                self._column_encoders[x.name].fit(self._decoded_df[x.name])
            return self._column_encoders[x.name].transform(x)

        if fitEncoders:
            self._column_encoders = defaultdict(LabelEncoder, verbose=verbose)
        else:
            self._column_encoders = column_encoders

        # Label encoding of the dataset
        self._encoded_df = self._decoded_df.copy().apply(
            lambda x: funcX(x, columns, featureMasking)
        )
        self.columns = [
            (k, list(self._column_encoders[k].classes_)) if v != [] else (k, [])
            for k, v in columns
        ]
        # self.columns = [(k, list(v.classes_)) for k, v in self._column_encoders.items()]

        # columns_mapper = {i: a for (i, a) in enumerate([a for (a, _) in self.columns])}
        # self._decoded_df = self._decoded_df.rename(columns=columns_mapper)

        # dict_columns = {k: v for (k, v) in self.columns}
        # self._encoded_df = self._encoded_df.rename(columns=columns_mapper)
        # self._encoded_df.columns=dict_columns.keys()

        self.metas = metas

        # Assume that already discretized
        self.discreteDataset = (
            self.X_decoded().astype(object)
            if discreteDataset is None
            else discreteDataset.reset_index(drop=True)
        )  # TODO
        # self.columns = columns
        self.encoder_nn = encoder_nn

    def getEncoders(self):
        return self._column_encoders

    def decodeAttribute(self, value, attribute):
        if attribute in self._column_encoders:
            if type(value) not in [np.ndarray, list]:
                value = [value]
            return self._column_encoders[attribute].inverse_transform(value)
        return value

    def encodeAttribute(self, value, attribute):
        if attribute in self._column_encoders:
            if type(value) not in [np.ndarray, list]:
                value = [value]
            return self._column_encoders[attribute].transform(value)
        return value

    def X_numpy_NN(self):
        return self.encoder_nn(self.X_numpy()) if self.encoder_nn else self.X_numpy()

    def class_values(self):
        """All the possible classes of an instance of the dataset
        ::

            In[35]: d.class_values()
            Out[35]: ['amphibian', 'bird', 'fish', 'insect', 'invertebrate', 'mammal', 'reptile']
        """
        return list(self.columns[-1][1])

    def X(self):
        """All rows' attributes as a pandas DataFrame. These attributes were
        encoded with scikit-learn's Label Encoder. See `X_decoded()` to get
        the original data.
        ::

            In[28]: d.X()
            Out[28]:
               hair  feathers  eggs  milk  airborne  ...  fins  legs  tail  domestic  catsize
            0     1         0     0     1         0  ...     0     2     0         0        1
            1     1         0     0     1         0  ...     0     2     1         0        1
        """
        return self._encoded_df.iloc[:, :-1]

    def Y(self):
        """All rows' classes as a pandas DataFrame. these classes were
        encoded with scikit-learn's Label Encoder. See `Y_decoded()` to get
        the original data.
        ::

            In[32]: d.Y()
            Out[32]:
            0    5
            1    5
            Name: type, dtype: int64
        """
        return self._encoded_df.iloc[:, -1]

    def X_decoded(self):
        """All rows' attributes as a pandas DataFrame.
        ::

            d.X_decoded()
            Out[29]:
              hair feathers eggs milk airborne  ... fins legs tail domestic catsize
            0    1        0    0    1        0  ...    0    4    0        0       1
            1    1        0    0    1        0  ...    0    4    1        0       1
        """
        return self._decoded_df.iloc[:, :-1]

    def Y_decoded(self):
        """All rows' classes as a pandas DataFrame.
        ::

            In[33]: d.Y_decoded()
            Out[33]:
            0    mammal
            1    mammal
            Name: type, dtype: object
        """
        return self._decoded_df.iloc[:, -1]

    def X_numpy(self):
        """All encoded rows' attributes as a numpy float64 array.
        ::

            In[30]: d.X_numpy()
            Out[30]:
            array([[1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 2., 0., 0., 1.],
                   [1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 2., 1., 0., 1.]])
        """

        return self._encoded_df.iloc[:, :-1].to_numpy().astype(np.float64)

    def Y_numpy(self):
        """All rows' classes as a numpy float64 array.
        ::

            In[34]: d.Y_numpy()
            Out[34]: array([5., 5.])
        """

        return self._encoded_df.iloc[:, -1].to_numpy().astype(np.float64)

    def attributes(self):
        return self.columns[:-1]

    def attributes_list(self):
        return [k1 for k1, k2 in self.columns[:-1]]

    def class_column_name(self):
        """The column name of the class attribute
        ::

            In[36]: d.class_column_name()
            Out[36]: 'type'
        """
        return self.columns[-1][0]

    def __len__(self):
        return len(self._decoded_df)

    def __getitem__(self, item) -> pd.Series:
        """Returns the i-th element of the encoded DataFrame of datset"""
        return self._encoded_df.iloc[item]

    def getDiscretizedInstance(self, item):
        return self.discreteDataset.iloc[item]

    def get_decoded(self, item) -> pd.Series:
        """Returns the i-th element of the decoded DataFrame of datset"""
        return self._decoded_df.iloc[item]

    def transform_instance(self, decoded_instance: pd.Series) -> pd.Series:
        """Transform a decoded instance to an encoded instance using the Dataset's column encoders"""
        return pd.Series(
            {
                col: self._column_encoders[col].transform([val])[0]
                for (col, val) in decoded_instance.items()
            }
        )

    def inverse_transform_instance(
        self, encoded_instance: pd.Series, featureMasking=False
    ) -> pd.Series:
        if featureMasking:
            # TODO: to int??
            # TODO TODO TODO CHECK
            return pd.Series(
                {
                    col: (
                        # updated 12/04
                        # self._column_encoders[col].inverse_transform([val])[0]
                        self._column_encoders[col].inverse_transform([int(val)])[0]
                        if col not in self.continuos_attributes
                        else val
                    )
                    for (col, val) in encoded_instance.items()
                }
            )
        return pd.Series(
            {
                col: self._column_encoders[col].inverse_transform([val])[0]
                for (col, val) in encoded_instance.items()
            }
        )

    def to_arff_obj(self) -> object:
        obj = {
            "relation": self.class_column_name(),
            "attributes": self.columns,
            "data": self._decoded_df.values.tolist(),
        }
        return obj

    def lenAttributes(self):
        return len(self.attributes())
