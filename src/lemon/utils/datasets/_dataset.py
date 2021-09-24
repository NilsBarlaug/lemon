from collections import namedtuple
from typing import Tuple, Union

import pandas as pd

RecordsLike = Union["Records", Tuple[pd.DataFrame, pd.DataFrame]]

Records = namedtuple("Records", ["a", "b"])


class Dataset:
    def __init__(
        self,
        records: RecordsLike,
        record_id_pairs: pd.DataFrame,
        labels: pd.Series = None,
    ):
        self.records = Records(*records)
        self.record_id_pairs = record_id_pairs
        self.labels = labels if labels is not None else None

    def __repr__(self):
        return f"<Dataset records={len(self.records.a)}x{len(self.records.b)} pairs={0 if self.record_id_pairs is None else len(self.record_id_pairs)} labels={self.labels is not None}>"


class SplittedDataset:
    def __init__(
        self,
        records: RecordsLike,
        *,
        record_id_pairs_train: pd.DataFrame,
        record_id_pairs_val: pd.DataFrame,
        record_id_pairs_test: pd.DataFrame,
        labels_train: pd.Series = None,
        labels_val: pd.Series = None,
        labels_test: pd.Series = None,
    ):
        self.records = Records(*records)
        self._record_id_pairs_train = record_id_pairs_train
        self._record_id_pairs_val = record_id_pairs_val
        self._record_id_pairs_test = record_id_pairs_test
        self._labels_train = labels_train if labels_train is not None else None
        self._labels_val = labels_val if labels_val is not None else None
        self._labels_test = labels_test if labels_test is not None else None

        self.train = Dataset(self.records, self._record_id_pairs_train, self._labels_train)
        self.val = Dataset(self.records, self._record_id_pairs_val, self._labels_val)
        self.test = Dataset(self.records, self._record_id_pairs_test, self._labels_test)

    def __repr__(self):
        return f"<SplittedDataset records={len(self.records.a)}x{len(self.records.b)}>"
