import sys
from typing import Tuple

import numpy as np
import pandas as pd


class DummyTkinterFrame:
    ...


class DummyTkinter:
    Frame = DummyTkinterFrame


sys.modules["Tkinter"] = DummyTkinter  # Magellan imports Tkinter, but we don't need it for our use case

import py_entitymatching as em
import py_entitymatching.feature.attributeutils as au
import py_entitymatching.feature.simfunctions as sim
import py_entitymatching.feature.tokenizers as tok

del sys.modules["Tkinter"]


class MagellanMatcher:
    def __init__(self, magellan_matcher=None):
        self.magellan_matcher = magellan_matcher

    def fit(self, records_a: pd.DataFrame, records_b: pd.DataFrame, record_id_pairs: pd.DataFrame, labels: pd.Series):
        records_a = records_a.rename_axis(index="rid")
        records_b = records_b.rename_axis(index="rid")
        A = records_a.reset_index()
        em.set_key(A, "rid")
        B = records_b.reset_index()
        em.set_key(B, "rid")

        A_without_rid = A.drop(columns="rid")
        B_without_rid = B.drop(columns="rid")

        self._attr_types_a = au.get_attr_types(A_without_rid)
        self._attr_types_b = au.get_attr_types(B_without_rid)

        self._attr_corres = au.get_attr_corres(A_without_rid, B_without_rid)

        feature_table = em.get_features(
            A_without_rid,
            B_without_rid,
            self._attr_types_a,
            self._attr_types_b,
            self._attr_corres,
            tok.get_tokenizers_for_matching(),
            sim.get_sim_funs_for_matching(),
        )

        G = (
            record_id_pairs.reset_index()
            .merge(records_a.add_prefix("a."), left_on="a.rid", right_index=True)
            .merge(records_b.add_prefix("b."), left_on="b.rid", right_index=True)
        ).sort_index()

        em.set_key(G, "pid")
        em.set_ltable(G, A)
        em.set_rtable(G, B)
        em.set_fk_ltable(G, "a.rid")
        em.set_fk_rtable(G, "b.rid")

        H = em.extract_feature_vecs(
            G, feature_table=feature_table, n_jobs=1, show_progress=False
        )  # Use n_jobs=1 to avoid Tkinter problems

        if self.magellan_matcher is None:
            self.magellan_matcher = em.RFMatcher()
        self.magellan_matcher.fit(
            table=H.fillna(0).assign(label=labels.astype("int").to_numpy()),
            exclude_attrs=["pid", "a.rid", "b.rid"],
            target_attr="label",
        )

    def _run_predict(
        self, records_a: pd.DataFrame, records_b: pd.DataFrame, record_id_pairs: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        records_a = records_a.rename_axis(index="rid")
        records_b = records_b.rename_axis(index="rid")
        A = records_a.reset_index()
        em.set_key(A, "rid")
        B = records_b.reset_index()
        em.set_key(B, "rid")

        A_without_rid = A.drop(columns="rid")
        B_without_rid = B.drop(columns="rid")

        self._attr_types_a["_table"] = A_without_rid
        self._attr_types_b["_table"] = B_without_rid

        self._attr_corres["ltable"], self._attr_corres["rtable"] = A_without_rid, B_without_rid

        feature_table = em.get_features(
            A_without_rid,
            B_without_rid,
            self._attr_types_a,
            self._attr_types_b,
            self._attr_corres,
            tok.get_tokenizers_for_matching(),
            sim.get_sim_funs_for_matching(),
        )

        C = (
            record_id_pairs.reset_index()
            .merge(records_a.add_prefix("a."), left_on="a.rid", right_index=True)
            .merge(records_b.add_prefix("b."), left_on="b.rid", right_index=True)
        ).sort_index()
        em.set_key(C, "pid")
        em.set_ltable(C, A)
        em.set_rtable(C, B)
        em.set_fk_ltable(C, "a.rid")
        em.set_fk_rtable(C, "b.rid")

        L = em.extract_feature_vecs(
            C, feature_table=feature_table, n_jobs=1, show_progress=False
        )  # Use n_jobs=1 to avoid Tkinter problems
        predictions, probabilities = self.magellan_matcher.predict(
            table=L.fillna(0),
            exclude_attrs=["pid", "a.rid", "b.rid"],
            return_probs=True,
        )
        return predictions, probabilities

    def predict_proba(
        self, record_a: pd.DataFrame, records_b: pd.DataFrame, record_id_pairs: pd.DataFrame
    ) -> pd.Series:
        _, probs = self._run_predict(record_a, records_b, record_id_pairs)
        return pd.Series(probs, index=record_id_pairs.index)

    def predict(self, record_a: pd.DataFrame, records_b: pd.DataFrame, record_id_pairs: pd.DataFrame) -> pd.Series:
        preds, _ = self._run_predict(record_a, records_b, record_id_pairs)
        return pd.Series(preds, index=record_id_pairs.index, dtype=bool)

    def evaluate(
        self, record_a: pd.DataFrame, records_b: pd.DataFrame, record_id_pairs: pd.DataFrame, labels: pd.Series
    ):
        preds, _ = self._run_predict(record_a, records_b, record_id_pairs)
        results = record_id_pairs.assign(label=labels.astype(int), prediction=preds).reset_index()
        em.set_key(results, "pid")
        em.set_fk_ltable(results, "a.rid")
        em.set_fk_rtable(results, "b.rid")
        evaluation = em.eval_matches(results, "label", "prediction")
        return {
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
        }
