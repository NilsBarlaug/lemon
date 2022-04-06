import math
import re
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._lemon_utils import TokenizedString, get_predictions_scores_for_perturbed_record_pairs, perturb_record_pair
from ._matching_attribution_explanation import (
    Attribution,
    MatchingAttributionExplanation,
    explanation_counterfactual_strength,
)


def _reduce_tokenized_string_granularity(s: TokenizedString, n: int) -> TokenizedString:
    num_tokens = len(s)
    num_output_tokens = math.ceil(num_tokens / n)
    group_size = len(s) / num_output_tokens
    num_tokens_consumed = 0

    for i in range(num_output_tokens):
        to_be_consumed = n
        while num_tokens_consumed + to_be_consumed >= (i + 1) * group_size + 1:
            to_be_consumed -= 1
        s = s.merge(start=i, end=i + to_be_consumed)
        num_tokens_consumed += to_be_consumed
    return s


class _InterpretableRecordPair:
    def __init__(
        self,
        record_pair: pd.DataFrame,
        granularity: str,
        token_representation: str,
        features_a: bool,
        features_b: bool,
        features_attr: bool,
        features_val: bool,
        token_regexes: Union[Dict[Tuple[str, str, str], Union[List[str], str]], List[str], str],
    ):
        if token_representation not in ["independent", "record-bow", "shared-bow"]:
            raise ValueError("token_repesentation must be one of ['independent', 'record-bow', 'shared-bow']")
        assert granularity in ["tokens", "attributes"] or re.fullmatch("[0-9]+-tokens", granularity)

        self.record_pair = record_pair
        self.granularity = granularity
        self.token_representation = token_representation
        self.features_a = features_a
        self.features_b = features_b
        self.features_attr = features_attr
        self.features_val = features_val
        self.token_regexes = token_regexes
        if isinstance(self.token_regexes, str):
            self.token_regexes = [self.token_regexes]
        if isinstance(self.token_regexes, list):
            rgs = self.token_regexes
            self.token_regexes = {}
            for source, attr in self.record_pair.columns.to_list():
                if self.features_attr:
                    self.token_regexes[(source, attr, "attr")] = rgs
                if self.features_val:
                    self.token_regexes[(source, attr, "val")] = rgs

        self.string_representation = {}

        if self.granularity.endswith("tokens"):
            for source, attr in self.record_pair.columns.to_list():
                if features_attr:
                    self.string_representation[(source, attr, "attr")] = TokenizedString.tokenize(
                        attr, self.token_regexes[(source, attr, "attr")]
                    )
                if features_val:
                    if pd.api.types.is_string_dtype(self.record_pair[source, attr]):
                        val = self.record_pair[source, attr].iloc[0]
                        if not pd.isna(val):
                            self.string_representation[(source, attr, "val")] = TokenizedString.tokenize(
                                val, self.token_regexes[(source, attr, "val")]
                            )

            if self.granularity != "tokens":
                n = int(self.granularity.split("-")[0])
                for pos, repr in self.string_representation.items():
                    if repr is not None:
                        self.string_representation[pos] = _reduce_tokenized_string_granularity(repr, n)

        self._feature_pos = []
        self._feature_values = []
        inverted_index = {}
        value_to_index = {}
        value_to_positions = {}
        for source in ["a", "b"]:
            if (source == "a" and not features_a) or (source == "b" and not features_b):
                continue
            if token_representation == "record-bow":
                value_to_index.clear()

            for attr in self.record_pair[source].columns:
                for attr_or_val in ["attr", "val"]:
                    if (attr_or_val == "attr" and not features_attr) or (attr_or_val == "val" and not features_val):
                        continue
                    if (source, attr, attr_or_val) in self.string_representation:
                        tokenized_string = self.string_representation[(source, attr, attr_or_val)]
                        for j, t in enumerate(tokenized_string):
                            if token_representation.endswith("bow"):
                                if t not in value_to_index:
                                    value_to_index[t] = len(self._feature_pos)
                                    self._feature_pos.append([])
                                    self._feature_values.append(t)
                                self._feature_pos[value_to_index[t]].append((source, attr, attr_or_val, j))
                                inverted_index[(source, attr, attr_or_val, j)] = value_to_index[t]
                            else:
                                inverted_index[(source, attr, attr_or_val, j)] = len(self._feature_pos)
                                self._feature_pos.append([(source, attr, attr_or_val, j)])
                                self._feature_values.append(t)

                            if t not in value_to_positions:
                                value_to_positions[t] = []
                            value_to_positions[t].append((source, attr, attr_or_val, j))
                    else:
                        val = record_pair[source, attr].iloc[0] if attr_or_val == "val" else attr
                        if pd.isna(val):
                            continue
                        inverted_index[(source, attr, attr_or_val, None)] = len(self._feature_pos)
                        self._feature_pos.append([(source, attr, attr_or_val, None)])
                        self._feature_values.append(val)

                        if val not in value_to_positions:
                            value_to_positions[val] = []
                        value_to_positions[val].append((source, attr))

    def __len__(self) -> int:
        return len(self._feature_pos)

    def get_all_pos(self, i: int) -> List[Tuple[str, str, str, Optional[int]]]:
        return self._feature_pos[i]

    def get_first_pos(self, i: int) -> Tuple[str, str, str, Optional[int]]:
        return self._feature_pos[i][0]

    def get_value(self, i: int) -> any:
        return self._feature_values[i]


class _InterpretableSamples:
    def __init__(
        self,
        num_samples: int,
        record_pair: _InterpretableRecordPair,
        random_state: np.random.Generator,
        perturb_injection=True,
    ):
        import sklearn.preprocessing

        self.num_samples = num_samples
        self.record_pair = record_pair
        self.random_state = random_state
        self.perturb_injection = perturb_injection

        num_features = len(record_pair)

        self._X = np.zeros((num_samples, num_features))
        self.distances = [0]

        min_num_exclusions = min(1, max(0, num_features - 1))
        max_num_exclusions = min(max(5, num_features // 5), num_features)
        min_num_injections = 0
        max_num_injections = min(max(3, num_features // 10), num_features)

        for i in range(1, num_samples):
            num_exclusions = random_state.integers(min_num_exclusions, max_num_exclusions) if num_features > 0 else 0
            num_injections = (
                random_state.integers(min_num_injections, max_num_injections)
                if perturb_injection and random_state.random() > 0.5 and num_features > 0
                else 0
            )
            num_changes = num_injections + num_exclusions
            self.distances.append(num_changes / (max_num_exclusions + max_num_injections))

            exclude_indices = random_state.choice(num_features, replace=False, size=num_exclusions)
            self._X[i, exclude_indices] = 1

            inject_indices = random_state.choice(num_features, replace=False, size=num_injections)
            self._X[i, inject_indices] = 2

        categories = [[0, 1, 2]] * self._X.shape[1] if self.perturb_injection else [[0, 1]] * self._X.shape[1]
        self._X_dummy = sklearn.preprocessing.OneHotEncoder(
            categories=categories, drop="first", sparse=False
        ).fit_transform(self._X)

        self.distances = np.array(self.distances)

    def features(self, dummy_encode: bool = True) -> np.ndarray:
        if dummy_encode:
            return self._X_dummy
        else:
            return self._X


def _get_perturbed_record_pairs(
    X: np.ndarray, record_pair: _InterpretableRecordPair, random_state: np.random.Generator
) -> Tuple[pd.DataFrame, List[Dict], List[int]]:
    exclusions = []
    injections = []
    for i in range(X.shape[0]):
        exclusions.append([record_pair.get_first_pos(j) for j in (X[i] == 1).nonzero()[0]])
        injections.append([record_pair.get_first_pos(j) for j in (X[i] == 2).nonzero()[0]])
    records_pairs, attr_strings, groups = perturb_record_pair(
        record_pair=record_pair.record_pair,
        string_representation=record_pair.string_representation,
        perturbations=list(zip(exclusions, injections)),
        random_state=random_state,
    )
    return records_pairs, attr_strings, groups


def _get_predictions(
    X: np.ndarray,
    record_pair: _InterpretableRecordPair,
    predict_proba: Callable,
    random_state,
    show_progress: bool = False,
) -> np.ndarray:
    record_pairs, attr_strings, groups = _get_perturbed_record_pairs(X, record_pair, random_state)
    return get_predictions_scores_for_perturbed_record_pairs(
        record_pairs, attr_strings, groups, predict_proba, show_progress
    )


def _kernel_fn(d: np.ndarray) -> np.ndarray:
    return np.exp(-2 * d)


def _forward_selection(
    X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, num_features: int, feature_group_size: int
) -> np.ndarray:
    from sklearn.linear_model import LinearRegression

    used_features = []
    for _ in range(min(num_features, X.shape[1] // feature_group_size)):
        max_ = float("-inf")
        best = -1
        for feature in range(0, X.shape[1], feature_group_size):
            if feature in used_features:
                continue
            feature_group = list(range(feature, feature + feature_group_size))
            X_used = X[:, used_features + feature_group]
            clf = LinearRegression(fit_intercept=False)
            clf.fit(X_used, y, sample_weight=sample_weights)
            score = clf.score(X_used, y, sample_weight=sample_weights)
            if score > max_:
                best = feature_group
                max_ = score
        used_features.extend(best)
    return np.array(used_features)


def _lime(
    X: np.ndarray, y: np.ndarray, distances: np.ndarray, num_features: int, feature_group_size: int = 1
) -> Tuple[Dict[int, Tuple[float, ...]], float, float]:
    from sklearn.linear_model import LinearRegression

    sample_weights = _kernel_fn(distances)
    sample_weights[0] *= min(distances[1:]) * len(distances)

    y = y - y[0]

    used_features = _forward_selection(X, y, sample_weights, num_features, feature_group_size)
    X_used = X[:, used_features]
    model = LinearRegression(fit_intercept=False)
    model.fit(X_used, y, sample_weight=sample_weights)

    prediction_score = float(model.score(X_used, y, sample_weight=sample_weights))
    local_prediction = model.predict(X_used[0:1])[0]
    coefs = {}
    for f in range(0, len(used_features), feature_group_size):
        features = used_features[f : f + feature_group_size].tolist()
        coefs[features[0] // feature_group_size] = model.coef_[f : f + feature_group_size].tolist()

    return coefs, prediction_score, local_prediction


def _harmonic_mean(x, y):
    return 2 * (x * y) / (x + y)


def _create_explanation(
    interpretable_record_pair: _InterpretableRecordPair,
    coefs: Dict[int, Tuple[float, ...]],
    prediction_score: float,
    dual_explanation: bool,
    metadata: Dict,
) -> MatchingAttributionExplanation:
    def str_val(val):
        return None if pd.isna(val) else str(val)

    record_pair = interpretable_record_pair.record_pair
    string_representation = {}
    for source, attr in record_pair.columns.to_list():
        string_representation[(source, attr, "attr")] = interpretable_record_pair.string_representation.get(
            (source, attr, "attr"), attr
        )
        string_representation[(source, attr, "val")] = interpretable_record_pair.string_representation.get(
            (source, attr, "val"), str_val(record_pair[source, attr].iloc[0])
        )

    attributions = []
    for i in coefs:
        weight = -coefs[i][0]
        if len(coefs[i]) == 2:
            potential = coefs[i][1]
        else:
            potential = None
        attributions.append(
            Attribution(weight=weight, potential=potential, positions=interpretable_record_pair.get_all_pos(i))
        )

    return MatchingAttributionExplanation(
        record_pair, string_representation, attributions, prediction_score, dual_explanation, metadata=metadata
    )


def _explain(
    record_pair: pd.DataFrame,
    predict_proba: Callable,
    explain_sources: str,
    num_features: int,
    num_samples: Optional[int],
    granularity: str,
    token_representation: str,
    token_patterns: Union[str, List[str], Dict],
    dual_explanation: bool,
    estimate_potential: bool,
    explain_attrs: bool,
    attribution_method: str,
    show_progress: bool,
    random_state: np.random.Generator,
) -> MatchingAttributionExplanation:
    if not (
        granularity in ["tokens", "attributes", "counterfactual"]
        or re.fullmatch("[0-9]+-tokens", granularity)
        or re.fullmatch("counterfactual-x[0-9]+", granularity)
    ):
        raise ValueError(
            "granularity must be 'tokens', 'attributes', 'counterfactual', or on the format '*-tokens' / 'counterfactual-x*' (where * is an integer)"
        )
    if attribution_method not in ["lime", "shap"]:
        raise ValueError("attribution_method must be either 'lime' or 'shap'")
    if attribution_method == "shap" and estimate_potential == True:
        raise ValueError("attribution_method='shap' can't be used when estimate_potential=True")

    if granularity.startswith("counterfactual"):
        max_tokens_in_attribute = 1
        if explain_sources in ["a", "both"]:
            for attr, value in record_pair["a"].iloc[0].items():
                max_tokens_in_attribute = max(
                    max_tokens_in_attribute,
                    len(TokenizedString.tokenize(value, token_patterns)) if isinstance(value, str) else 1,
                )
        if explain_sources in ["b", "both"]:
            for attr, value in record_pair["b"].iloc[0].items():
                max_tokens_in_attribute = max(
                    max_tokens_in_attribute,
                    len(TokenizedString.tokenize(value, token_patterns)) if isinstance(value, str) else 1,
                )

        if re.fullmatch(".+-x[0-9]+", granularity):
            base = int(granularity.split("-")[-1][1:])
        else:
            base = 2

        granularities = ["tokens"]
        n = base
        while n < max_tokens_in_attribute:
            granularities.append(f"{n}-tokens")
            n *= base
        granularities.append("attributes")

    else:
        granularities = [granularity]

    best_explanation: Tuple[float, Optional[MatchingAttributionExplanation]] = (float("-inf"), None)
    for g in granularities:
        interpretable_record_pair = _InterpretableRecordPair(
            record_pair,
            granularity=g,
            token_representation=token_representation,
            features_a=explain_sources in ["a", "both"],
            features_b=explain_sources in ["b", "both"],
            features_attr=explain_attrs,
            features_val=True,
            token_regexes=token_patterns,
        )

        if len(interpretable_record_pair) == 0:
            return _create_explanation(
                interpretable_record_pair,
                coefs={},
                prediction_score=float(
                    np.array(
                        predict_proba(
                            records_a=record_pair["a"].rename_axis(index="rid"),
                            records_b=record_pair["b"].rename_axis(index="rid"),
                            record_id_pairs=pd.DataFrame(
                                {"a.rid": [record_pair.index[0]], "b.rid": [record_pair.index[0]]}
                            ),
                        )
                    )[0]
                ),
                dual_explanation=dual_explanation,
                metadata={"r2_score": None, "granularity": granularity, "token_representation": token_representation},
            )

        if attribution_method == "lime":
            if num_samples is None:
                n = len(interpretable_record_pair)
                num_samples = max(min(30 * n, 3000), 500)

            samples = _InterpretableSamples(
                num_samples=num_samples,
                record_pair=interpretable_record_pair,
                random_state=random_state,
                perturb_injection=estimate_potential,
            )
            predictions = _get_predictions(
                samples.features(dummy_encode=False),
                interpretable_record_pair,
                predict_proba,
                random_state,
                show_progress=show_progress,
            )
            distances = samples.distances

            coefs, r2_score, local_prediction = _lime(
                samples.features(),
                predictions,
                distances,
                num_features,
                feature_group_size=(2 if estimate_potential else 1),
            )

            exp = _create_explanation(
                interpretable_record_pair,
                coefs,
                float(predictions[0]),
                dual_explanation,
                metadata={
                    "r2_score": r2_score,
                    "granularity": granularity,
                    "token_representation": token_representation,
                },
            )
        elif attribution_method == "shap":
            try:
                import shap
            except ImportError:
                raise ImportError("You need to have the shap library installed to use attribution_method='shap'")

            def wrapped_predict_proba(X):
                return _get_predictions(
                    X, interpretable_record_pair, predict_proba, random_state, show_progress=show_progress
                )

            explainer = shap.KernelExplainer(wrapped_predict_proba, np.ones((1, len(interpretable_record_pair))))
            shap_values = explainer.shap_values(
                np.zeros((1, len(interpretable_record_pair))),
                nsamples=num_samples if num_samples is not None else "auto",
            )

            coefs = {i: (-w,) for i, w in enumerate(shap_values[0])}
            exp = _create_explanation(
                interpretable_record_pair,
                coefs,
                float(wrapped_predict_proba(np.zeros((1, len(interpretable_record_pair))))[0]),
                dual_explanation,
                metadata={
                    "granularity": granularity,
                    "token_representation": token_representation,
                },
            )
        else:
            raise AssertionError

        if granularity.startswith("counterfactual"):
            counterfactual_strength, predicted_counterfactual_strength, _ = explanation_counterfactual_strength(
                exp, predict_proba, random_state
            )
            h_mean_cfs = _harmonic_mean(
                counterfactual_strength + 0.5 + 1e-6, predicted_counterfactual_strength + 0.5 + 1e-6
            )
            if h_mean_cfs > best_explanation[0]:
                best_explanation = (h_mean_cfs, exp)
            if counterfactual_strength >= 0.1 and predicted_counterfactual_strength >= 0.1:
                break
        else:
            best_explanation = (0, exp)

    return best_explanation[1]


def _explain_record_pair(
    record_pair,
    predict_proba,
    num_features,
    dual_explanation,
    estimate_potential,
    granularity,
    num_samples,
    token_representation,
    token_patterns,
    explain_attrs,
    attribution_method,
    show_progress,
    random_state,
):
    if not dual_explanation:
        return _explain(
            record_pair,
            predict_proba,
            explain_sources="both",
            num_features=num_features,
            num_samples=num_samples,
            granularity=granularity,
            token_representation=token_representation,
            token_patterns=token_patterns,
            dual_explanation=False,
            estimate_potential=estimate_potential,
            explain_attrs=explain_attrs,
            attribution_method=attribution_method,
            show_progress=show_progress,
            random_state=random_state,
        )
    else:
        explanation_a: MatchingAttributionExplanation
        explanation_b: MatchingAttributionExplanation
        explanation_a, explanation_b = [
            _explain(
                record_pair,
                predict_proba,
                explain_sources=source,
                num_features=num_features,
                num_samples=num_samples,
                granularity=granularity,
                token_representation=token_representation,
                token_patterns=token_patterns,
                dual_explanation=True,
                estimate_potential=estimate_potential,
                explain_attrs=explain_attrs,
                attribution_method=attribution_method,
                show_progress=show_progress,
                random_state=random_state,
            )
            for source in ["a", "b"]
        ]
        if not math.isclose(explanation_a.prediction_score, explanation_b.prediction_score, rel_tol=1e-2):
            warnings.warn(
                f"The prediction score from explanation a and b should be (at least almost) identical, but was {explanation_a.prediction_score} and {explanation_b.prediction_score}"
            )
        string_representation = {
            **{p: s for p, s in explanation_a.string_representation.items() if p[0] == "a"},
            **{p: s for p, s in explanation_b.string_representation.items() if p[0] == "b"},
        }
        return MatchingAttributionExplanation(
            record_pair,
            string_representation,
            attributions=explanation_a.attributions + explanation_b.attributions,
            prediction_score=(explanation_a.prediction_score + explanation_b.prediction_score) / 2,
            dual=True,
            metadata={"a": explanation_a.metadata, "b": explanation_b.metadata},
        )


def explain(
    records_a: pd.DataFrame,
    records_b: pd.DataFrame,
    record_id_pairs: pd.DataFrame,
    predict_proba: Callable,
    *,
    num_features: int = 5,
    dual_explanation: bool = True,
    estimate_potential: bool = True,
    granularity: str = "counterfactual",
    num_samples: int = None,
    token_representation: str = "record-bow",
    token_patterns: Union[str, List[str], Dict] = "[^ ]+",
    explain_attrs: bool = False,
    attribution_method: str = "lime",
    show_progress: bool = True,
    random_state: Union[int, np.random.Generator, None] = 0,
    return_dict: bool = None,
) -> Union[MatchingAttributionExplanation, Dict[any, MatchingAttributionExplanation]]:
    if return_dict == False and len(record_id_pairs) != 1:
        raise ValueError("If return_dict=False you can only explain one record pair (but multiple were provided)")
    if random_state is None:
        random_state = np.random.default_rng()
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    records_a = records_a.convert_dtypes()
    records_b = records_b.convert_dtypes()
    records_a = (
        record_id_pairs[["a.rid"]]
        .merge(records_a, how="left", left_on="a.rid", right_index=True)
        .rename(columns={"a.rid": "rid"})
    )
    records_b = (
        record_id_pairs[["b.rid"]]
        .merge(records_b, how="left", left_on="b.rid", right_index=True)
        .rename(columns={"b.rid": "rid"})
    )
    records_a = records_a.drop(columns="rid")
    records_b = records_b.drop(columns="rid")
    record_pairs = pd.concat((records_a, records_b), axis=1, keys=["a", "b"], names=["source", "attribute"])

    explanations = {}
    s = random_state.bit_generator.state
    if show_progress and len(record_pairs) > 1:
        from tqdm.auto import trange
    else:
        trange = range
    for i in trange(len(record_pairs)):
        random_state.bit_generator.state = s
        explanations[record_pairs.index[i]] = _explain_record_pair(
            record_pairs.iloc[i : i + 1],
            predict_proba,
            num_features=num_features,
            dual_explanation=dual_explanation,
            estimate_potential=estimate_potential,
            granularity=granularity,
            num_samples=num_samples,
            token_representation=token_representation,
            token_patterns=token_patterns,
            explain_attrs=explain_attrs,
            attribution_method=attribution_method,
            show_progress=show_progress and len(record_pairs) == 1,
            random_state=random_state,
        )

    if return_dict == True:
        return explanations
    elif return_dict == False:
        assert len(explanations) == 1
        return list(explanations.values())[0]
    elif return_dict is None:
        if len(explanations) == 1:
            return list(explanations.values())[0]
        else:
            return explanations
    else:
        raise TypeError
