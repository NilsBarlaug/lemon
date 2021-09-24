import inspect
import random
import re
import statistics
from bisect import bisect_left
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


class _StringSpans:
    __slots__ = ("string", "spans")

    def __init__(self, string, spans):
        self.string = string
        self.spans = spans

    def __getitem__(self, i):
        return self.string[self.spans[2 * i] : self.spans[2 * i + 1]]

    def __iter__(self):
        return iter(self[i] for i in range(len(self.spans) // 2))


class TokenizedString:
    def __init__(
        self,
        string: str,
        spans: np.ndarray,
        omitted_tokens: Set[int] = None,
        default_delimiter=None,
    ):
        assert spans[0] == 0 and spans[-1] == len(string)
        self.string = string
        self.spans = spans
        self.omitted_tokens = set() if omitted_tokens is None else omitted_tokens
        self.default_delimiter = default_delimiter
        self._num_tokens = len(self.spans) // 2 - 1

        self.tokens = _StringSpans(string, spans[1:])
        self.delimiters = _StringSpans(string, spans)

        if self.default_delimiter is None:
            if self._num_tokens >= 2:
                self.default_delimiter = statistics.mode([self.delimiters[i] for i in range(1, self._num_tokens)])
            else:
                self.default_delimiter = " "

    def save(self):
        assert not self.omitted_tokens
        return {"string": self.string, "spans": self.spans.tolist(), "default_delimiter": self.default_delimiter}

    @classmethod
    def load(cls, d):
        return cls(d["string"], np.array(d["spans"], dtype=int), default_delimiter=d["default_delimiter"])

    def _update(self, string=None, spans=None, omitted_tokens=None, default_delimiter=None):
        return TokenizedString(
            string=string if string is not None else self.string,
            spans=spans if spans is not None else self.spans,
            omitted_tokens=omitted_tokens if omitted_tokens is not None else self.omitted_tokens,
            default_delimiter=default_delimiter if default_delimiter is not None else self.default_delimiter,
        )

    @staticmethod
    def tokenize(s: str, token_regexes: str = "[^ ]+", default_delimiter=None) -> "TokenizedString":
        if isinstance(token_regexes, str):
            token_regexes = [token_regexes]
        pattern = "|".join(f"(?:{r})" for r in token_regexes)
        spans = [0]
        for m in re.finditer(pattern, s):
            spans.append(m.start())
            spans.append(m.end())
        spans.append(len(s))
        return TokenizedString(
            s,
            np.array(spans, dtype=int),
            default_delimiter,
        )

    @staticmethod
    def from_str(s: str) -> "TokenizedString":
        return TokenizedString(s, np.array([0, 0, len(s), len(s)], dtype=int))

    @staticmethod
    def from_tokens_and_delimiters(tokens: List[str], delimiters: List[str] = None) -> "TokenizedString":
        if delimiters is None:
            delimiters = [" "] * (len(tokens) + 1)
        s = [delimiters[0]]
        offset = len(delimiters[0])
        spans = [0, offset]
        for t, d in zip(tokens, delimiters[1:]):
            s.append(t)
            offset += len(t)
            spans.append(offset)
            s.append(d)
            offset += len(d)
            spans.append(offset)

        return TokenizedString(string="".join(s), spans=np.array(spans, dtype=int))

    @staticmethod
    def empty(default_delimiter=" ") -> "TokenizedString":
        return TokenizedString("", np.array([0, 0], dtype=int), default_delimiter=default_delimiter)

    def insert(self, i, token, *, before_delimiter=None, delimiters=None):
        assert 0 <= i <= self._num_tokens

        if i < self._num_tokens and i in self.omitted_tokens:
            cur_start = self.spans[2 * i + 1]
            cur_end = self.spans[2 * i + 2]
            string = self.string[:cur_start] + token + self.string[cur_end:]
            spans = self.spans.copy()
            index_change = (cur_start + len(token)) - cur_end
            spans[2 * i + 2 :] += index_change
            omitted_tokens = self.omitted_tokens - {i}
            return self._update(string, spans, omitted_tokens)

        if before_delimiter is None:
            before_delimiter = i != self._num_tokens

        if delimiters is None:
            if i == 0 and before_delimiter:
                if self._num_tokens > 0:
                    delimiters = ["", self.default_delimiter if self.delimiters[0] == "" else self.delimiters[0]]
                else:
                    delimiters = ["", self.delimiters[0]]
            elif i == self._num_tokens and not before_delimiter:
                if self._num_tokens > 0:
                    delimiters = [self.default_delimiter if self.delimiters[-1] == "" else self.delimiters[-1], ""]
                else:
                    delimiters = [self.delimiters[-1], ""]
            elif before_delimiter:
                if self.delimiters[i - 1] == self.delimiters[i]:
                    delimiters = [self.delimiters[i - 1], self.delimiters[i]]
                else:
                    delimiters = [self.default_delimiter, self.default_delimiter if i < self._num_tokens else ""]
            elif not before_delimiter:
                if self.delimiters[i] == self.delimiters[i + 1]:
                    delimiters = [self.delimiters[i], self.delimiters[i + 1]]
                else:
                    delimiters = [self.default_delimiter if i > 0 else "", self.default_delimiter]
            else:
                raise AssertionError

        string = "".join(
            (
                self.string[: self.spans[2 * i]],
                delimiters[0],
                token,
                delimiters[1],
                self.string[self.spans[2 * i + 1] :],
            )
        )
        token_start = self.spans[2 * i] + len(delimiters[0])
        token_end = token_start + len(token)
        second_delimiter_end = token_end + len(delimiters[1])
        spans = np.empty(self.spans.size + 2, dtype=int)
        spans[: 2 * i + 1] = self.spans[: 2 * i + 1]
        spans[2 * i + 1] = token_start
        spans[2 * i + 2] = token_end
        spans[2 * i + 3] = second_delimiter_end
        if i < self._num_tokens:
            index_change = second_delimiter_end - self.spans[2 * i + 1]
            spans[2 * i + 4 :] = self.spans[2 * i + 2 :] + index_change

        return self._update(string, spans)

    def omit(self, start, end=None):
        if not end:
            end = start + 1

        assert 0 <= start < self._num_tokens
        assert 0 < end <= self._num_tokens

        return self._update(omitted_tokens=self.omitted_tokens | set(range(start, end)))

    def merge(self, start, end):
        assert 0 <= start < self._num_tokens
        assert 0 < end <= self._num_tokens

        spans = np.concatenate(
            (
                self.spans[: 2 * start + 2],
                self.spans[2 * end :],
            )
        )

        return self._update(spans=spans)

    @lru_cache(maxsize=1)
    def untokenize(self):
        if not self.omitted_tokens:
            return self.string

        s = []
        prev_i = -1
        new_delimiter = self.delimiters[0]
        for i in sorted(self.omitted_tokens):
            if i > 0 and prev_i < i - 1:
                s.append(new_delimiter)
                s.append(self.string[self.spans[2 * prev_i + 2] : self.spans[2 * i]])
                new_delimiter = self.delimiters[i]
            if new_delimiter != self.delimiters[i + 1]:
                if i == 0 or i == self._num_tokens - 1:
                    new_delimiter = ""
                else:
                    new_delimiter = self.default_delimiter
            prev_i = i
        if new_delimiter is not None:
            s.append(new_delimiter)
        if prev_i + 1 < self._num_tokens:
            s.append(self.string[self.spans[2 * prev_i + 3] :])
        return "".join(s)

    def __len__(self):
        return self._num_tokens

    def __getitem__(self, item):
        return self.tokens[item]

    def __bool__(self):
        return bool(len(self))

    def __str__(self):
        return self.untokenize()

    def __repr__(self):
        return f"<TokenizedString string='{self.string}' spans={self.spans.tolist()} omitted_tokens={self.omitted_tokens} default_delimiter='{self.default_delimiter}'>"


def _fast_choice(population, weights, k, std_random):
    weights = weights.copy()
    samples = []
    for _ in range(k):
        cum_weights = weights.cumsum()
        i = bisect_left(cum_weights, cum_weights[-1] * std_random.random())
        weights[i] = 0
        samples.append(population[i])
    return samples


def _materialize_represenation(repr):
    repr = {p: v.untokenize() if isinstance(v, TokenizedString) else v for p, v in repr.items()}
    attrs = {(source, attr): v for (source, attr, attr_or_val), v in repr.items() if attr_or_val == "attr"}
    vals = {(source, attr): v for (source, attr, attr_or_val), v in repr.items() if attr_or_val == "val"}
    return attrs, vals


def perturb_record_pair(
    record_pair,
    perturbations,
    string_representation=None,
    random_state: np.random.Generator = None,
    num_injection_sampling: int = None,
    injection_only_append_to_same_attr: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[Tuple[str, str], str]], List[int]]:
    if string_representation is None:
        string_representation = {}
    representation = string_representation.copy()
    for source, attr in record_pair.columns.to_list():
        if (source, attr, "attr") not in representation:
            representation[(source, attr, "attr")] = attr
        if (source, attr, "val") not in representation:
            representation[(source, attr, "val")] = record_pair[source, attr].iloc[0]

    if random_state is None:
        random_state = np.random.default_rng()
    std_random = random.Random(random_state.integers(1e9))

    relevant_target_attrs = {}
    for source, attr in record_pair.columns.to_list():
        injection_type = record_pair.dtypes[source, attr]
        target_source = "a" if source == "b" else "b"

        relevant_target_attrs[(source, attr, "attr")] = list(c for c in record_pair[target_source].columns if c != attr)

        relevant_target_attrs[(source, attr, "val")] = [
            target_attr
            for target_attr in list(record_pair[target_source].columns)
            if str(injection_type) == str(record_pair.dtypes[target_source, target_attr])
            or pd.api.types.is_string_dtype(record_pair.dtypes[target_source, target_attr])
        ]

    relevant_targets = {}
    for (source, attr, attr_or_val), target_attrs in relevant_target_attrs.items():
        targets = []
        target_weights = []
        target_source = "a" if source == "b" else "b"

        if injection_only_append_to_same_attr:
            target_weights.append(1.0)
            targets.append((target_source, attr, attr_or_val, len(representation[(target_source, attr, attr_or_val)])))
        else:
            for target_attr in target_attrs:
                target_pp = (target_source, target_attr, attr_or_val)
                if isinstance(representation[target_pp], TokenizedString):
                    num_target_j = len(representation[target_pp]) + 1
                else:
                    num_target_j = 1
                target_attr_weight = max(1, len(target_attrs) - 1) if attr == target_attr else 1
                target_weights.extend([target_attr_weight / num_target_j for _ in range(num_target_j)])
                targets.extend(
                    [(target_source, target_attr, attr_or_val, target_j) for target_j in range(num_target_j)]
                )

        target_weights = np.array(target_weights)
        target_weights = target_weights / target_weights.sum() if target_weights.size else target_weights
        relevant_targets[(source, attr, attr_or_val)] = (targets, target_weights)

    perturbed_representations = []
    groups = []
    for group_i, (exclusions, injections) in enumerate(perturbations):
        with_exclusions = representation.copy()
        token_exclusions = [p for p in exclusions if p[3] is not None]
        value_exclusions = [p for p in exclusions if p[3] is None]
        for source, attr, attr_or_val, j in token_exclusions:
            with_exclusions[(source, attr, attr_or_val)] = with_exclusions[(source, attr, attr_or_val)].omit(j)
        for source, attr, attr_or_val, _ in value_exclusions:
            existing_value = with_exclusions[(source, attr, attr_or_val)]
            if isinstance(existing_value, TokenizedString):
                new_value = TokenizedString.empty(default_delimiter=existing_value.default_delimiter)
            elif isinstance(existing_value, str):
                new_value = ""
            else:
                new_value = None
            with_exclusions[(source, attr, attr_or_val)] = new_value

        if not injections or not injection_only_append_to_same_attr:
            perturbed_representations.append(with_exclusions)
            groups.append(group_i)

        if injections:
            max_injection_sampling = 0
            for (source, attr, attr_or_val, j) in injections:
                target_source = "a" if source == "b" else "b"
                target_attrs = relevant_target_attrs[(source, attr, attr_or_val)]
                suggested_injection_sampling = 0
                for target_attr in target_attrs:
                    target_value = with_exclusions[(target_source, target_attr, attr_or_val)]
                    if isinstance(target_value, TokenizedString):
                        suggested_injection_sampling += min(3, len(target_value))
                    else:
                        suggested_injection_sampling += 1
                suggested_injection_sampling = min(10, suggested_injection_sampling)
                max_injection_sampling = max(max_injection_sampling, suggested_injection_sampling)

            if num_injection_sampling is None:
                num_injection_sampling = max_injection_sampling
            if injection_only_append_to_same_attr:
                num_injection_sampling = 1

            injection_targets_used = {p: set() for p in injections}
            sampled_targets = {}
            for p in injections:
                targets, target_weights = relevant_targets[p[:3]]
                sampled_targets[p] = []
                while len(sampled_targets[p]) < num_injection_sampling and targets:
                    sampled_targets[p].extend(
                        _fast_choice(
                            targets,
                            weights=target_weights,
                            k=min(len(targets), num_injection_sampling - len(sampled_targets[p])),
                            std_random=std_random,
                        )
                    )

            for sampling_i in range(num_injection_sampling):
                perturbed = with_exclusions.copy()
                if injection_only_append_to_same_attr:
                    injections = sorted(injections, reverse=True, key=lambda inj: inj[3])
                else:
                    std_random.shuffle(injections)
                for p in injections:
                    source, attr, attr_or_val, j = p
                    pp = p[:3]

                    if isinstance(representation[pp], TokenizedString):
                        injection_value = representation[pp][j]
                    else:
                        injection_value = representation[pp]

                    if pd.isna(injection_value):
                        continue

                    if not sampled_targets[p]:
                        continue
                    target = sampled_targets[p][sampling_i]

                    injection_targets_used[p].add(target)
                    target_pp = target[:3]

                    if isinstance(perturbed[target_pp], TokenizedString):
                        perturbed[target_pp] = perturbed[target_pp].insert(
                            target[3], str(injection_value), before_delimiter=std_random.random() < 0.5
                        )
                    elif isinstance(perturbed[target_pp], str):
                        perturbed[target_pp] = TokenizedString.from_str(str(injection_value))
                    elif pd.isna(perturbed[target_pp]):
                        if isinstance(injection_value, str):
                            perturbed[target_pp] = TokenizedString.from_str(str(injection_value))
                        else:
                            perturbed[target_pp] = injection_value
                    else:
                        perturbed[target_pp] = injection_value

                perturbed_representations.append(perturbed)
                groups.append(group_i)

    all_attrs, all_vals = zip(*[_materialize_represenation(repr) for repr in perturbed_representations])
    record_pairs = pd.DataFrame(all_vals, columns=record_pair.columns).astype(record_pair.dtypes)
    return record_pairs, all_attrs, groups


def get_predictions_scores_for_perturbed_record_pairs(
    record_pairs, attr_strings, groups, predict_proba, show_progress
) -> np.ndarray:

    # Avoid running prediction on duplicates
    num_groups = groups[-1] + 1
    dtypes = record_pairs.dtypes
    record_pairs = record_pairs.assign(attr_strings=[str(x) for x in attr_strings], group=groups)
    record_pairs = (
        record_pairs.groupby(by=record_pairs.columns[:-1].to_list(), as_index=False, dropna=False)
        .agg(list)
        .astype(dtypes)
    )
    groups_per_unique_pair = record_pairs["group"]
    record_pairs = record_pairs[record_pairs.columns[:-2]]

    records_a = record_pairs["a"].rename_axis(index="rid")
    records_b = record_pairs["b"].rename_axis(index="rid")
    record_id_pairs = pd.DataFrame({"a.rid": range(len(record_pairs)), "b.rid": range(len(record_pairs))}).rename_axis(
        index="pid"
    )

    predict_proba_kwargs = {}
    if "show_progress" in inspect.signature(predict_proba).parameters:
        predict_proba_kwargs["show_progress"] = show_progress
    if "attr_strings" in inspect.signature(predict_proba).parameters:
        predict_proba_kwargs["attr_strings"] = attr_strings
    all_predictions = np.array(predict_proba(records_a, records_b, record_id_pairs, **predict_proba_kwargs))

    predictions = [float("-inf")] * num_groups
    for p, groups_for_pair in zip(all_predictions, groups_per_unique_pair):
        for g in groups_for_pair:
            predictions[g] = max(predictions[g], p)
    predictions = np.array(predictions)

    return predictions
