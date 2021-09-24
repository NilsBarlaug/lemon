import base64
import colorsys
import dataclasses
import io
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._lemon_utils import TokenizedString, get_predictions_scores_for_perturbed_record_pairs, perturb_record_pair


def _correct_intensity(w):
    if w > 0:
        return 0.05 + 0.95 * w
    else:
        return -0.05 + 0.95 * w


def _highlight_string(string, weight, potential):
    h = 1 / 3 if weight > 0 else 0
    l = np.clip(1 - _correct_intensity(weight + potential - min(0.0, weight)) / 2, 0.5, 1)
    s = np.clip(_correct_intensity(abs(weight) / (abs(weight) + potential)), 0, 1) if potential != 0 else 1
    rgb = colorsys.hls_to_rgb(h, l, s)
    rgb = tuple(int(round(255 * x)) for x in rgb)
    r, g, b = rgb
    return f'<span style="background-color: rgb({r}, {g}, {b});">{string}</span>'


def _highlight_tokenized_string(s: TokenizedString, attributions):
    html = [s.delimiters[0]]
    for i, token in enumerate(s.tokens):
        if i in attributions:
            html.append(_highlight_string(token, *attributions[i]))
        else:
            html.append(token)
        html.append(s.delimiters[i + 1])
    return "".join(html)


@dataclass
class Attribution:
    weight: float
    positions: List[Union[Tuple[str, str, str, Optional[int]]]]
    potential: float = None
    name: str = None

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def consistent_potential(self):
        return max(0.0, 0.0 if self.potential is None else self.potential, -self.weight)


class MatchingAttributionExplanation:
    def __init__(
        self,
        record_pair: pd.DataFrame,
        string_representation: Dict[Tuple, Union[None, str, TokenizedString]],
        attributions: List[Attribution],
        prediction_score: float,
        dual: bool,
        *,
        metadata: Dict[str, any] = None,
    ):
        self.record_pair = record_pair
        self.string_representation = string_representation
        self.attributions = attributions
        self.prediction_score = prediction_score
        self.dual = dual
        self.metadata = metadata if metadata is not None else {}

    def only_attributions_from(self, source) -> "MatchingAttributionExplanation":
        return MatchingAttributionExplanation(
            record_pair=self.record_pair,
            string_representation=self.string_representation,
            attributions=[a for a in self.attributions if all(p[0] == source for p in a.positions)],
            prediction_score=self.prediction_score,
            dual=self.dual,
        )

    def save(self, path: str = None):
        record_pair_feather = io.BytesIO()
        rp = self.record_pair.copy()
        rp.columns = [".".join(c) for c in rp.columns.to_flat_index()]
        rp = rp.reset_index()
        rp.to_feather(record_pair_feather)
        string_representation = [
            (p, val.save() if isinstance(val, TokenizedString) else val)
            for p, val in self.string_representation.items()
        ]
        dump = {
            "record_pair.feather": base64.b64encode(record_pair_feather.getvalue()).decode(),
            "string_representation": string_representation,
            "attributions": [a.to_dict() for a in self.attributions],
            "prediction_score": self.prediction_score,
            "dual": self.dual,
            "metadata": self.metadata,
        }
        if path is None:
            return dump
        else:
            with open(path, "w") as f:
                json.dump(dump, f)

    @classmethod
    def load(cls, path: Union[str, Dict]):
        if isinstance(path, str):
            with open(path, "r") as f:
                path = json.load(f)
        else:
            if not isinstance(path, dict):
                raise TypeError(f"Expecting dict, but got {path}")
        d = path
        record_pair_feather = io.BytesIO(base64.b64decode(d["record_pair.feather"].encode()))
        rp = pd.read_feather(record_pair_feather).set_index("pid")
        rp.columns = pd.MultiIndex.from_tuples([c.split(".", 1) for c in rp.columns], names=["source", "attribute"])

        string_representation = {
            tuple(p): TokenizedString.load(val) if isinstance(val, dict) else val
            for p, val in d["string_representation"]
        }

        attributions = [Attribution(**a) for a in d["attributions"]]
        for a in attributions:
            a.positions = [tuple(p) for p in a.positions]

        return cls(
            record_pair=rp,
            string_representation=string_representation,
            attributions=attributions,
            prediction_score=d["prediction_score"],
            dual=d["dual"],
            metadata=d["metadata"],
        )

    def _highlighted_record(self, s):
        html = [
            """
        <style>
            table.record tr:not(:last-child) {
                border-bottom: 1px solid #CCC;
            }
        </style>
        <table class="record" style="width: 100%; table-layout: fixed; border-collapse: collapse; border-width: 1px 0px 1px 0px; border-color: #777; border-style: solid; font-size: 14px;">
        <tbody>
        """
        ]
        for attr in self.record_pair[s].columns:
            html.append('<tr style="padding: 10px; background-color: white;">')
            attr_attributions = []
            val_attributions = []
            for attribution in self.attributions:
                for pos in attribution.positions:
                    if pos[0] == s and pos[1] == attr:
                        if pos[2] == "attr":
                            attr_attributions.append(
                                (attribution.weight, max(0.0, attribution.consistent_potential), pos[3])
                            )
                        else:
                            val_attributions.append(
                                (attribution.weight, max(0.0, attribution.consistent_potential), pos[3])
                            )

            attr_repr = self.string_representation[(s, attr, "attr")]
            attr_str = "" if attr_repr is None else str(attr_repr)
            if attr_attributions:
                if isinstance(attr_repr, TokenizedString):
                    attr_str = _highlight_tokenized_string(attr_repr, {i: (w, p) for w, p, i in attr_attributions})
                else:
                    assert len(attr_attributions) == 1
                    w, p, i = attr_attributions[0]
                    assert i is None
                    attr_str = _highlight_string(attr_str, w, p)

            val_repr = self.string_representation[(s, attr, "val")]
            val_str = "" if val_repr is None else str(val_repr)
            if val_attributions:
                if isinstance(val_repr, TokenizedString):
                    val_str = _highlight_tokenized_string(val_repr, {i: (w, p) for w, p, i in val_attributions})
                else:
                    assert len(val_attributions) == 1
                    w, p, i = val_attributions[0]
                    assert i is None
                    val_str = _highlight_string(val_str, w, p)

            html.append(
                f'<td style="width: 150px; padding: 5px; text-align: left;">{attr_str}</td><td style="padding: 5px; text-align: left;">{val_str}</td>'
            )
            html.append("</tr>")
        html.append("</tbody></table>")
        return "".join(html)

    def _prediction_header(self, use_percentage):
        prediction_score_str = f"{self.prediction_score:.0%}" if use_percentage else f"{self.prediction_score:.2f}"
        return f"""
        <div style="margin: 0 auto; width: 200px;">
            Prediction: {'<span style="color: green;">Match</span>' if self.prediction_score > 0.5 else '<span style="color: red;">Not match</span>'} ({prediction_score_str})
            <div style="width: 200px; height: 20px; border: 1px solid black;">
                <div style="width: {100 * self.prediction_score:.4f}%; height: 100%; background-color: green;"></div>
            </div>
        </div>
        """

    def as_html(self, min_attribution=0.0, use_percentage=True):
        return f"""
            <div style="display: grid; grid-template-rows: auto; grid-template-columns: 1fr 1fr; column-gap: 20px; row-gap: 10px; font-family: Helvetica; font-size: 14px;">
                <div style="grid-area: 1 / 1 / 2 / 3;">{self._prediction_header(use_percentage)}</div>
                <div style="grid-area: 2 / 1 / 3 / 2;">{self._highlighted_record("a")}</div>
                <div style="grid-area: 2 / 2 / 3 / 3;">{self._highlighted_record("b")}</div>
                <div style="grid-area: 3 / 1 / 4 / 2;">{self.plot("a", min_attribution, return_html=True)}</div>
                <div style="grid-area: 3 / 2 / 4 / 3;">{self.plot("b", min_attribution, return_html=True)}</div>
            </div>
        """

    def _repr_html_(self):
        return self.as_html()

    def plot(
        self, source, min_attribution=0.0, return_html=False, max_features=5, use_percentage=True, show_values=False
    ):
        import matplotlib.pyplot as plt

        values = []
        for attribution in self.attributions:
            if source != "both" and all(s != source for s, _, _, _ in attribution.positions):
                continue
            if abs(attribution.weight) < min_attribution and abs(attribution.consistent_potential) < min_attribution:
                continue
            if attribution.name is not None:
                string = attribution.name
            else:
                s, attr, attr_or_val, j = attribution.positions[0]
                val = self.string_representation[(s, attr, attr_or_val)]
                if j is None:
                    if val is None:
                        string = f"<{attr}>" if attr_or_val == "attr" else f"[{attr}]"
                    else:
                        string = "" if val is None else str(val)
                        if len(string) > 33:
                            string = f"<{attr}>" if attr_or_val == "attr" else f"[{attr}]"
                else:
                    string = val[j]
            if len(string) > 33:
                string = string[:30] + "..."
            values.append((attribution.consistent_potential, attribution.weight, string))

        values.sort(key=lambda v: v[1] + v[0] - min(0.0, v[1]), reverse=True)
        values = values[:max_features]

        fig, ax = plt.subplots(figsize=(8, 0.7 * max(1, len(values))))

        xs = np.arange(len(values)) - len(values) + 0.5
        weights = [v[1] for v in values]
        potential = [v[1] + v[0] for v in values]

        ax.barh(xs, potential, 0.3, color="#ddd", zorder=2)
        ax.barh(xs, weights, 0.3, color=["g" if e > 0 else "r" for e in weights], zorder=2)
        for i, (x, (p, w, s)) in enumerate(zip(xs, values)):
            ax.text(
                0.02 if w > 0 else -0.02,
                x - 0.35,
                s,
                color="black",
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="left" if w > 0 else "right",
            )
            if show_values:
                ax.text(
                    w + 0.03 if w > 0 else w - 0.03,
                    x,
                    f"{abs(w):.0%}" if use_percentage else f"{abs(w):.2f}",
                    color="black",
                    fontsize=8,
                    verticalalignment="center",
                    horizontalalignment="left" if w > 0 else "right",
                )
                if w + p - 0.20 > w:
                    ax.text(
                        w + p + 0.03,
                        x,
                        f"{abs(w+p):.0%}" if use_percentage else f"{abs(w + p):.2f}",
                        color="black",
                        fontsize=8,
                        verticalalignment="center",
                        horizontalalignment="left",
                    )
        ax.invert_yaxis()
        ax.set_xlim((-1.2, 1.2))
        ax.set_ylim((0.5, -max(1, len(values))))
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        xticks = [-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.xaxis.set_ticks(xticks)
        if use_percentage:
            ax.set_xticklabels([f"{x:.0%}" for x in xticks])
        ax.yaxis.set_visible(False)
        ax.vlines(x=xticks, ymin=-1e9, ymax=0, colors="#EEE", linewidth=1)

        if return_html:
            f = BytesIO()
            fig.savefig(f, format="png", dpi=120, bbox_inches="tight")
            f.seek(0)

            png_data = base64.b64encode(f.getvalue())
            plt.close()
            return f'<img style="display: block; margin: 0 auto; max-width: 500px; width: 100%; height: auto;" src="data:image/png;base64,{png_data.decode()}" />'

        return ax


def explanation_counterfactual_strength(
    exp: MatchingAttributionExplanation,
    predict_proba: Callable,
    random_state: np.random.RandomState,
    max_features=None,
    injection_only_append_to_same_attr: bool = False,
) -> Tuple[float, float, int]:
    target_prediction = exp.prediction_score
    exclusions = []
    injections = []
    num_actions = 0
    if exp.prediction_score > 0.5:
        available_attributions = sorted(exp.attributions, reverse=True, key=lambda a: a.weight)
        if max_features is not None:
            available_attributions = available_attributions[:max_features]
        i = 0
        while target_prediction > 0.4 and i < len(available_attributions):
            attribution = available_attributions[i]
            if attribution.weight > 0:
                target_prediction -= attribution.weight
                exclusions.extend(attribution.positions)
                num_actions += 1
            else:
                break
            i += 1
    else:
        available_attributions = sorted(exp.attributions, reverse=True, key=lambda a: a.consistent_potential)
        if max_features is not None:
            available_attributions = available_attributions[:max_features]
        i = 0
        while target_prediction < 0.6 and i < len(available_attributions):
            attribution = available_attributions[i]
            if attribution.consistent_potential > 0.0:
                if -attribution.weight >= attribution.consistent_potential:
                    target_prediction -= attribution.weight
                    exclusions.extend(attribution.positions)
                else:
                    target_prediction += attribution.consistent_potential
                    injections.append(attribution.positions[0])
                num_actions += 1
            else:
                break
            i += 1

    string_representation = {p: v for p, v in exp.string_representation.items() if isinstance(v, TokenizedString)}
    record_pairs, attr_strings, groups = perturb_record_pair(
        exp.record_pair,
        perturbations=[(exclusions, injections)],
        string_representation=string_representation,
        random_state=random_state,
        injection_only_append_to_same_attr=injection_only_append_to_same_attr,
    )
    prediction = get_predictions_scores_for_perturbed_record_pairs(
        record_pairs, attr_strings, groups, predict_proba, show_progress=False
    )[0]

    if exp.prediction_score > 0.5:
        return 0.5 - prediction, 0.5 - target_prediction, num_actions
    else:
        return prediction - 0.5, target_prediction - 0.5, num_actions
