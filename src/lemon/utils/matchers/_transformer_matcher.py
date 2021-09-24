from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments


def _format_records(records, attr_strings):
    cols = list(records.columns)
    return pd.DataFrame(
        data={
            "record": [
                " ".join(f"COL {attr_strings(i, c)} VAL {'' if pd.isna(v) else v}" for c, v in zip(cols, r))
                for i, r in enumerate(records.itertuples(index=False, name=None))
            ]
        },
        index=records.index,
        dtype="string",
    )


class _EntityMatchingTransformerDataset(TorchDataset):
    def __init__(
        self,
        record_pairs: pd.DataFrame,
        labels: pd.Series = None,
        pretrained_model=None,
        tokenizer=None,
        max_length=None,
        defer_encoding=False,
        attr_strings: List[Dict[Tuple[str, str], str]] = None,
    ):
        assert pretrained_model is not None or tokenizer is not None

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

        self.max_length = max_length if max_length is not None else self.tokenizer.model_max_length
        self.defer_encoding = defer_encoding
        self._len = len(record_pairs)

        self._labels = labels.astype("int") if labels is not None else None

        if attr_strings is None:
            attr_strings = [{}] * len(record_pairs)
        record_pairs = pd.concat(
            (
                _format_records(
                    record_pairs[[c for c in record_pairs.columns if c.startswith("a.")]].rename(
                        columns=lambda c: c[2:]
                    ),
                    lambda i, attr: attr_strings[i].get(("a", attr), attr),
                ).add_prefix("a."),
                _format_records(
                    record_pairs[[c for c in record_pairs.columns if c.startswith("b.")]].rename(
                        columns=lambda c: c[2:]
                    ),
                    lambda i, attr: attr_strings[i].get(("b", attr), attr),
                ).add_prefix("b."),
            ),
            axis=1,
        )

        if self.defer_encoding:
            self._record_pairs = record_pairs
        else:
            self._encoded_pairs = self.tokenizer(
                record_pairs["a.record"].tolist(),
                record_pairs["b.record"].tolist(),
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        if self.defer_encoding:
            encoded_pair = self.tokenizer(
                self._record_pairs.iloc[index]["a.record"],
                self._record_pairs.iloc[index]["b.record"],
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            encoded_pair = {k: v[index] for k, v in {**self._encoded_pairs}.items()}

        if self._labels is not None:
            return {**encoded_pair, "labels": torch.tensor(self._labels.iloc[index])}
        else:
            return encoded_pair


def _compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = (preds * labels).sum() / preds.sum() if preds.sum() > 0 else 0.0
    recall = (preds * labels).sum() / labels.sum() if labels.sum() > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


class TransformerMatcher:
    def __init__(
        self,
        pretrained_model: Union[str, PreTrainedModel],
        *,
        training_args: Dict = None,
        tokenizer_args: Dict = None,
        extra_input_generator=None,
    ):
        tokenizer_args = tokenizer_args if tokenizer_args is not None else {}
        self._tokenizer_args = {"use_fast": True, **tokenizer_args}
        if isinstance(pretrained_model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **tokenizer_args)
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model.config.name_or_path, **tokenizer_args)
            self.model = pretrained_model
        self.extra_input_generator = extra_input_generator

        default_training_args = {
            "output_dir": "./output/checkpoints",
            "num_train_epochs": 5,
            "evaluation_strategy": "epoch",
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "learning_rate": 3e-5,
            "warmup_steps": 50,
            "logging_dir": "./output/logs",
            "logging_steps": 10,
            "fp16": torch.cuda.is_available(),
            "no_cuda": not torch.cuda.is_available(),
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "dataloader_num_workers": 0,
            "save_strategy": "epoch",
            "report_to": "all",
        }
        self.training_args = {**default_training_args, **(training_args if training_args is not None else {})}

    def _create_dataset(
        self,
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        labels: pd.Series = None,
        attr_strings: List[Dict[Tuple[str, str], str]] = None,
    ) -> TorchDataset:
        record_pairs = (
            (
                record_id_pairs.merge(records_a.add_prefix("a."), left_on="a.rid", right_index=True).merge(
                    records_b.add_prefix("b."), left_on="b.rid", right_index=True
                )
            )
            .sort_index()
            .drop(columns=["a.rid", "b.rid"])
        )
        return _EntityMatchingTransformerDataset(
            record_pairs,
            labels=labels,
            tokenizer=self.tokenizer,
            defer_encoding=self.training_args["dataloader_num_workers"] > 0,
            attr_strings=attr_strings,
        )

    def _create_trainer(self, training_args=None, **kwargs) -> Trainer:
        training_args = {} if training_args is None else training_args
        training_args.setdefault("fp16", torch.cuda.is_available())
        training_args.setdefault("no_cuda", not torch.cuda.is_available())
        return Trainer(
            self.model,
            args=TrainingArguments(**{**self.training_args, **training_args}),
            compute_metrics=_compute_metrics,
            tokenizer=self.tokenizer,
            **kwargs,
        )

    def fit(
        self,
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        labels: pd.Series,
        val_record_id_pairs: pd.DataFrame = None,
        val_labels: pd.Series = None,
        *,
        show_progress: bool = True,
    ):
        train_dataset = self._create_dataset(records_a, records_b, record_id_pairs, labels)
        val_dataset = (
            self._create_dataset(records_a, records_b, val_record_id_pairs, val_labels)
            if val_record_id_pairs is not None
            else None
        )

        trainer = self._create_trainer(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_args={"disable_tqdm": not show_progress},
        )

        trainer.train()

    def predict_proba(
        self,
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        *,
        attr_strings: List[Dict[Tuple[str, str], str]] = None,
        show_progress: bool = True,
    ) -> pd.Series:
        import numpy as np

        transformer_dataset = self._create_dataset(records_a, records_b, record_id_pairs, attr_strings=attr_strings)
        trainer = self._create_trainer(
            training_args={
                "disable_tqdm": not show_progress,
                "skip_memory_metrics": True,
                "seed": np.random.randint(0, 2 ** 31),
            }
        )
        return pd.Series(
            data=torch.softmax(
                torch.from_numpy(trainer.predict(transformer_dataset).predictions.astype("float64")),
                dim=1,
            )[:, 1]
            .detach()
            .numpy(),
            index=record_id_pairs.index,
        )

    def predict(
        self,
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        *,
        show_progress: bool = True,
    ) -> pd.Series:
        confidences = self.predict_proba(records_a, records_b, record_id_pairs, show_progress=show_progress)
        return confidences >= 0.5

    def evaluate(
        self,
        records_a: pd.DataFrame,
        records_b: pd.DataFrame,
        record_id_pairs: pd.DataFrame,
        labels: pd.Series,
        *,
        show_progress: bool = True,
    ) -> Dict:
        dataset = self._create_dataset(records_a, records_b, record_id_pairs, labels)
        trainer = self._create_trainer(
            training_args={
                "disable_tqdm": not show_progress,
                "skip_memory_metrics": True,
                "seed": np.random.randint(0, 2 ** 31),
            }
        )
        metrics = trainer.predict(dataset, metric_key_prefix="test").metrics
        return {
            "precision": metrics["test_precision"],
            "recall": metrics["test_recall"],
            "f1": metrics["test_f1"],
        }
