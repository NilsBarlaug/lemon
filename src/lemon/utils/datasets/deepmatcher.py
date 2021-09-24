import os

import pandas as pd

from ._dataset import SplittedDataset
from ._utils import download_file, get_cache_path


def _deepmatcher_dataset(name, url, dtypes, root=None, download=True):
    name = os.path.join("deepmatcher", name)
    if download:
        filepath = os.path.join(name, url.split("/")[-1])
        download_file(url, filepath, cache_dir=root, unzip=True)
    return _load_deepmatcher_dataset(name, dtypes, root)


def _load_deepmatcher_dataset(name, dtypes, root):
    dir = "exp_data" if os.path.exists(get_cache_path(name, "exp_data", cache_dir=root)) else ""
    records_a, records_b = [
        pd.read_csv(get_cache_path(name, dir, f, cache_dir=root), index_col="id", dtype=dtype).rename_axis(index="__id")
        for f, dtype in [("tableA.csv", dtypes["a"]), ("tableB.csv", dtypes["b"])]
    ]
    train_pairs, val_pairs, test_pairs = [
        pd.read_csv(get_cache_path(name, dir, f, cache_dir=root))
        .rename(columns={"ltable_id": "a.rid", "rtable_id": "b.rid"})
        .astype({"label": "bool"})
        .rename_axis(index="pid")
        for f in ["train.csv", "valid.csv", "test.csv"]
    ]
    return SplittedDataset(
        records=(records_a, records_b),
        record_id_pairs_train=train_pairs[["a.rid", "b.rid"]],
        record_id_pairs_val=val_pairs[["a.rid", "b.rid"]],
        record_id_pairs_test=test_pairs[["a.rid", "b.rid"]],
        labels_train=train_pairs["label"],
        labels_val=val_pairs["label"],
        labels_test=test_pairs["label"],
    )


def structured_amazon_google(root=None, download=True):
    return _deepmatcher_dataset(
        "Structured/Amazon-Google",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/amazon_google_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "title": "string",
                "manufacturer": "string",
                "price": "Float64",
            },
            "b": {
                "id": "int64",
                "title": "string",
                "manufacturer": "string",
                "price": "Float64",
            },
        },
        root=root,
        download=download,
    )


def structured_beer(root=None, download=True):
    return _deepmatcher_dataset(
        "Structured/Beer",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/beer_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "Beer_Name": "string",
                "Brew_Factory_Name": "string",
                "Style": "string",
                "ABV": "string",
            },
            "b": {
                "id": "int64",
                "Beer_Name": "string",
                "Brew_Factory_Name": "string",
                "Style": "string",
                "ABV": "string",
            },
        },
        root=root,
        download=download,
    )


def structured_dblp_acm(root=None, download=True):
    return _deepmatcher_dataset(
        "Structured/DBLP-ACM",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
            "b": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
        },
        root=root,
        download=download,
    )


def structured_dblp_google_scholar(root=None, download=True):
    return _deepmatcher_dataset(
        "Structured/DBLP-GoogleScholar",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/dblp_scholar_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
            "b": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
        },
        root=root,
        download=download,
    )


def structured_fodors_zagat(root=None, download=True):
    dataset = _deepmatcher_dataset(
        "Structured/Fodors-Zagats",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/fodors_zagat_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "name": "string",
                "addr": "string",
                "city": "string",
                "phone": "string",
                "type": "string",
            },
            "b": {
                "id": "int64",
                "name": "string",
                "addr": "string",
                "city": "string",
                "phone": "string",
                "type": "string",
            },
        },
        root=root,
        download=download,
    )
    dataset.records.a.drop(columns="class", inplace=True)
    dataset.records.b.drop(columns="class", inplace=True)
    return dataset


def structured_walmart_amazon(root=None, download=True):
    return _deepmatcher_dataset(
        "Structured/Walmart-Amazon",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "title": "string",
                "category": "string",
                "brand": "string",
                "modelno": "string",
                "price": "Float64",
            },
            "b": {
                "id": "int64",
                "title": "string",
                "category": "string",
                "brand": "string",
                "modelno": "string",
                "price": "Float64",
            },
        },
        root=root,
        download=download,
    )


def structured_itunes_amazon(root=None, download=True):
    return _deepmatcher_dataset(
        "Structured/iTunes-Amazon",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/itunes_amazon_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "Song_Name": "string",
                "Artist_Name": "string",
                "Album_Name": "string",
                "Genre": "string",
                "Price": "string",
                "CopyRight": "string",
                "Time": "string",
                "Released": "string",
            },
            "b": {
                "id": "int64",
                "Song_Name": "string",
                "Artist_Name": "string",
                "Album_Name": "string",
                "Genre": "string",
                "Price": "string",
                "CopyRight": "string",
                "Time": "string",
                "Released": "string",
            },
        },
        root=root,
        download=download,
    )


def dirty_dblp_acm(root=None, download=True):
    return _deepmatcher_dataset(
        "Dirty/DBLP-ACM",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-ACM/dirty_dblp_acm_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
            "b": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
        },
        root=root,
        download=download,
    )


def dirty_dblp_google_scholar(root=None, download=True):
    return _deepmatcher_dataset(
        "Dirty/DBLP-GoogleScholar",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-GoogleScholar/dirty_dblp_scholar_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
            "b": {
                "id": "int64",
                "title": "string",
                "author": "string",
                "venue": "string",
                "year": "Int64",
            },
        },
        root=root,
        download=download,
    )


def dirty_walmart_amazon(root=None, download=True):
    return _deepmatcher_dataset(
        "Dirty/Walmart-Amazon",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/Walmart-Amazon/dirty_walmart_amazon_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "title": "string",
                "category": "string",
                "brand": "string",
                "modelno": "string",
                "price": "Float64",
            },
            "b": {
                "id": "int64",
                "title": "string",
                "category": "string",
                "brand": "string",
                "modelno": "string",
                "price": "Float64",
            },
        },
        root=root,
        download=download,
    )


def dirty_itunes_amazon(root=None, download=True):
    return _deepmatcher_dataset(
        "Dirty/iTunes-Amazon",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/iTunes-Amazon/dirty_itunes_amazon_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "Song_Name": "string",
                "Artist_Name": "string",
                "Album_Name": "string",
                "Genre": "string",
                "Price": "string",
                "CopyRight": "string",
                "Time": "string",
                "Released": "string",
            },
            "b": {
                "id": "int64",
                "Song_Name": "string",
                "Artist_Name": "string",
                "Album_Name": "string",
                "Genre": "string",
                "Price": "string",
                "CopyRight": "string",
                "Time": "string",
                "Released": "string",
            },
        },
        root=root,
        download=download,
    )


def textual_abt_buy(root=None, download=True):
    return _deepmatcher_dataset(
        "Textual/Abt-Buy",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/abt_buy_exp_data.zip",
        dtypes={
            "a": {
                "id": "int64",
                "name": "string",
                "description": "string",
                "price": "Float64",
            },
            "b": {
                "id": "int64",
                "name": "string",
                "description": "string",
                "price": "Float64",
            },
        },
        root=root,
        download=download,
    )


def textual_company(root=None, download=True):
    return _deepmatcher_dataset(
        "Textual/Company",
        "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/company_exp_data.zip",
        dtypes={
            "a": {
                "id": "string",
                "content": "string",
            },
            "b": {
                "id": "string",
                "content": "string",
            },
        },
        root=root,
        download=download,
    )
