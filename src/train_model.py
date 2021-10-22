# %%

import binascii
import json
import os
import pickle
import random
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from imblearn.over_sampling import RandomOverSampler
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# %% md

# Data processing

# %%


def main(config: dict):
    random_state = config["base"]["random_state"]
    data_path = Path(config["train"]["input_path"])
    interim_path = Path(config["train"]["interim_path"])
    interim_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(config["train"]["models_path"])
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path = config["train"]["reports_path"]
    reports_path.mkdir(parents=True, exist_ok=True)

    pca_components = config["train"]["pca_components"]

    data, topics, categories = [], [], {}
    for item in config["categories"]:
        categories[item["name"]] = item["files"]

    for category, files in categories.items():
        logger.info(f"reading category {category} with files {files}")
        tmp = []
        topics.append(category)
        for filename in files:
            with open(data_path / (filename + ".txt"), "r", encoding="utf-8") as f:
                text = f.read().splitlines()
                tmp.extend(text)
        data.append(tmp)
    # I'm so stupid here, lol :\ U can just set `is_unbalance = True` for lgb
    data[1] = data[1] * 3
    data[2] = random.sample(data[2], 20000)
    data[3] = random.sample(data[3], 20000)
    data[4] = data[4] * 3
    data[5] = data[5] * 3
    data[6] = data[6] * 8
    data[7] = data[7] * 2

    data[1] = data[1] * 3
    data[4] = data[4] * 3
    data[5] = data[5] * 3
    data[6] = data[6] * 11

    ## Stack all data into 1 list for comfortable work
    all_data = [s for cat in data for s in cat]
    ## Create y lables manualy

    y = []
    for i in range(len(topics)):
        y += [i] * len(data[i])

    ## train/test split

    X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        y,
        test_size=config["train"]["test_size"],
        random_state=config["base"]["random_state"],
    )

    ## Data vectorization using tfidf
    # we are using custom tokenizer because
    # data is already tokenized dumped string
    def tokenizer(string):
        return json.loads(string)

    max_features = config["train"]["max_features"]
    tfidf = TfidfVectorizer(
        tokenizer=tokenizer, lowercase=False, max_features=max_features
    )

    # We fit vectorizer only on training data

    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    ## Save trained vectorizer
    with open(models_path / "tfidf.pickle", "wb") as f:
        pickle.dump(tfidf, f)

    with open(interim_path / "tfidf_vocab.txt", "w") as f:
        for word in tfidf.get_feature_names():
            f.write(str(binascii.crc32(word.encode("utf8"))) + "\n")

    with open(interim_path / "tfidf_idf.txt", "w") as f:
        for idf in tfidf.idf_:
            f.write(str(idf) + "\n")

    with open(interim_path / "tfidf_vocab_words.txt", "w", encoding="utf-8") as f:
        for word in tfidf.get_feature_names():
            f.write(word + "\n")

    # Using PCA to compress data

    # % % time
    pca = PCA(random_state=random_state)
    pca.fit(X_train.toarray())

    ## Determining a sufficient number of components

    plt.figure(figsize=(16, 9))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.grid()
    plt.savefig(reports_path / "pca_fig.pdf")

    ## Lets leave some number of components
    pca = PCA(n_components=pca_components, random_state=random_state)

    X_train = pca.fit_transform(X_train.toarray())
    X_test = pca.transform(X_test.toarray())

    ## save

    # ```python
    # n_components
    # n_features
    # mean_[0]...mean_[n_features - 1]
    # components_[0][0]...components_[0][n_features - 1]
    # ....................................................
    # components_[n_components - 1][0]...components_[n_components - 1][n_features - 1]
    # ```

    with open(models_path / "pca.pickle", "wb") as f:
        pickle.dump(pca, f)
    ## load
    with open(models_path / "pca.pickle", "rb") as f:
        pca = pickle.load(f)

    with open(interim_path / "pca.txt", "w") as f:
        f.write("%d %d\n" % (pca_components, max_features))
        for m in pca.mean_:
            f.write(str(m) + " ")

        f.write("\n")

        for i in pca.components_:
            for j in i:
                f.write(str(j) + " ")
            f.write("\n")

    # Model
    ## Creating datasets obgects for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    test_data = lgb.Dataset(X_test, y_test, reference=train_data, free_raw_data=False)

    ## Set model parameters

    # To see all posible parameters visit https: // lightgbm.readthedocs.io / en / latest / Parameters.html

    params = config["train"]["lightgbm_parameters"]
    params["num_class"] = len(topics)

    ## Train and save model
    bst = lgb.train(params, train_data, valid_sets=[test_data])
    bst.save_model(models_path / "model.txt", num_iteration=bst.best_iteration)
    bst = lgb.Booster(model_file=models_path / "model.txt")

    # Model quality
    ## Get predicted lables
    y_pred_train = bst.predict(X_train)
    y_pred_test = bst.predict(X_test)
    ## Calculate different metrics
    y_pred_train = np.argmax(y_pred_train, axis=1)
    y_pred_test = np.argmax(y_pred_test, axis=1)

    ### Accuracy score

    score_train = accuracy_score(y_pred_train, y_train)
    score_test = accuracy_score(y_pred_test, y_test)
    with open(reports_path / "metrics.json", "w") as f:
        json.dump({"accuracy_train": score_train, "accuracy_test": score_test}, f)

    ### Confusion matrix
    cm = confusion_matrix(y_pred_test, y_test)
    # Save for DVC
    pd.DataFrame([y_pred_test, y_test], columns=["actual", "predicted"]).to_csv(
        reports_path / "confusion.csv", sep=",", index=False
    )

    df = pd.DataFrame(
        cm, index=[i for i in categories.keys()], columns=[i for i in categories.keys()]
    )

    plt.figure(figsize=(16, 9))
    sn.heatmap(df, annot=True, fmt="d", vmax=100, cmap="OrRd")
    plt.xticks(rotation=60, horizontalalignment="right")
    plt.savefig(reports_path / "confusion.pdf")
