import binascii
import json
from argparse import ArgumentParser
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils import load_config

# %% md

# Data processing

# %%

categories_all = {
    "Finance&Banking": {"Finance", "Banks", "Audit", "finance_banking"},
    "Legal": {"Lawyers", "legal"},
    "IT/Research&Development": {"IT", "Research", "it_research_development"},
    "Medical/Medicine/Paramedical": {"medical_medicine_paramedical"},
    "Businises&Corporate": {"businises_corporate", "Management", "Operations"},
    "Sales/Marketing/PR": {
        "Advertising & Marketing",
        "Marketing",
        "sales_marketing_pr",
    },
    "HR": {"HR"},
    "Manufacturing": {
        "Manufacturing",
        "Auto & Truck Manufacturers",
        "Manufacturing",
        "Metals & Mining",
    },
    "Others": {"others", "others_wiki"},
}


# we are using custom tokenizer because
# data is already tokenized dumped string
def tokenizer(string):
    return json.loads(string)


def main(config: dict, locale: str):
    random_state = config["base"]["random_seed"]
    data_path = (
        Path(config["base"]["data_dir"]) / Path(config["train"]["input_path"]) / locale
    )
    interim_path = (
        Path(config["base"]["data_dir"])
        / Path(config["train"]["interim_path"])
        / locale
    )
    interim_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(config["train"]["models_path"]) / locale
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path = Path(config["train"]["reports_path"]) / locale
    reports_path.mkdir(parents=True, exist_ok=True)

    pca_explained_variance_threshold = config["train"][
        "pca_explained_variance_threshold"
    ]

    data, topics, categories = [], [], {}
    # Only found in tokenizer results are considered
    available_files = config["tokenizer"]["industries"]
    for category, files in categories_all.items():
        interc = set(available_files).intersection(files)
        if interc:
            categories[category] = interc

    for category, files in categories.items():
        logger.info(f"reading category {category} with files {files}")
        tmp = []
        topics.append(category)
        for filename in files:
            with open(data_path / (filename + ".txt"), "r", encoding="utf-8") as f:
                text = f.read().splitlines()
                tmp.extend(text)
        data.append(tmp)

    logger.info("Stack all data into 1 list for comfortable work")
    all_data = [s for cat in data for s in cat]
    logger.info("Create y lables manualy")
    y = []
    for i in range(len(topics)):
        y += [i] * len(data[i])

    logger.info("train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        y,
        test_size=config["train"]["test_size"],
        random_state=random_state,
    )

    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=config["train"]["validate_size"],
        random_state=random_state,
    )

    logger.info("Data vectorization using tfidf")

    max_features = config["train"]["max_features"]
    tfidf = TfidfVectorizer(
        tokenizer=tokenizer, lowercase=False, max_features=max_features
    )

    # We fit vectorizer only on training data

    X_train = tfidf.fit_transform(X_train)
    X_validate = tfidf.transform(X_validate)
    X_test = tfidf.transform(X_test)

    # TODO deterministic output
    # logger.info("Save trained vectorizer")
    # with open(models_path / "tfidf.pkl", "wb") as f:
    #     joblib.dump(tfidf, f)

    logger.info("Save TF-IDF vocabulary")
    with open(interim_path / "tfidf_vocab.txt", "w") as f:
        for word in tfidf.get_feature_names():
            f.write(str(binascii.crc32(word.encode("utf8"))) + "\n")

    logger.info("Save TF-IDF strange file")
    with open(interim_path / "tfidf_idf.txt", "w") as f:
        for idf in tfidf.idf_:
            f.write(str(idf) + "\n")

    logger.info("Save TF-IDF vocabulary words")
    with open(interim_path / "tfidf_vocab_words.txt", "w", encoding="utf-8") as f:
        for word in tfidf.get_feature_names():
            f.write(word + "\n")

    logger.info("Using PCA to compress data")

    # % % time
    pca = PCA(random_state=random_state)
    pca.fit(X_train.toarray())

    logger.info("Determining a sufficient number of components")

    # TODO deterministic output
    # plt.figure(figsize=(16, 9))
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel("number of components")
    # plt.ylabel("cumulative explained variance")
    # plt.grid()
    # plt.savefig(reports_path / "pca_fig.pdf")

    logger.info("Lets leave some number of components")
    # take minimal number of components required to achiev `pca_explained_variance_threshold`
    pca_components = np.min(
        np.argwhere(
            np.cumsum(pca.explained_variance_ratio_) > pca_explained_variance_threshold
        )
    )
    pca = PCA(n_components=pca_components, random_state=random_state)

    X_train = pca.fit_transform(X_train.toarray())
    X_test = pca.transform(X_test.toarray())
    X_validate = pca.transform(X_validate.toarray())

    logger.info("Saving PCA model")

    # ```python
    # n_components
    # n_features
    # mean_[0]...mean_[n_features - 1]
    # components_[0][0]...components_[0][n_features - 1]
    # ....................................................
    # components_[n_components - 1][0]...components_[n_components - 1][n_features - 1]
    # ```

    with open(models_path / "pca.pkl", "wb") as f:
        joblib.dump(pca, f)
    ## load
    with open(models_path / "pca.pkl", "rb") as f:
        pca = joblib.load(f)

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
    logger.info("Creating datasets objects for LightGBM")
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    validate_data = lgb.Dataset(
        X_validate, y_validate, reference=train_data, free_raw_data=False
    )

    ## Set model parameters

    # To see all posible parameters visit https: // lightgbm.readthedocs.io / en / latest / Parameters.html

    params = config["train"]["lightgbm_parameters"]
    params["num_class"] = len(topics)

    logger.info("Train and save model")
    bst = lgb.train(params, train_data, valid_sets=[validate_data])
    bst.save_model(models_path / "model.txt", num_iteration=bst.best_iteration)
    bst = lgb.Booster(model_file=models_path / "model.txt")

    # Model quality
    logger.info("Get predicted lables")
    y_pred_train = bst.predict(X_train)
    y_pred_test = bst.predict(X_test)
    logger.info("Calculate different metrics")
    y_pred_train = np.argmax(y_pred_train, axis=1)
    y_pred_test = np.argmax(y_pred_test, axis=1)

    logger.info("Accuracy score")

    score_train = accuracy_score(y_pred_train, y_train)
    score_test = accuracy_score(y_pred_test, y_test)
    with open(reports_path / "metrics.json", "w") as f:
        json.dump({"accuracy_train": score_train, "accuracy_test": score_test}, f)

    logger.info("Confusion matrix")
    cm = confusion_matrix(y_pred_test, y_test)
    logger.info("Save for DVC")
    pd.DataFrame(
        zip(y_pred_test.reshape(-1), y_test), columns=["actual", "predicted"]
    ).to_csv(reports_path / "confusion.csv", sep=",", index=False)


if __name__ == "__main__":
    a_parser = ArgumentParser()
    a_parser.add_argument("--config", type=Path, required=True)
    a_parser.add_argument("--locale", type=str, required=True)
    args = a_parser.parse_args()
    config = load_config(args.config)
    logger.add(
        Path(config["base"]["log_path"]) / "train_model_{time}.log",
        level=config["base"]["log_level"],
    )
    main(config=config, locale=args.locale)
