import binascii
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.utils import load_config


class PrecisionPCA(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold: float = 0.9, random_state: int = 42):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self._tmp_pca = PCA()
        self.pca = PCA()

    def fit(self, X, y):
        self._tmp_pca = PCA(random_state=self.random_state)
        self._tmp_pca.fit(X.toarray())
        logger.info(
            f"Determining a sufficient number of components for threshold={self.variance_threshold}"
        )
        # take minimal number of components required to achieve `pca_explained_variance_threshold`
        pca_components = np.min(
            np.argwhere(
                np.cumsum(self._tmp_pca.explained_variance_ratio_)
                > self.variance_threshold
            )
        )
        logger.info(
            f"Leaving {pca_components} components by threshold={self.variance_threshold}"
        )
        self.pca = PCA(n_components=pca_components, random_state=self.random_state)
        self.pca.fit(X.toarray())
        return self

    def transform(self, X):
        return self.pca.transform(X.toarray())


class LightGbmCpp(BaseEstimator, ClassifierMixin):
    def __init__(self, **params: dict):
        self.params = params
        self.evals_result = {}
        self.bst = None  # type: lgb.Booster

    def fit(self, X, y):
        train_data = lgb.Dataset(X, label=y, free_raw_data=False)
        self.bst = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data],
            valid_names=["training"],
            evals_result=self.evals_result,
        )

    def predict(self, X):
        preds = self.bst.predict(X)
        if len(preds.shape) > 1:
            preds = np.argmax(preds, axis=1)
        return preds

    # HACK: keep all params as a dict rather than class fields for LightGBM lib compatibility
    def set_params(self, **params):
        # super().set_params(**params)
        self.params = params

    def get_params(self, deep=True):
        return self.params


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


def save_models(config: dict, locale: str, pipeline: Pipeline):
    pca, tfidf, classifier = (
        pipeline.named_steps["pca"].pca,
        pipeline.named_steps["tfidf"],
        pipeline.named_steps["classifier"],
    )  # type: PCA, TfidfVectorizer, LightGbmCpp
    interim_path = (
        Path(config["base"]["data_dir"])
        / Path(config["train"]["interim_path"])
        / locale
    )
    interim_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(config["train"]["models_path"]) / locale
    models_path.mkdir(parents=True, exist_ok=True)

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
        for word in pipeline.named_steps["tfidf"].get_feature_names():
            f.write(word + "\n")
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

    with open(interim_path / "pca.txt", "w") as f:
        f.write("%d %d\n" % (pca.n_components, tfidf.max_features))
        for m in pca.mean_:
            f.write(str(m) + " ")
        f.write("\n")
        for i in pca.components_:
            for j in i:
                f.write(str(j) + " ")
            f.write("\n")

    logger.info("saving LightGBM model")
    classifier.bst.save_model(
        models_path / "model.txt", num_iteration=classifier.bst.best_iteration
    )


def save_report(
    config: dict,
    locale: str,
    pipeline: Pipeline,
    y_train: Sequence,
    y_test: Sequence,
    y_pred_train: np.array,
    y_pred_test: np.array,
):
    classifier = pipeline.named_steps["classifier"]  # type: LightGbmCpp
    reports_path = Path(config["train"]["reports_path"]) / locale
    target_metric = classifier.params["metric"]
    reports_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving train report for DVC")
    # NOTE: same as `valid_names` in class definition
    train_info = classifier.evals_result["training"][target_metric]
    y = train_info
    x = list(range(1, len(y) + 1))
    pd.DataFrame(y, index=x, columns=[target_metric]).to_csv(
        reports_path / "train_progress.csv", index_label="iteration"
    )
    logger.info("Accuracy score")
    score_train = accuracy_score(y_pred_train, y_train)
    score_test = accuracy_score(y_pred_test, y_test)
    with open(reports_path / "metrics.json", "w") as f:
        json.dump({"accuracy_train": score_train, "accuracy_test": score_test}, f)
    logger.info("Confusion matrix for DVC")
    pd.DataFrame(
        zip(y_pred_test.reshape(-1), y_test), columns=["actual", "predicted"]
    ).to_csv(reports_path / "confusion.csv", sep=",", index=False)
    logger.info("saving best model params")
    with open(reports_path / "best_params.yaml", "w") as f:
        yaml.dump(classifier.params, f)


def read_data(config: dict, locale: str):
    data_path = (
        Path(config["base"]["data_dir"]) / Path(config["train"]["input_path"]) / locale
    )
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
    return all_data, y, topics


def main(config: dict, locale: str):
    random_state = config["base"]["random_seed"]
    max_features = config["train"]["max_features"]
    pca_threshold = config["train"]["pca_explained_variance_threshold"]
    test_size, validate_size = (
        config["train"]["test_size"],
        config["train"]["validate_size"],
    )
    classifier_grid = config["train"]["lightgbm_parameters"]

    logger.info("reading data")
    all_data, y, topics = read_data(config, locale)
    classifier_grid["num_class"] = [len(topics)]
    classifier_grid["random_state"] = [random_state]
    logger.info("train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=validate_size,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenizer, lowercase=False, max_features=max_features
                ),
            ),
            (
                "pca",
                PrecisionPCA(
                    random_state=random_state, variance_threshold=pca_threshold
                ),
            ),
            ("classifier", LightGbmCpp()),
        ]
    )

    logger.info("Fitting pipeline")
    cv_grid = {f"classifier__{key}": value for key, value in classifier_grid.items()}
    cv = GridSearchCV(estimator=pipeline, param_grid=cv_grid)
    cv.fit(X_train, y_train)
    logger.info("Get predicted lables")
    y_pred_train = cv.predict(X_train)
    y_pred_test = cv.predict(X_test)

    save_models(config, locale, cv.best_estimator_)
    save_report(
        config=config,
        locale=locale,
        pipeline=cv.best_estimator_,
        y_train=y_train,
        y_test=y_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
    )


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
