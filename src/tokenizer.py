import json
from argparse import ArgumentParser
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List

import spacy
from loguru import logger

from src.utils import load_config

LOCALE_TO_DICT = {
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "ja": "ja_core_news_sm",
    "ru": "ru_core_news_sm",
}


def tokenizer(text):
    text_tokens = nlp(text)

    return [
        token.lemma_.lower()
        for token in text_tokens
        if (not token.is_space) and (not token.is_punct)
    ]


def process(
    crawlers: List[str],
    industries: List[str],
    max_symbols: int,
    min_tokens: int,
    data_dir: Path,
    result_dir: Path,
    lang: str,
):
    i = 0
    result_dir.mkdir(parents=True, exist_ok=True)
    for crawl in crawlers:
        logger.info(f"processing crawler {crawl}")
        for ind in industries:
            logger.info(f"processing industry {ind}")
            # with open(
            #     "/parsed data/pt-br_dataset/" + ind + ".txt", "a", encoding="utf-8"
            # ) as result_file:
            with open(
                result_dir / (ind + ".txt"), "a", encoding="utf-8"
            ) as result_file:
                # for each site and each document in it
                # for path in Path("/parsed data/" + crawl + "/" + ind).glob("**/*"):
                for path in (data_dir / crawl / ind).glob("**/*"):
                    if path.is_file():
                        with open(path, "r", encoding="utf-8") as inp_f:
                            try:
                                tmp = json.load(inp_f)
                            except JSONDecodeError as e:
                                logger.warning(f"Error while opening file {path}: {e}")
                                continue
                            if (
                                "language" in tmp["metadata"]
                                and tmp["metadata"]["language"] == lang
                            ):
                                i += 1
                                if i % 100 == 0:
                                    logger.info(f"Processed {i} files")
                                try:
                                    tokens = tokenizer(tmp["content"][:max_symbols])
                                except UnicodeDecodeError as e:
                                    logger.error(
                                        f"Error while decoding file {tmp}: {e}"
                                    )
                                else:
                                    if len(tokens) >= min_tokens:
                                        result_file.write(
                                            "{}\n".format(" ".join(tokens))
                                        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--locale", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    base_conf, t_conf = config["base"], config["tokenizer"]
    logger.add(
        Path(base_conf["log_path"]) / "tokenizer_{time}.log",
        level=base_conf["log_level"],
    )
    nlp = spacy.load(LOCALE_TO_DICT[args.locale], disable=["parser", "ner"])

    src_dir = (
        Path(config["base"]["data_dir"]) / Path(t_conf["input_path"]) / args.locale
    )
    dest_dir = (
        Path(config["base"]["data_dir"]) / Path(t_conf["output_path"]) / args.locale
    )

    process(
        crawlers=base_conf["crawlers"],
        industries=t_conf["industries"],
        max_symbols=t_conf["max_symbols"],
        min_tokens=t_conf["min_tokens"],
        lang=args.locale,
        data_dir=src_dir,
        result_dir=dest_dir,
    )
