import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import tika
from loguru import logger
from tika import language, parser

from src.utils import load_config


def extract(from_path: Union[Path, str]) -> dict:
    parsed_data = parser.from_file(str(from_path))
    # NOTE destroys reproducibility since parse time varies from launch to launch
    del parsed_data["metadata"]["X-TIKA:parse_time_millis"]
    parsed_data["metadata"]["language"] = language.from_buffer(parsed_data["content"])
    return parsed_data


def main(source_dir: Path, dest_dir: Path):
    for path in Path(source_dir).glob("**/*"):
        result_path = dest_dir / path.relative_to(source_dir)
        if path.is_file():
            logger.debug(f"path {path} is file")
            try:
                with open(result_path, "w", encoding="utf-8") as f:
                    logger.debug(f"processing with tika {path}")
                    parsed = extract(path)
                    logger.debug(f"saving parsed info to {result_path}")
                    json.dump(parsed, f, ensure_ascii=False)
                    del parsed
            except Exception as e:
                logger.error(str(e) + "\n")

        elif path.is_dir():
            logger.debug(f"path {path} is directory")
            try:
                logger.debug(f"creating directory {result_path}")
                result_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(str(e) + "\n")


if __name__ == "__main__":
    a_parser = ArgumentParser()
    a_parser.add_argument("--config", type=Path, required=True)
    a_parser.add_argument("--locale", type=str, required=True)
    args = a_parser.parse_args()
    config = load_config(args.config)
    # prepare tika
    tika.TikaClientOnly = True
    # prepare logger
    logger.add(
        Path(config["base"]["log_path"]) / "tika_{time}.log",
        level=config["base"]["log_level"],
    )
    src_dir = (
        Path(config["base"]["data_dir"])
        / Path(config["tika"]["input_path"])
        / args.locale
    )
    dest_dir = (
        Path(config["base"]["data_dir"])
        / Path(config["tika"]["output_path"])
        / args.locale
    )
    main(
        source_dir=src_dir,
        dest_dir=dest_dir,
    )
