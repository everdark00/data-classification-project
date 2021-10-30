from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from shutil import copy2

from loguru import logger

from src.utils import load_config


def main(config: dict, test_dir: Path):
    files_per_dir = 1
    # raw_data_path = Path(config["base"]["data_dir"]) / "raw"
    raw_data_path = Path("data") / "raw"
    new_data_path = test_dir / "raw"
    logger.info(f"creating dir {new_data_path}")
    new_data_path.mkdir(parents=True, exist_ok=True)
    dir_counter = defaultdict(int)
    for path in raw_data_path.glob("**/*"):
        new_path = new_data_path / path.relative_to(raw_data_path)
        if new_path.name.endswith("dvc") or new_path.name.endswith(".gitignore"):
            logger.info(f"skipping DVC file {new_path}")
            continue
        if path.is_dir():
            logger.info(f"creating data dir {new_path}")
            new_path.mkdir(parents=True, exist_ok=True)
        else:
            p_dir = str(path.parent)
            if dir_counter[p_dir] < files_per_dir:
                copy2(path, new_path)
                dir_counter[p_dir] += 1


if __name__ == "__main__":
    a_parser = ArgumentParser()
    a_parser.add_argument("--config", type=Path, required=True)
    a_parser.add_argument("--test_dir", type=Path, required=True)
    args = a_parser.parse_args()
    config = load_config(args.config)
    logger.add(
        Path(config["base"]["log_path"]) / "train_model_{time}.log",
        level=config["base"]["log_level"],
    )
    main(config=config, test_dir=Path(args.test_dir))
