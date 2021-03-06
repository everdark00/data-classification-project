# Data classification
Full repo for data classification project

Repo contains **both** source code and data.
Data is kept on Google Cloud and managed by [DVC](https://dvc.org).

[[_TOC_]]

## Source code structure
```
data/           <- data
  raw/          <- raw documents obtained from crawlers
    de/
    fr/
    ...
  interim/      <- any intermediate data (e.g. processed docs)
    de/         <- split by language
    fr/
    ...
data_test/      <- small dataset for debugging
  <same structure as `data/`>
go_crawler/     <- directory for all crawlers related code (crawlers + config)
de-crawler/     <- German config.toml + URL database
fr-crawler/     <- French config.toml + URL database
...
logs/           <- all logs from stages
models/         <- dupmed model files
  de/
  fr/
  ...
reports/        <- all metrics, plot data, etc
  de/
  fr/
  ...
src/            <- Python sources, mainly pipeline files

dvc.lock        <- lock file for data versioning. DO NOT edit manually
dvc.yaml        <- description of all ML stages in project pipeline
params.yaml     <- single place for all parameters, tunings and configurations
requirements.txt <- package list
```

## Development cycle
| N | Server                                                                                                                                                                      | Local machine                                                                                                                                                                                                         |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | do data crawling to `data/raw` folder                                                                                                                                       | -                                                                                                                                                                                                                     |
| 2 | Unfreeze `make_small_dataset` stage, run `dvc repro make_small_dataset -f`, then add to DVC `dvc add data_test/`, commit and push `git push && dvc push` | -                                                                                                                                                                                                                     |
| 3 | Push the rest of data: `dvc push`                                                                                                                             | Checkout `main` branch, do `dvc fetch make_small_dataset && dvc checkout make_small_dataset`. You will have all data from step #2 (small dataset). Set `data_dir` in `params.yaml` to `data_test`! Conduct experiments, then commit them.                                                                    |
| 4 | -                                                                                                                                                                           | Open MR to `main`, wait for review and approval process to finish. **DO NOT** merge yet, you need to revert `data_dir` to `data`.                                                                                            |
| 6 | Fetch merged `master`: `git checkout master && git pull`. Run pipeline with `dvc repro`, then push results (git add, git commit, dvc push)                                  | Continue development in a separated branch                                                                                                                                                                                  |

## How to add a language (long way)
_For quick setup read "Local development" section below._

**Do the following steps on production server!**
1. Create a new branch `lang/<lang>`.
2. Create a folder `<lang>-crawler`, e.g. `es-crawler` and put `config.toml` from `de-crawler` there.
3. Set correct paths in your new `config.toml` for data
4. Copy URL database file to your `<lang>-crawler` directory, set correct db-file path in `config.toml`.
   Do not forget to reset all `is_crawled` flags:
```sql
UPDATE companies SET is_common_crawled=0 WHERE 1=1;
UPDATE companies SET is_google_crawled=0 WHERE 1=1;
UPDATE companies SET is_colly_crawled=0 WHERE 1=1;
```
5. Build crawler, copy bin file to `<lang>-crawler/` dir and run it:
```bash
go build
cp dataclassification-crawler ../<lang>-crawler/
cd ../<lang>-crawler/
./dataclassification-crawler
```
6. Wait for the crawler to finish (can take up to 24 hours)
7. Once crawler is done, add files to DVC (this will set up data tracking):
```bash
dvc add data/raw/<lang>
# DVC will suggest these commands - do as it asks
git add data/raw/<lang>/.gitignore
git add data/raw/<lang>/.<lang>.dvc
```
8. Add pipeline to `dvc.yaml` -
   simple copy-paste one of existing (e.g. `de`) and correct the language in configuration.
9. Run pipeline for your language with `dvc repro train_<lang>`. You can rerun **all languages** with `dvc repro`
10. Add all files that DVC asks, to git, commit it and push with `git push`.
11. Push data (not sources) with `dvc push`.

Good job! Model files are waiting for you at `models/` folder.
They are tracked among git branches, reproducible and visualized.

### Bonus
All metrics and plots are uploaded
[here](https://studio.iterative.ai/user/alekseik1/views/dataclassification-crawler-e1vtfsj7dv).

## Local development
1. Create a new branch from `main`.
2. Set up packages
```bash
python -m venv venv
# on Linux and MacOS
source venv/bin/activate
# on Windows
venv/Scripts/activate # TODO check it
# all systems
pip install -r requirements.txt
```
3. Fetch small data (_all dataset is too huge_):
```bash
dvc fetch make_small_dataset
dvc checkout make_small_dataset
```
4. You should have `data_test` directory filled with data. This directory is your primary data source!
5. Make sure that `base.data_dir` in `params.yaml` is set to `data_test`
6. You can now use `dvc repro` as usual. Make changes, commit experiments and have fun.
7. Open MR to `main` branch.

## How does it work?
DVC tracks md5 hashes to determine whether the data was changed.
If so, DVC re-runs all dependant stages in pipeline, saves newly produced md5 hashes and commits them to git.

## Caveats
1. When checking out a branch, don't forget to do `dvc fetch && dvc checkout` ([docs](https://dvc.org/doc/command-reference/checkout)).