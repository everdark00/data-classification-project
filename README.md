# Data classification
Full repo for data classification project

Repo contains **both** source code and data.
Data is kept on Google Cloud and managed by [DVC](https://dvc.org).

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

## How to add a language
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

## How does it work?
DVC tracks md5 hashes to determine whether the data was changed.
If so, DVC re-runs all dependant stages in pipeline, saves newly produced md5 hashes and commits them to git.

## Caveats
1. When checking out a branch, don't forget to do `dvc fetch && dvc checkout` ([docs](https://dvc.org/doc/command-reference/checkout)).