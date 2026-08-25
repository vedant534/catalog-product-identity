# Data

`python run_pipeline.py --stage develop` downloads and extracts the Amazon–Google Products
benchmark automatically when `data/raw/` is empty.

If automatic download is unavailable, download
[Amazon-GoogleProducts.zip](https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip)
from the [Database Group Leipzig benchmark page](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
and extract these files into `data/raw/`:

- `Amazon.csv`
- `GoogleProducts.csv`
- `Amzon_GoogleProducts_perfectMapping.csv`

The misspelling in the mapping filename is present in the original archive.
Raw data is excluded from Git. The benchmark page provides the applicable
license and attribution information.
