# Housing Regularization Benchmark

OLS, ridge, lasso, elastic net, and principal-component regression on a legacy housing sample.

**Author and maintainer:** Abhinav Tanwar.

## Run

Requires Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`.

```sh
python analysis.py
python -m unittest discover -s tests
```

## Methods and limitations

Combines the supplied regularization bonus and housing PCA regression work using the supplied Boston-format CSV. The legacy B variable is excluded from predictors. OLS, ridge, lasso, elastic net, and principal-component regression share one held-out split. Scalers and PCA are fitted inside their pipelines; hyperparameter selection is confined to training folds. These legacy observations are a historical modeling benchmark, not a basis for present property decisions.

## Measured results

- **OLS:** rmse = 4.761; mae = 3.271; r squared = 0.6934.
- **Ridge:** rmse = 4.81; mae = 3.268; r squared = 0.6872.
- **Lasso:** rmse = 4.77; mae = 3.27; r squared = 0.6923.
- **Elastic net:** rmse = 4.78; mae = 3.268; r squared = 0.6911.
- **PCA regression:** rmse = 5.252; mae = 3.556; r squared = 0.627.

[Full calculated metrics](results/metrics.json) · [Input provenance](data/README.md) · [Method changes](docs/methodology.md). Open `report.html` for the visual summary.

The clean notebook in `notebooks/` delegates to the same module. Notebook cells were syntax-checked with outputs cleared; native scripts and tests were executed.
