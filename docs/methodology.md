# Methodology and limitations

Combines the supplied regularization bonus and housing PCA regression work using the supplied Boston-format CSV. The legacy B variable is excluded from predictors. OLS, ridge, lasso, elastic net, and principal-component regression share one held-out split. Scalers and PCA are fitted inside their pipelines; hyperparameter selection is confined to training folds. These legacy observations are a historical modeling benchmark, not a basis for present property decisions.
