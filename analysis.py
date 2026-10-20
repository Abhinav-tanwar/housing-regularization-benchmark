import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from utils import ROOT, save, reg


def run():
    d = pd.read_csv(ROOT / "data/input.csv")
    y = d.pop("MEDV")
    X = d
    a, b = train_test_split(np.arange(len(X)), test_size=0.25, random_state=24)
    cv = KFold(5, shuffle=True, random_state=24)
    result = dict(
        rows=len(X),
        train_rows=len(a),
        test_rows=len(b),
        excluded=["B"],
        baseline=reg(y.iloc[b], np.repeat(y.iloc[a].mean(), len(b))),
        models={},
    )
    configs = {
        "OLS": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso": (Lasso(max_iter=20000), {"model__alpha": [0.001, 0.01, 0.1, 1]}),
        "Elastic_net": (
            ElasticNet(max_iter=20000),
            {"model__alpha": [0.01, 0.1, 1], "model__l1_ratio": [0.1, 0.5, 0.9]},
        ),
    }
    for name, (model, grid) in configs.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        fit = GridSearchCV(
            pipe, grid or {}, cv=cv, scoring="neg_root_mean_squared_error"
        ).fit(X.iloc[a], y.iloc[a])
        result["models"][name] = {
            **reg(y.iloc[b], fit.predict(X.iloc[b])),
            "selected": fit.best_params_,
        }
    pcr = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("model", LinearRegression()),
        ]
    ).fit(X.iloc[a], y.iloc[a])
    result["models"]["PCA_regression"] = {
        **reg(y.iloc[b], pcr.predict(X.iloc[b])),
        "components": int(pcr.named_steps["pca"].n_components_),
    }
    return save(result)


if __name__ == "__main__":
    run()
