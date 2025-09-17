import base64
import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera


def _normal_pdf(x, mu=0.0, sigma=1.0):
    """Standard normal PDF used for overlay on the histogram."""
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fig_to_b64(fig, *, dpi=360) -> str:
    """Serialize a Matplotlib Figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


def residual_diagnostics(
    residuals,
    sigma=None,
    title=None,
    return_plots=False,
    make_grid=True,
    dpi=120,
):
    """
    Minimal normality & dependence diagnostics for residuals.

    Parameters
    ----------
    residuals : array-like (pd.Series/np.ndarray)
        Model residuals (e.g., OLS/ARIMA). For GARCH use 'result.resid'.
    sigma : array-like or None
        Estimated conditional volatility (e.g., 'result.conditional_volatility' from 'arch').
        If None -> standardize by sample mean/std.
    title : str or None
        Title suffix for plots.
    return_plots : bool
        If True, returns a dict with base64 PNGs of plots.
    make_grid : bool
        If True and return_plots=True, also returns 'grid' (2×2 combined figure).
    dpi : int
        DPI used for PNG rendering.

    Returns
    -------
    res_std : pd.Series
        Standardized residuals.
    summary : pd.DataFrame
        Table with JB, LB(lag1, lag2), ARCH-LM(lag1, lag2), skewness, excess kurtosis.
    plots : dict[str, str] (optional)
        Base64 PNGs keyed by {'hist','qq','acf','acf2',('grid' if make_grid)}.
    """
    res = pd.Series(residuals, name="resid").dropna()
    if sigma is not None:
        sig = pd.Series(sigma, index=res.index).reindex(res.index).astype(float)
        res_std = res / sig.replace(0, np.nan)
    else:
        res_std = (res - res.mean()) / res.std(ddof=1)

    # --- Build figures explicitly so we can export them ---
    # 1) Histogram + N(0,1)
    fig_hist, ax1 = plt.subplots()
    ax1.hist(res_std, bins="auto", density=True)
    xs = np.linspace(res_std.quantile(0.001), res_std.quantile(0.999), 400)
    ax1.plot(xs, _normal_pdf(xs), linewidth=1.0)
    ax1.set_title(f"Histogram of standardized residuals{f' — {title}' if title else ''}")

    # 2) QQ-plot vs Normal
    fig_qq = qqplot(res_std, line="45")
    fig_qq.suptitle(f"QQ-plot vs Normal{f' — {title}' if title else ''}")

    # --- Stats / tests ---
    jb_stat, jb_p, skew, kurt = jarque_bera(res_std)
    summary = pd.DataFrame(
        {
            "stat": [
                "Jarque–Bera",
                "Skewness",
                "Excess kurtosis",
            ],
            "value": [
                float(jb_stat),
                float(skew),
                float(kurt - 3.0),
            ],
            "p_value": [
                float(jb_p),
                np.nan,
                np.nan,
            ],
        }
    )

    if not return_plots:
        # free figures from memory since not needed
        plt.close(fig_hist)
        plt.close(fig_qq)
        return res_std, summary

    # Serialize individual plots
    plots = {
        "hist": _fig_to_b64(fig_hist, dpi=dpi),
        "qq": _fig_to_b64(fig_qq, dpi=dpi),
    }

    # Optional combined 2×2 grid for convenience
    if make_grid:
        fig_grid, axes = plt.subplots(1, 2, figsize=(12, 4))
        # left: histogram
        axes[0].hist(res_std, bins="auto", density=True)
        axes[0].plot(xs, _normal_pdf(xs), linewidth=1.0)
        axes[0].set_title("Histogram (std. residuals)")
        axes[0].set_xlabel("Standardized residuals")
        axes[0].set_ylabel("Standardized residuals")
        axes[0].grid(True, which="both", alpha=0.5, linewidth=0.7)

        # right: QQ
        qqplot(res_std, line="45", ax=axes[1])
        axes[1].set_title("QQ-plot vs Normal")
        axes[1].grid(True, which="both", alpha=0.5, linewidth=0.7)

        fig_grid.suptitle(title or "Residual diagnostics")
        fig_grid.tight_layout()
        plots["grid"] = _fig_to_b64(fig_grid, dpi=dpi)

    return res_std, summary, plots
