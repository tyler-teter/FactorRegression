from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from pandas_datareader import data as web
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


@dataclass
class RegressionResult:
    summary_table: pd.DataFrame
    fitted_series: pd.Series
    residual_series: pd.Series
    r_squared: float
    adjusted_r_squared: float
    annualized_alpha: float
    f_statistic: float
    f_pvalue: float
    durbin_watson: float
    breusch_pagan_pvalue: float


def load_returns_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Upload CSV or Excel.")

    df.columns = [str(column).strip() for column in df.columns]
    if "Date" not in df.columns:
        raise ValueError("Input data must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Date"])
    return df


def load_holdings_file(uploaded_file) -> dict[str, float]:
    if uploaded_file is None:
        return {}

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported holdings file format. Upload CSV or Excel.")

    df.columns = [str(column).strip() for column in df.columns]
    required_columns = {"Ticker", "Weight"}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError("Holdings file must contain `Ticker` and `Weight` columns.")

    holdings = {
        str(row["Ticker"]).strip().upper(): float(row["Weight"])
        for _, row in df.iterrows()
        if pd.notna(row["Ticker"]) and pd.notna(row["Weight"])
    }
    if not holdings:
        raise ValueError("No valid holdings were found in the uploaded file.")
    return holdings


def normalize_returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    value_columns = [column for column in result.columns if column != "Date"]
    for column in value_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values("Date").dropna(subset=["Date"])


def parse_holdings_text(raw_text: str) -> dict[str, float]:
    holdings: dict[str, float] = {}
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if "," in cleaned:
            ticker, weight = cleaned.split(",", 1)
        else:
            parts = cleaned.split()
            if len(parts) != 2:
                raise ValueError("Holdings text must be in `TICKER,WEIGHT` format.")
            ticker, weight = parts

        holdings[ticker.strip().upper()] = float(weight.strip())

    if not holdings:
        raise ValueError("No holdings were provided.")
    return holdings


def fetch_price_history(
    tickers: list[str],
    start_date: str,
    end_date: str,
    frequency: str,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required.")

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise ValueError("No price data returned from yfinance for the selected tickers.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded price history did not include close prices.")
        close_prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded price history did not include close prices.")
        close_prices = raw[["Close"]].copy()
        close_prices.columns = [tickers[0].upper()]

    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices.columns = [str(column).upper() for column in close_prices.columns]

    if frequency == "Weekly":
        close_prices = close_prices.resample("W-FRI").last()
    elif frequency == "Monthly":
        close_prices = close_prices.resample("ME").last()

    return close_prices.dropna(how="all")


def build_portfolio_returns_from_prices(
    prices: pd.DataFrame,
    weights: dict[str, float],
    portfolio_name: str = "Portfolio",
) -> pd.DataFrame:
    normalized_weights = pd.Series(weights, dtype=float)
    if np.isclose(normalized_weights.sum(), 0):
        raise ValueError("Holdings weights must sum to a non-zero value.")

    normalized_weights = normalized_weights / normalized_weights.sum()
    returns = prices.pct_change().dropna(how="all")
    usable_columns = [column for column in normalized_weights.index if column in returns.columns]
    if not usable_columns:
        raise ValueError("No usable price history was found for the selected holdings.")

    aligned_returns = returns[usable_columns].dropna()
    portfolio_returns = aligned_returns.mul(normalized_weights[usable_columns], axis=1).sum(axis=1)
    return pd.DataFrame({"Date": portfolio_returns.index, portfolio_name: portfolio_returns.values})


def fetch_fama_french_factors(
    dataset_name: str,
    frequency: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    dataset_map = {
        "3 Factor": {
            "Monthly": "F-F_Research_Data_Factors",
            "Weekly": "F-F_Research_Data_Factors_weekly",
            "Daily": "F-F_Research_Data_Factors_daily",
        },
        "5 Factor": {
            "Monthly": "F-F_Research_Data_5_Factors_2x3",
            "Daily": "F-F_Research_Data_5_Factors_2x3_daily",
        },
    }

    if dataset_name not in dataset_map or frequency not in dataset_map[dataset_name]:
        raise ValueError(f"{dataset_name} factors are not available for {frequency.lower()} frequency.")

    ff_data = web.DataReader(
        dataset_map[dataset_name][frequency],
        "famafrench",
        start=start_date,
        end=end_date,
    )[0].copy()
    ff_data.index.name = "Date"
    ff_data = ff_data.reset_index()

    if frequency == "Monthly":
        ff_data["Date"] = ff_data["Date"].dt.to_timestamp("M")
    else:
        ff_data["Date"] = pd.to_datetime(ff_data["Date"])

    value_columns = [column for column in ff_data.columns if column != "Date"]
    for column in value_columns:
        ff_data[column] = pd.to_numeric(ff_data[column], errors="coerce") / 100

    return ff_data.dropna().sort_values("Date")


def prepare_analysis_frame(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    target_column: str,
    factor_columns: list[str],
    risk_free_column: str | None = None,
) -> pd.DataFrame:
    merged = returns_df.merge(factors_df, on="Date", how="inner")
    needed = [target_column, *factor_columns]
    if risk_free_column:
        needed.append(risk_free_column)

    frame = merged[["Date", *needed]].copy().dropna()
    if risk_free_column:
        frame["TargetExcess"] = frame[target_column] - frame[risk_free_column]
    else:
        frame["TargetExcess"] = frame[target_column]

    return frame


def run_factor_regression(
    frame: pd.DataFrame,
    factor_columns: list[str],
    periods_per_year: int,
) -> RegressionResult:
    y = frame["TargetExcess"]
    x = sm.add_constant(frame[factor_columns], has_constant="add")
    model = sm.OLS(y, x).fit()

    params = model.params.rename("Coefficient")
    p_values = model.pvalues.rename("P-Value")
    t_stats = model.tvalues.rename("T-Stat")
    conf_int = model.conf_int()
    conf_int.columns = ["CI Lower", "CI Upper"]

    try:
        _, bp_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
    except Exception:
        bp_pvalue = np.nan

    summary_table = pd.concat([params, t_stats, p_values, conf_int], axis=1)
    # Factor-model alpha is additive, so annualize the periodic intercept arithmetically.
    annualized_alpha = model.params["const"] * periods_per_year

    return RegressionResult(
        summary_table=summary_table,
        # Use arrays to avoid pandas aligning the model's integer index to date labels.
        fitted_series=pd.Series(model.fittedvalues.to_numpy(), index=frame["Date"].to_numpy(), name="Fitted"),
        residual_series=pd.Series(model.resid.to_numpy(), index=frame["Date"].to_numpy(), name="Residual"),
        r_squared=float(model.rsquared),
        adjusted_r_squared=float(model.rsquared_adj),
        annualized_alpha=float(annualized_alpha),
        f_statistic=float(model.fvalue) if model.fvalue is not None else np.nan,
        f_pvalue=float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        durbin_watson=float(durbin_watson(model.resid)),
        breusch_pagan_pvalue=float(bp_pvalue),
    )


def compute_rolling_betas(
    frame: pd.DataFrame,
    factor_columns: list[str],
    window: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    if len(frame) < window:
        return pd.DataFrame()

    for end_idx in range(window, len(frame) + 1):
        sample = frame.iloc[end_idx - window : end_idx]
        y = sample["TargetExcess"]
        x = sm.add_constant(sample[factor_columns], has_constant="add")
        model = sm.OLS(y, x).fit()

        row: dict[str, float | pd.Timestamp] = {"Date": sample["Date"].iloc[-1]}
        for factor in factor_columns:
            row[factor] = float(model.params.get(factor, np.nan))
        rows.append(row)

    return pd.DataFrame(rows)


def compute_rolling_regression_diagnostics(
    frame: pd.DataFrame,
    factor_columns: list[str],
    window: int,
    periods_per_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    if len(frame) < window:
        return pd.DataFrame()

    for end_idx in range(window, len(frame) + 1):
        sample = frame.iloc[end_idx - window : end_idx]
        y = sample["TargetExcess"]
        x = sm.add_constant(sample[factor_columns], has_constant="add")
        model = sm.OLS(y, x).fit()

        alpha = float(model.params.get("const", np.nan))
        alpha_annualized = alpha * periods_per_year if pd.notna(alpha) else np.nan
        rows.append(
            {
                "Date": sample["Date"].iloc[-1],
                "Alpha": alpha,
                "Annualized Alpha": alpha_annualized,
                "R-Squared": float(model.rsquared),
                "Alpha Standard Error": float(model.bse.get("const", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def compute_style_consistency(rolling_betas: pd.DataFrame, factor_columns: list[str]) -> pd.DataFrame:
    if rolling_betas.empty:
        return pd.DataFrame()

    records = []
    for factor in factor_columns:
        series = rolling_betas[factor].dropna()
        if series.empty:
            continue

        volatility = float(series.std(ddof=0))
        consistency_score = float(1 / (1 + volatility))
        if consistency_score >= 0.80:
            assessment = "Stable"
        elif consistency_score >= 0.70:
            assessment = "Watch"
        else:
            assessment = "Style drift"

        records.append(
            {
                "Factor": factor,
                "Average Beta": float(series.mean()),
                "Beta Volatility": volatility,
                "Min Beta": float(series.min()),
                "Max Beta": float(series.max()),
                "Style Consistency Score": consistency_score,
                "Assessment": assessment,
            }
        )

    return pd.DataFrame(records).sort_values("Style Consistency Score", ascending=False)


def compute_contribution_table(
    frame: pd.DataFrame,
    summary_table: pd.DataFrame,
    factor_columns: list[str],
) -> pd.DataFrame:
    contributions = []
    for factor in factor_columns:
        avg_factor_return = frame[factor].mean()
        beta = summary_table.loc[factor, "Coefficient"]
        contributions.append(
            {
                "Factor": factor,
                "Average Factor Return": float(avg_factor_return),
                "Beta": float(beta),
                "Estimated Contribution": float(avg_factor_return * beta),
            }
        )

    alpha = summary_table.loc["const", "Coefficient"]
    contributions.append(
        {
            "Factor": "Alpha",
            "Average Factor Return": np.nan,
            "Beta": np.nan,
            "Estimated Contribution": float(alpha),
        }
    )
    return pd.DataFrame(contributions)


def compute_factor_return_summary(
    frame: pd.DataFrame,
    factor_columns: list[str],
    target_column: str,
    annualized_alpha: float,
    r_squared: float,
    periods_per_year: int,
) -> pd.DataFrame:
    row: dict[str, Any] = {
        "Target": target_column,
        "First Return Date": frame["Date"].min(),
        "Last Return Date": frame["Date"].max(),
        "Annualized Alpha": annualized_alpha,
        "R-Squared": r_squared,
    }
    for factor in factor_columns:
        row[f"{factor} Annual Premium"] = float(frame[factor].mean() * periods_per_year)
    return pd.DataFrame([row])


def compute_risk_return_attribution(
    frame: pd.DataFrame,
    summary_table: pd.DataFrame,
    factor_columns: list[str],
    target_column: str,
    periods_per_year: int,
    risk_free_column: str,
) -> pd.DataFrame:
    target_returns = frame[target_column]
    row: dict[str, Any] = {
        "Target": target_column,
        "First Return Date": frame["Date"].min(),
        "Last Return Date": frame["Date"].max(),
        "Cumulative Return": float((1 + target_returns).prod() - 1),
        "Annualized Return": float(annualize_return(target_returns, periods_per_year)),
        "Annualized Volatility": float(target_returns.std(ddof=0) * np.sqrt(periods_per_year)),
        "Arithmetic Annualized Excess Return": float(frame["TargetExcess"].mean() * periods_per_year),
        "Annualized Risk-Free Return": float(annualize_return(frame[risk_free_column], periods_per_year)),
    }

    for factor in factor_columns:
        beta = float(summary_table.loc[factor, "Coefficient"])
        row[f"{factor} Arithmetic Contribution"] = float(
            frame[factor].mean() * periods_per_year * beta
        )

    row["Alpha Arithmetic Contribution"] = float(
        summary_table.loc["const", "Coefficient"] * periods_per_year
    )
    return pd.DataFrame([row])


def compute_cumulative_return_series(return_series: pd.Series) -> pd.Series:
    return (1 + return_series.fillna(0)).cumprod() - 1


def compute_cumulative_attribution(
    frame: pd.DataFrame,
    summary_table: pd.DataFrame,
    factor_columns: list[str],
) -> pd.DataFrame:
    data = pd.DataFrame({"Date": frame["Date"]})
    explained_periods = pd.DataFrame(index=frame.index)

    for factor in factor_columns:
        explained_periods[factor] = frame[factor] * float(summary_table.loc[factor, "Coefficient"])
        data[factor] = explained_periods[factor].cumsum()

    explained_periods["Alpha"] = float(summary_table.loc["const", "Coefficient"])
    data["Alpha"] = explained_periods["Alpha"].cumsum()
    data["Fitted Excess"] = explained_periods.sum(axis=1).cumsum()
    data["Actual Excess"] = frame["TargetExcess"].cumsum()
    data["Residual"] = data["Actual Excess"] - data["Fitted Excess"]
    return data


def compute_rolling_window_returns(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window).apply(np.prod, raw=True) - 1


def build_factor_significance_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, values in summary_table.iterrows():
        label = "Alpha" if factor == "const" else factor
        p_value = float(values["P-Value"])
        ci_lower = float(values["CI Lower"])
        ci_upper = float(values["CI Upper"])
        significant = p_value < 0.05
        ci_excludes_zero = ci_lower > 0 or ci_upper < 0

        if significant and ci_excludes_zero:
            why = "P-value is below 0.05 and the 95% confidence interval stays away from zero."
        elif significant:
            why = "P-value is below 0.05, so the coefficient is unlikely to be zero under the model assumptions."
        elif ci_lower <= 0 <= ci_upper:
            why = "The confidence interval crosses zero, so the direction and existence of the effect are uncertain."
        else:
            why = "Evidence is weak even though the interval is near one side of zero; treat this effect cautiously."

        rows.append(
            {
                "Factor": label,
                "Coefficient": float(values["Coefficient"]),
                "T-Stat": float(values["T-Stat"]),
                "P-Value": p_value,
                "CI Lower": ci_lower,
                "CI Upper": ci_upper,
                "Significant": "Yes" if significant else "No",
                "Why": why,
            }
        )

    return pd.DataFrame(rows)


def build_regression_glossary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "T-Stat",
                "What it means": "How many standard errors the estimated coefficient is away from zero.",
                "How to read it": "Larger absolute values mean stronger evidence that the factor exposure is not zero. Values above about 2 in absolute value are often treated as meaningful.",
            },
            {
                "Metric": "P-Value",
                "What it means": "The probability of seeing a coefficient this extreme if the true coefficient were actually zero.",
                "How to read it": "Smaller is stronger. A value below 0.05 is a common threshold for statistical significance.",
            },
            {
                "Metric": "CI Lower",
                "What it means": "The lower bound of the model's 95% confidence interval for the coefficient.",
                "How to read it": "If both CI bounds are above zero, the factor exposure is reliably positive. If both are below zero, it is reliably negative.",
            },
            {
                "Metric": "CI Upper",
                "What it means": "The upper bound of the model's 95% confidence interval for the coefficient.",
                "How to read it": "If the interval from CI Lower to CI Upper crosses zero, the estimated effect is not statistically distinct from zero at roughly the 95% level.",
            },
        ]
    )


def compute_bt_risk_return_stats(
    return_series: pd.Series,
    strategy_name: str,
    date_index: pd.Series | pd.Index | None = None,
    risk_free_series: pd.Series | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    inputs = pd.DataFrame({"Return": return_series})
    inputs["RF"] = 0.0 if risk_free_series is None else risk_free_series
    inputs = inputs.dropna()
    if inputs.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    clean = inputs["Return"].copy()
    periodic_rf = inputs["RF"].copy()
    excess_returns = clean - periodic_rf

    excess_volatility = float(excess_returns.std(ddof=1))
    sharpe = (
        float(excess_returns.mean() / excess_volatility * np.sqrt(periods_per_year))
        if excess_volatility > 0
        else np.nan
    )
    downside_deviation = float(np.sqrt(np.mean(np.square(np.minimum(excess_returns, 0.0)))))
    sortino = (
        float(excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year))
        if downside_deviation > 0
        else np.nan
    )

    if date_index is not None:
        aligned_dates = pd.to_datetime(pd.Series(date_index).iloc[clean.index])
        clean.index = pd.DatetimeIndex(aligned_dates)
    elif not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("bt risk/return stats require a datetime index.")

    initial_date = clean.index[0] - pd.DateOffset(days=1)
    price_index = pd.DatetimeIndex([initial_date]).append(pd.DatetimeIndex(clean.index))
    prices = np.concatenate(([100.0], 100 * (1 + clean).cumprod().to_numpy()))
    price_frame = pd.DataFrame({strategy_name: prices}, index=price_index)

    strategy = bt.Strategy(
        strategy_name,
        [bt.algos.RunOnce(), bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()],
    )
    backtest = bt.Backtest(strategy, price_frame)
    result = bt.run(backtest)
    stats = result.stats[[strategy_name]].copy().reset_index()
    stats.columns = ["Metric", "Value"]
    # bt uses synthetic initialization dates to seed its price index. Replace
    # those implementation details with the actual return dates users selected.
    stats = stats[~stats["Metric"].astype(str).str.lower().isin(["start", "end"])]
    stats = stats[
        ~stats["Metric"].astype(str).str.contains("sharpe|sortino", case=False, regex=True)
    ]
    custom_ratios = pd.DataFrame(
        {
            "Metric": ["Sharpe (Actual Periodic RF)", "Sortino (Actual Periodic RF)"],
            "Value": [sharpe, sortino],
        }
    )
    return_dates = pd.DataFrame(
        {
            "Metric": ["First Return Date", "Last Return Date"],
            "Value": [clean.index.min(), clean.index.max()],
        }
    )
    stats = pd.concat([return_dates, custom_ratios, stats], ignore_index=True)
    stats = stats.dropna(subset=["Value"])
    return stats


def format_bt_stat_value(metric: str, value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, (np.floating, float, np.integer, int)):
        metric_lower = metric.lower()
        percent_keywords = [
            "return",
            "cagr",
            "drawdown",
            "mean",
            "vol",
            "ytd",
            "mtd",
            "incep",
            "month",
            "year",
            "rf",
            "win",
        ]
        ratio_keywords = ["sharpe", "sortino", "calmar"]
        day_keywords = ["days"]

        if any(keyword in metric_lower for keyword in day_keywords):
            return f"{value:,.0f}"
        if any(keyword in metric_lower for keyword in ratio_keywords):
            return f"{value:.2f}"
        if any(keyword in metric_lower for keyword in percent_keywords):
            return f"{value:.2%}"
        return f"{value:.4f}"

    return str(value)


def annualize_return(series: pd.Series, periods_per_year: int) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    compounded = float((1 + clean).prod())
    years = len(clean) / periods_per_year
    if years <= 0:
        return float("nan")
    return compounded ** (1 / years) - 1