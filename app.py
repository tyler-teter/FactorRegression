from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    annualize_return,
    build_factor_significance_table,
    build_portfolio_returns_from_prices,
    build_regression_glossary,
    compute_bt_risk_return_stats,
    compute_contribution_table,
    compute_cumulative_attribution,
    compute_cumulative_return_series,
    compute_factor_return_summary,
    compute_fixed_income_factors,
    compute_risk_return_attribution,
    compute_rolling_betas,
    compute_rolling_regression_diagnostics,
    compute_rolling_window_returns,
    compute_style_consistency,
    fetch_aqr_equity_factors,
    fetch_fama_french_factors,
    fetch_price_history,
    format_bt_stat_value,
    load_holdings_file,
    load_returns_file,
    normalize_returns_frame,
    parse_holdings_text,
    prepare_analysis_frame,
    run_factor_regression,
)


st.set_page_config(
    page_title="Factor Regression Lab",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


def format_value(value, kind: str = "number") -> str:
    if pd.isna(value):
        return "-"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "pvalue":
        if value == 0:
            return "<1e-300"
        if value < 0.0001:
            return f"{value:.2e}"
        return f"{value:.4f}"
    if kind == "integer":
        return f"{int(value):,}"
    if kind == "date":
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return f"{value:.4f}"


def format_dataframe(df: pd.DataFrame, percent_cols: list[str] | None = None, pvalue_cols: list[str] | None = None, date_cols: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    percent_cols = percent_cols or []
    pvalue_cols = pvalue_cols or []
    date_cols = date_cols or []

    for column in result.columns:
        if column in percent_cols:
            result[column] = result[column].apply(lambda x: format_value(x, "percent"))
        elif column in pvalue_cols:
            result[column] = result[column].apply(lambda x: format_value(x, "pvalue"))
        elif column in date_cols:
            result[column] = result[column].apply(lambda x: format_value(x, "date"))
        elif pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].apply(lambda x: format_value(x, "number"))
    return result


STYLE_CONSISTENCY_HELP = (
    "For each factor, the app takes the rolling beta series and calculates its volatility "
    "as the standard deviation of those rolling beta estimates. The style consistency score is "
    "`1 / (1 + beta volatility)`, so lower beta volatility produces a score closer to 1. "
    "The note reports the average score across selected factors and the weakest individual factor score. "
    "The app classifies the profile as relatively stable when the average score is at least 0.80 "
    "and every individual factor score is at least 0.70."
)


COLUMN_HELP = {
    "Metric": "The regression or portfolio statistic being reported.",
    "Value": "The calculated value for the selected analysis period.",
    "Factor": "The explanatory return factor or regression intercept (Alpha).",
    "Coefficient": "Estimated sensitivity to the factor; a one-unit factor move implies this change in excess return.",
    "T-Stat": "Coefficient divided by its standard error; larger absolute values provide stronger evidence the exposure differs from zero.",
    "P-Value": "Probability of observing a result this extreme if the true coefficient were zero; below 0.05 is commonly significant.",
    "CI Lower": "Lower endpoint of the estimated 95% confidence interval.",
    "CI Upper": "Upper endpoint of the estimated 95% confidence interval.",
    "Significant": "Yes when the p-value is below 0.05 and the 95% confidence interval does not include zero.",
    "Interpretation": "Plain-language reading of the statistical result.",
    "Average Factor Return": "Average periodic return of this factor over the matched sample.",
    "Estimated Contribution": "Average factor return multiplied by the estimated factor loading.",
    "Annualized Alpha": "Periodic regression intercept multiplied by the number of periods per year.",
    "Residual": "Cumulative actual excess return not explained by alpha and the selected factor contributions.",
    "Annualized Risk-Free Return": "Geometrically annualized Ken French risk-free return over the matched sample.",
    "Arithmetic Annualized Excess Return": "Average periodic excess return multiplied by the number of periods per year.",
    "R-Squared": "Share of variation in excess returns explained by the regression.",
    "First Return Date": "First overlapping return observation used in the analysis.",
    "Last Return Date": "Last overlapping return observation used in the analysis.",
    "Style Consistency Score": "A higher score indicates that the rolling factor exposure has been more stable.",
    "Assessment": "Stable is 0.80 or higher, Watch is 0.70 to 0.79, and Style drift is below 0.70.",
    "Average Beta": "Average rolling sensitivity to the factor.",
    "Beta Volatility": "Standard deviation of the rolling beta; lower values indicate a more stable exposure.",
    "Min Beta": "Lowest rolling factor sensitivity observed during the analysis period.",
    "Max Beta": "Highest rolling factor sensitivity observed during the analysis period.",
    "Average Loading": "Average rolling sensitivity to the factor.",
    "Loading Volatility": "Standard deviation of the rolling factor loading; lower values indicate greater stability.",
    "Check": "A data-availability or model-input validation item.",
}


def show_table(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    """Render a table with hover definitions on its column headers."""
    column_config = {
        column: st.column_config.Column(
            column,
            help=COLUMN_HELP.get(column, f"Reported {column.lower()} value."),
        )
        for column in df.columns
    }
    st.dataframe(df, use_container_width=True, hide_index=hide_index, column_config=column_config)


FF_FACTOR_OPTIONS = ["Mkt-RF", "SMB", "HML", "Mom", "RMW", "CMA"]
AQR_FACTOR_OPTIONS = ["MKT", "SMB", "HML", "HML-DEV", "UMD", "QMJ", "BAB"]
FIXED_INCOME_FACTOR_OPTIONS = ["TRM", "CDT"]

FF_FACTOR_HELP = {
    "Mkt-RF": "Market minus risk-free. The broad equity market's excess return over Treasury bills; captures market beta.",
    "SMB": "Small minus big. Return premium of small-cap stocks over large-cap stocks; captures size exposure.",
    "HML": "High minus low. Return premium of value stocks over growth stocks; captures value exposure.",
    "Mom": "Momentum. Return premium of recent winners over recent losers; captures trend-following equity momentum.",
    "RMW": "Robust minus weak. Return premium of highly profitable companies over weakly profitable companies; captures profitability quality.",
    "CMA": "Conservative minus aggressive. Return premium of firms with conservative investment over aggressive investment; captures investment style.",
}

AQR_FACTOR_HELP = {
    "MKT": "Market factor. AQR's market return in excess of cash for the selected region.",
    "SMB": "Small minus big. AQR's size factor: long smaller stocks and short larger stocks in the selected region.",
    "HML": "High minus low. AQR's value factor using book-to-market style value definitions.",
    "HML-DEV": "HML Devil. AQR's value factor variant using a more timely value measure from Asness and Frazzini's value work.",
    "UMD": "Up minus down. AQR's momentum factor: long stocks with strong recent returns and short weak recent performers.",
    "QMJ": "Quality minus junk. Long higher-quality companies and short lower-quality companies based on profitability, growth, safety, and payout quality.",
    "BAB": "Betting against beta. Long low-beta stocks and short high-beta stocks, designed to capture the low-beta anomaly.",
}

FIXED_INCOME_FACTOR_HELP = {
    "TRM": "Term risk. Long-term Treasury bond return minus Treasury bill return; captures compensation for interest-rate duration exposure.",
    "CDT": "Credit risk. Long-term corporate bond return minus long-term Treasury bond return; captures corporate credit spread exposure.",
}

KEN_FRENCH_LIBRARY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
AQR_QMJ_FACTORS_URL = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx"
AQR_BAB_FACTORS_URL = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx"
YFINANCE_VUSTX_URL = "https://finance.yahoo.com/quote/VUSTX/history/"
YFINANCE_VLTCX_URL = "https://finance.yahoo.com/quote/VLTCX/history/"



def aqr_factor_column(region_name: str, factor_name: str) -> str:
    region_label = "US" if region_name == "US" else "DEV"
    return f"AQR {region_label} {factor_name}"


def merge_factor_frame(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty:
        return left
    if left.empty:
        return right

    duplicate_columns = [column for column in right.columns if column != "Date" and column in left.columns]
    right = right.drop(columns=duplicate_columns)
    if len(right.columns) == 1:
        return left
    return left.merge(right, on="Date", how="outer")


def build_factor_glossary_table() -> pd.DataFrame:
    rows = []
    for factor in FF_FACTOR_OPTIONS:
        rows.append(
            {
                "Family": "Fama-French Equity Factors",
                "Factor": factor,
                "Meaning": FF_FACTOR_HELP[factor],
                "Data Pulled From": "Ken French Data Library",
                "Calculation Notes": "Downloaded directly through pandas-datareader. Ken French percent returns are converted to decimals in the app.",
                "Source URL": KEN_FRENCH_LIBRARY_URL,
            }
        )

    for factor in AQR_FACTOR_OPTIONS:
        source_url = AQR_BAB_FACTORS_URL if factor == "BAB" else AQR_QMJ_FACTORS_URL
        rows.append(
            {
                "Family": "AQR Equity Factors",
                "Factor": factor,
                "Meaning": AQR_FACTOR_HELP[factor],
                "Data Pulled From": "AQR data workbook",
                "Calculation Notes": "Monthly AQR factor returns are read from the USA and Global Ex USA columns for the selected region.",
                "Source URL": source_url,
            }
        )

    rows.extend(
        [
            {
                "Family": "Fixed Income Factors",
                "Factor": "TRM",
                "Meaning": FIXED_INCOME_FACTOR_HELP["TRM"],
                "Data Pulled From": "VUSTX price history and Ken French RF",
                "Calculation Notes": "TRM = Vanguard Long-Term Treasury Bond Index return minus Ken French RF.",
                "Source URL": YFINANCE_VUSTX_URL,
            },
            {
                "Family": "Fixed Income Factors",
                "Factor": "CDT",
                "Meaning": FIXED_INCOME_FACTOR_HELP["CDT"],
                "Data Pulled From": "VLTCX and VUSTX price histories",
                "Calculation Notes": "CDT = Vanguard Long-Term Corporate Bond Index return minus Vanguard Long-Term Treasury Bond Index return.",
                "Source URL": YFINANCE_VLTCX_URL,
            },
            {
                "Family": "Risk-Free Rate",
                "Factor": "RF",
                "Meaning": "Risk-free return used to convert target returns into excess returns.",
                "Data Pulled From": "Ken French Data Library",
                "Calculation Notes": "TargetExcess = target return minus the same-period Ken French RF observation.",
                "Source URL": KEN_FRENCH_LIBRARY_URL,
            },
        ]
    )
    return pd.DataFrame(rows)


def checkbox_row(
    label: str,
    options: list[str],
    default_selected: list[str] | None = None,
    disabled_options: set[str] | None = None,
    disabled: bool = False,
    key_prefix: str = "factor",
    help_by_option: dict[str, str] | None = None,
) -> list[str]:
    st.markdown(f"**{label}**")
    selected: list[str] = []
    default_selected = default_selected or []
    disabled_options = disabled_options or set()
    help_by_option = help_by_option or {}

    for start in range(0, len(options), 3):
        columns = st.columns(3)
        for column, option in zip(columns, options[start : start + 3]):
            option_disabled = disabled or option in disabled_options
            with column:
                checked = st.checkbox(
                    option,
                    value=option in default_selected and not option_disabled,
                    disabled=option_disabled,
                    key=f"{key_prefix}_{option}",
                    help=help_by_option.get(option),
                )
            if checked and not option_disabled:
                selected.append(option)
    return selected


def polish_chart(fig: go.Figure, hover_format: str = ".2%") -> go.Figure:
    """Apply consistent hover labels to each chart trace."""
    fig.update_traces(
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{fullData.name}}: %{{y:{hover_format}}}<extra></extra>"
    )
    fig.update_layout(hovermode="x unified")
    return fig


def build_rolling_beta_analysis(rolling_betas: pd.DataFrame, factor_columns: list[str], rolling_window: int) -> str:
    """Create a plain-English readout for the rolling factor sensitivity chart."""
    if rolling_betas.empty:
        return ""

    beta_frame = rolling_betas[factor_columns].dropna(how="all")
    if beta_frame.empty:
        return ""

    avg_abs_beta = beta_frame.abs().mean().dropna()
    beta_changes = (beta_frame.ffill().iloc[-1] - beta_frame.bfill().iloc[0]).dropna()
    beta_volatility = beta_frame.std().dropna()

    if avg_abs_beta.empty:
        return ""

    dominant_factor = avg_abs_beta.idxmax()
    dominant_avg = beta_frame[dominant_factor].mean()

    drift_sentence = ""
    if not beta_changes.empty:
        drift_factor = beta_changes.abs().idxmax()
        drift_amount = beta_changes[drift_factor]
        drift_direction = "increased" if drift_amount > 0 else "decreased"
        drift_sentence = (
            f"The largest endpoint-to-endpoint shift is in {drift_factor}, which {drift_direction} "
            f"by {abs(drift_amount):.2f} beta points over the rolling sample. "
        )

    stability_sentence = ""
    if not beta_volatility.empty:
        most_stable = beta_volatility.idxmin()
        least_stable = beta_volatility.idxmax()
        if most_stable == least_stable:
            stability_sentence = f"The rolling estimate has a beta volatility of {beta_volatility[most_stable]:.2f}. "
        else:
            stability_sentence = (
                f"{most_stable} is the most stable exposure, while {least_stable} varies the most across windows. "
            )

    return (
        f"Interpretation: each point is a local regression coefficient estimated from the prior "
        f"{rolling_window} periods, so the lines show how the fund's factor-loading vector changes through time. "
        f"On average, the largest absolute exposure is {dominant_factor} with an average beta of "
        f"{dominant_avg:.2f}. {drift_sentence}{stability_sentence}"
        "Read persistent level shifts as possible style drift; read short spikes cautiously because rolling "
        "regressions can be noisy when the window is small or factor returns are highly correlated."
    )


def build_rolling_sensitivity_workbook(
    frame: pd.DataFrame,
    rolling_betas: pd.DataFrame,
    rolling_diag: pd.DataFrame,
    factor_columns: list[str],
    target_column: str,
    rolling_window: int,
    frequency: str,
) -> bytes:
    """Create an Excel workbook explaining and auditing the rolling beta chart."""
    output = BytesIO()
    rolling_export = rolling_betas.copy()
    if not rolling_diag.empty:
        diagnostic_columns = [column for column in ["Date", "Alpha", "Annualized Alpha", "R-Squared"] if column in rolling_diag.columns]
        rolling_export = rolling_export.merge(rolling_diag[diagnostic_columns], on="Date", how="left")

    input_columns = [column for column in ["Date", target_column, "TargetExcess", "RF", *factor_columns] if column in frame.columns]
    input_export = frame[input_columns].copy()

    latest_window = frame.tail(rolling_window).copy()
    latest_input_columns = [column for column in ["Date", "TargetExcess", *factor_columns] if column in latest_window.columns]
    latest_window = latest_window[latest_input_columns]

    latest_coefficients = []
    if not rolling_betas.empty:
        latest_beta_row = rolling_betas.dropna(how="all").iloc[-1]
        latest_date = latest_beta_row["Date"]
        latest_coefficients.extend(
            {"Term": factor, "Coefficient": latest_beta_row.get(factor, np.nan)}
            for factor in factor_columns
        )
        if not rolling_diag.empty:
            latest_diag_row = rolling_diag.loc[rolling_diag["Date"] == latest_date]
            if not latest_diag_row.empty:
                latest_coefficients.insert(0, {"Term": "Alpha", "Coefficient": latest_diag_row.iloc[0].get("Alpha", np.nan)})
                latest_coefficients.append({"Term": "R-Squared", "Coefficient": latest_diag_row.iloc[0].get("R-Squared", np.nan)})
    latest_coefficients_df = pd.DataFrame(latest_coefficients)

    readme_rows = [
        ["Rolling Factor Sensitivities Workbook", ""],
        ["Purpose", "Audit the data and math behind the Rolling Factor Sensitivities chart."],
        ["Frequency", frequency],
        ["Rolling window", f"{rolling_window} periods"],
        ["Target return column", target_column],
        ["Selected factors", ", ".join(factor_columns)],
        ["Model", "TargetExcess_t = alpha + beta_1*Factor_1_t + ... + beta_k*Factor_k_t + error_t"],
        ["Rolling method", f"For each chart date, the app uses the trailing {rolling_window} observations ending on that date and refits the regression."],
        ["OLS estimator", "beta_hat = (X'X)^(-1) X'y, where X contains a constant column plus the selected factor returns."],
        ["Chart intuition", "Each plotted line is a time series of local regression slopes. Higher beta means more sensitivity to that factor in that trailing window."],
        ["Stable style", "A relatively flat line means that factor exposure was stable. A persistent level shift can indicate style drift."],
        ["Noise warning", "Short spikes can be noise, especially with small windows, volatile returns, or correlated factors."],
        ["Rolling Betas sheet", "The exact beta values plotted in the chart, one row per rolling-window end date."],
        ["Regression Inputs sheet", "The return observations used by the regression after date alignment and excess-return calculation."],
        ["Latest Window Math sheet", "The latest rolling window's input observations, coefficients, and Excel LINEST formula reference."],
    ]
    readme_df = pd.DataFrame(readme_rows, columns=["Topic", "Explanation"])

    formula_rows = [
        ["Dependent variable", "TargetExcess = target return minus RF."],
        ["Independent variables", "The selected factor return columns."],
        ["Per-window regression", f"Use the trailing {rolling_window} rows for each ending date."],
        ["Excel formula reference", "=LINEST(y_range, x_range, TRUE, TRUE)"],
        ["Important LINEST nuance", "When multiple X columns are supplied, Excel returns slope coefficients in reverse X-column order, with the intercept at the end of the first row."],
        ["App calculation", "The app uses statsmodels OLS with an explicit constant column; coefficients are displayed in normal factor order."],
    ]
    formula_df = pd.DataFrame(formula_rows, columns=["Item", "Detail"])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        readme_df.to_excel(writer, sheet_name="Read Me", index=False)
        rolling_export.to_excel(writer, sheet_name="Rolling Betas", index=False)
        input_export.to_excel(writer, sheet_name="Regression Inputs", index=False)
        latest_window.to_excel(writer, sheet_name="Latest Window Math", index=False, startrow=8)
        latest_coefficients_df.to_excel(writer, sheet_name="Latest Window Math", index=False, startrow=1, startcol=6)
        formula_df.to_excel(writer, sheet_name="Formula Reference", index=False)

        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.style = "Headline 4"
            for column_cells in sheet.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 55)

        readme_sheet = workbook["Read Me"]
        readme_sheet.column_dimensions["A"].width = 28
        readme_sheet.column_dimensions["B"].width = 95
        for row in readme_sheet.iter_rows(min_row=2, max_col=2):
            row[1].alignment = row[1].alignment.copy(wrap_text=True, vertical="top")

        math_sheet = workbook["Latest Window Math"]
        math_sheet["A1"] = "Latest rolling window input observations"
        math_sheet["G1"] = "Latest rolling-window regression outputs"
        math_sheet["G8"] = "To reproduce in Excel with LINEST"
        math_sheet["G9"] = "Use TargetExcess as y_range and the selected factor columns as x_range."
        latest_start_row = 10
        latest_end_row = latest_start_row + len(latest_window) - 1
        y_range = f"B{latest_start_row}:B{latest_end_row}"
        x_end_column = math_sheet.cell(row=9, column=len(latest_input_columns)).column_letter
        x_range = f"C{latest_start_row}:{x_end_column}{latest_end_row}"
        math_sheet["G10"] = f"Example pattern: =LINEST({y_range},{x_range},TRUE,TRUE)"
        math_sheet["G11"] = "Excel returns multiple-regression slopes in reverse X-column order; the app displays them in normal factor order."
        for cell in ["A1", "G1", "G8"]:
            math_sheet[cell].style = "Headline 4"
        for row in math_sheet.iter_rows(min_row=9, min_col=7, max_col=8):
            for cell in row:
                cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")

        percent_like_sheets = ["Rolling Betas", "Regression Inputs", "Latest Window Math"]
        for sheet_name in percent_like_sheets:
            sheet = workbook[sheet_name]
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    header = sheet.cell(row=1, column=cell.column).value
                    if header == "Date" or cell.value is None:
                        continue
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "0.0000"

        for sheet_name in ["Rolling Betas", "Regression Inputs"]:
            sheet = workbook[sheet_name]
            for cell in sheet["A"]:
                if cell.row > 1:
                    cell.number_format = "yyyy-mm-dd"

    return output.getvalue()


st.title("Factor Regression Lab")
st.caption(
    "Built by Tyler Teter, CFP®, CFA | "
    "[LinkedIn](https://www.linkedin.com/in/tylerteter/)"
)
st.caption(
    "Analyze a fund or portfolio with yfinance price histories, auto-downloaded "
    "Fama-French, AQR, and fixed income factors, rolling exposure analysis, and backtested risk/return diagnostics."
)
st.warning(
    "**Important disclosure:** This application is provided for educational and informational purposes only "
    "and does not constitute investment or financial advice, or a recommendation to buy or sell "
    "any security. Results are based on historical third-party data and statistical models that may contain "
    "errors, assumptions, delays, or omissions. Past performance and modeled results do not guarantee future "
    "outcomes. You are solely responsible for your investment decisions and should consult a qualified "
    "professional before acting on this information."
)

with st.sidebar:
    st.header("Data Setup")
    analysis_mode = st.radio(
        "Target return source",
        options=["Single fund ticker", "Portfolio holdings"],
        index=0,
    )
    frequency = st.selectbox("Return frequency", ["Monthly", "Weekly", "Daily"], index=0)

    st.subheader("Factor Selection")
    ff_disabled = {"RMW", "CMA", "Mom"} if frequency == "Weekly" else set()
    ff_default = ["Mkt-RF", "SMB", "HML"] if frequency == "Weekly" else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    selected_ff_factors = checkbox_row(
        "FF Equity Factors",
        FF_FACTOR_OPTIONS,
        default_selected=ff_default,
        disabled_options=ff_disabled,
        key_prefix="ff_factor",
        help_by_option=FF_FACTOR_HELP,
    )

    aqr_disabled = frequency != "Monthly"
    selected_aqr_us_factors = checkbox_row(
        "AQR US Equity Factors",
        AQR_FACTOR_OPTIONS,
        disabled=aqr_disabled,
        key_prefix="aqr_us_factor",
        help_by_option=AQR_FACTOR_HELP,
    )
    selected_aqr_dev_factors = checkbox_row(
        "AQR Dev ex-US Factors",
        AQR_FACTOR_OPTIONS,
        disabled=aqr_disabled,
        key_prefix="aqr_dev_factor",
        help_by_option=AQR_FACTOR_HELP,
    )
    if aqr_disabled:
        st.caption("AQR workbook factors are monthly, so they are only available for monthly regressions.")

    selected_fixed_income_factors = checkbox_row(
        "Fixed Income Factors",
        FIXED_INCOME_FACTOR_OPTIONS,
        key_prefix="fixed_income_factor",
        help_by_option=FIXED_INCOME_FACTOR_HELP,
    )

    selected_aqr_factors_by_region = {
        "US": selected_aqr_us_factors,
        "Developed ex-US": selected_aqr_dev_factors,
    }
    selected_factor_names = [
        *selected_ff_factors,
        *[aqr_factor_column("US", factor) for factor in selected_aqr_us_factors],
        *[aqr_factor_column("Developed ex-US", factor) for factor in selected_aqr_dev_factors],
        *selected_fixed_income_factors,
    ]
    rolling_window = st.slider("Rolling window length", min_value=12, max_value=60, value=24, step=6)
    start_date = st.date_input("Analysis start date", value=pd.Timestamp("2018-01-01"))
    end_date = st.date_input("Analysis end date", value=pd.Timestamp.today().normalize())
    st.caption("We download an earlier price when needed to calculate the first return in your selected period.")
    period_status_sidebar = st.empty()
    risk_free_sidebar = st.empty()
    st.link_button(
        "Ken French Data Library",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        use_container_width=True,
    )


periods_per_year = {"Monthly": 12, "Weekly": 52, "Daily": 252}[frequency]
selected_start = pd.Timestamp(start_date)
selected_end = pd.Timestamp(end_date)
if selected_start > selected_end:
    st.error("Analysis start date must be on or before analysis end date.")
    st.stop()

price_lookback = {
    "Monthly": pd.DateOffset(months=1, days=7),
    "Weekly": pd.DateOffset(days=10),
    "Daily": pd.DateOffset(days=10),
}[frequency]
price_download_start = selected_start - price_lookback
# yfinance treats end as exclusive, so request the following day.
price_download_end = selected_end + pd.DateOffset(days=1)

st.info(
    "You can analyze a single fund ticker or a weighted portfolio built from holdings. "
    "Factor data is downloaded from Ken French, AQR, and yfinance-backed bond proxies based on your selections."
)

with st.expander("View sample holdings format", expanded=False):
    sample_holdings = pd.DataFrame(
        {
            "Ticker": ["SPY", "IWM", "EFA", "AGG"],
            "Weight": [0.40, 0.20, 0.20, 0.20],
        }
    )
    st.dataframe(sample_holdings, use_container_width=True, hide_index=True)

returns_df = pd.DataFrame()
factors_df = pd.DataFrame()
source_description = ""
factor_source_description = "Selected factor data from Ken French, AQR, and yfinance bond proxies."

if analysis_mode == "Single fund ticker":
    ticker_input = st.text_input("Fund, ETF, or index proxy ticker", value="VFINX").strip().upper()
    target_column_hint = ticker_input
elif analysis_mode == "Portfolio holdings":
    holdings_file = st.file_uploader("Upload holdings file (optional)", type=["csv", "xlsx", "xls"])
    holdings_text = st.text_area(
        "Or paste holdings as `TICKER,WEIGHT`",
        value="SPY,0.40\nIWM,0.20\nEFA,0.20\nAGG,0.20",
        height=140,
    )
    portfolio_name = st.text_input("Portfolio name", value="Model Portfolio").strip() or "Model Portfolio"
    target_column_hint = portfolio_name
else:
    target_file = st.file_uploader("Upload fund or portfolio returns", type=["csv", "xlsx", "xls"])
    target_column_hint = None

if not selected_factor_names:
    st.warning("Choose at least one factor to run the regression.")
    st.stop()

try:
    if analysis_mode == "Single fund ticker":
        if not ticker_input:
            st.warning("Enter a ticker to download returns.")
            st.stop()
        price_history = fetch_price_history(
            [ticker_input], str(price_download_start.date()), str(price_download_end.date()), frequency
        )
        price_returns = price_history.pct_change().dropna(how="all")
        returns_df = pd.DataFrame({"Date": price_returns.index, ticker_input: price_returns[ticker_input].values})
        source_description = f"Downloaded return history for `{ticker_input}` from yfinance."
    elif analysis_mode == "Portfolio holdings":
        if holdings_file is not None:
            holdings_map = load_holdings_file(holdings_file)
        else:
            holdings_map = parse_holdings_text(holdings_text)

        price_history = fetch_price_history(
            list(holdings_map.keys()),
            str(price_download_start.date()),
            str(price_download_end.date()),
            frequency,
        )
        returns_df = build_portfolio_returns_from_prices(price_history, holdings_map, portfolio_name=portfolio_name)
        source_description = f"Built `{portfolio_name}` from {len(holdings_map)} holdings using yfinance price history."
    else:
        if target_file is None:
            st.warning("Upload a target returns file to begin analysis.")
            st.stop()
        returns_df = load_returns_file(target_file)
        source_description = "Loaded target returns from an uploaded file."

    ff_model = "5 Factor" if any(factor in selected_ff_factors for factor in ["RMW", "CMA"]) else "3 Factor"
    include_momentum_factor = "Mom" in selected_ff_factors
    ff_data = fetch_fama_french_factors(
        ff_model,
        frequency,
        start_date=str(start_date),
        end_date=str(end_date),
        include_momentum=include_momentum_factor,
    )
    ff_columns = ["Date", "RF", *[factor for factor in selected_ff_factors if factor in ff_data.columns]]
    factors_df = ff_data[ff_columns].copy()

    aqr_selected = {region: factors for region, factors in selected_aqr_factors_by_region.items() if factors}
    if aqr_selected:
        aqr_df = fetch_aqr_equity_factors(
            aqr_selected,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        factors_df = merge_factor_frame(factors_df, aqr_df)

    if selected_fixed_income_factors:
        fixed_income_df = compute_fixed_income_factors(
            factors_df[["Date", "RF"]],
            frequency,
            str(price_download_start.date()),
            str(price_download_end.date()),
            selected_fixed_income_factors,
        )
        factors_df = merge_factor_frame(factors_df, fixed_income_df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

returns_df = normalize_returns_frame(returns_df)
factors_df = normalize_returns_frame(factors_df)

returns_df = returns_df[returns_df["Date"].between(selected_start, selected_end)].copy()
factors_df = factors_df[factors_df["Date"].between(selected_start, selected_end)].copy()

target_first_available = returns_df["Date"].min() if not returns_df.empty else pd.NaT
target_last_available = returns_df["Date"].max() if not returns_df.empty else pd.NaT
factor_first_available = factors_df["Date"].min() if not factors_df.empty else pd.NaT
factor_last_available = factors_df["Date"].max() if not factors_df.empty else pd.NaT

target_candidates = [column for column in returns_df.columns if column != "Date"]
factor_candidates = [column for column in factors_df.columns if column != "Date"]

if not target_candidates or not factor_candidates:
    st.error("The selected data source did not produce usable return columns.")
    st.stop()

target_column = (
    target_column_hint
    if target_column_hint and target_column_hint in target_candidates
    else target_candidates[0]
)

factor_columns = [column for column in selected_factor_names if column in factor_candidates and column != "RF"]
missing_factors = [column for column in selected_factor_names if column not in factor_columns]
if missing_factors:
    st.warning("Some selected factors were unavailable for the matched data: " + ", ".join(missing_factors))

risk_free_column = "RF"
if risk_free_column not in factor_candidates:
    st.error("The selected factor data did not provide the required RF column.")
    st.stop()

if not factor_columns:
    st.warning("Choose at least one factor to run the regression.")
    st.stop()

frame = pd.DataFrame()
try:
    frame = prepare_analysis_frame(
        returns_df=returns_df,
        factors_df=factors_df,
        target_column=target_column,
        factor_columns=factor_columns,
        risk_free_column=risk_free_column,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

if len(frame) <= len(factor_columns) + 2:
    st.error("Not enough overlapping observations to run a stable regression.")
    st.stop()

actual_start = frame["Date"].min()
actual_end = frame["Date"].max()
if frequency == "Monthly":
    expected_first_return = selected_start + pd.offsets.MonthEnd(0)
    expected_last_return = selected_end + pd.offsets.MonthEnd(0)
    if expected_last_return > selected_end:
        expected_last_return -= pd.offsets.MonthEnd(1)
elif frequency == "Weekly":
    expected_first_return = selected_start.to_period("W-FRI").end_time.normalize()
    expected_last_return = selected_end.to_period("W-FRI").end_time.normalize()
    if expected_last_return > selected_end:
        expected_last_return -= pd.DateOffset(days=7)
else:
    expected_first_return = selected_start
    expected_last_return = selected_end

date_limit_notes = []
if actual_start > expected_first_return:
    start_limiter = (
        "factor data"
        if pd.notna(factor_first_available)
        and (pd.isna(target_first_available) or factor_first_available >= target_first_available)
        else "target return data"
    )
    date_limit_notes.append(f"The first matched return is limited by {start_limiter}.")
if actual_end < expected_last_return:
    end_limiter = (
        "factor data"
        if pd.notna(factor_last_available)
        and (pd.isna(target_last_available) or factor_last_available <= target_last_available)
        else "target return data"
    )
    date_limit_notes.append(f"The last matched return is limited by {end_limiter}.")

frequency_note = f"{frequency} returns are labeled by their period-ending date."
period_status = (
    f"**Requested period:** {selected_start:%Y-%m-%d} to {selected_end:%Y-%m-%d}  \n"
    f"**Actual matched returns:** {actual_start:%Y-%m-%d} to {actual_end:%Y-%m-%d}  \n"
    f"{frequency_note}"
)
if date_limit_notes:
    period_status += "  \n" + " ".join(date_limit_notes)
    period_status_sidebar.warning(period_status)
else:
    period_status_sidebar.info(period_status)

result = run_factor_regression(frame, factor_columns, periods_per_year)
rolling_betas = compute_rolling_betas(frame, factor_columns, rolling_window)
rolling_diag = compute_rolling_regression_diagnostics(frame, factor_columns, rolling_window, periods_per_year)
style_table = compute_style_consistency(rolling_betas, factor_columns)
contribution_table = compute_contribution_table(frame, result.summary_table, factor_columns)
significance_table = build_factor_significance_table(result.summary_table)
glossary_table = build_regression_glossary()
factor_return_summary = compute_factor_return_summary(frame, factor_columns, target_column, result.annualized_alpha, result.r_squared, periods_per_year)
risk_return_attribution = compute_risk_return_attribution(
    frame,
    result.summary_table,
    factor_columns,
    target_column,
    periods_per_year,
    risk_free_column,
)
cumulative_attribution = compute_cumulative_attribution(frame, result.summary_table, factor_columns)
annual_risk_free_rate = annualize_return(frame[risk_free_column], periods_per_year)
risk_free_sidebar.info(
    f"**Risk-free rate:** Ken French {risk_free_column} at {frequency.lower()} frequency. "
    f"Its annualized rate over the matched sample is **{annual_risk_free_rate:.2%}**. "
    "The regression subtracts each period's RF observation rather than using one constant rate."
)
bt_stats_table = compute_bt_risk_return_stats(
    frame[target_column],
    target_column,
    frame["Date"],
    frame[risk_free_column],
    periods_per_year,
)
if not bt_stats_table.empty:
    bt_stats_table["Display"] = [format_bt_stat_value(metric, value) for metric, value in bt_stats_table[["Metric", "Value"]].itertuples(index=False, name=None)]

periodic_fitted_total = result.fitted_series.copy()
periodic_fitted_total = periodic_fitted_total + frame[risk_free_column].values
rolling_return_windows = [window for window in [12, 36] if len(frame) >= window]
rolling_return_chart_data = pd.DataFrame({"Date": frame["Date"]})
for window in rolling_return_windows:
    rolling_return_chart_data[f"Actual {window}-Period"] = compute_rolling_window_returns(frame[target_column], window)
    rolling_return_chart_data[f"Fitted {window}-Period"] = compute_rolling_window_returns(periodic_fitted_total, window)

regression_overview = pd.DataFrame(
    {
        "Metric": [
            "Target",
            "Time Period",
            "Regression Basis",
            "Observations",
            "R-Squared",
            "Adjusted R-Squared",
            "F-Statistic",
            "Model p-value",
            "Durbin-Watson",
            "Breusch-Pagan p-value",
            "Annualized Alpha",
        ],
        "Value": [
            target_column,
            f"{frame['Date'].min():%Y-%m-%d} to {frame['Date'].max():%Y-%m-%d}",
            f"{frequency} returns",
            f"{len(frame):,}",
            format_value(result.r_squared, "percent"),
            format_value(result.adjusted_r_squared, "percent"),
            format_value(result.f_statistic),
            format_value(result.f_pvalue, "pvalue"),
            format_value(result.durbin_watson),
            format_value(result.breusch_pagan_pvalue, "pvalue"),
            format_value(result.annualized_alpha, "percent"),
        ],
    }
)

st.caption(source_description)
st.caption(
    "Auto-downloaded selected factor data: "
    f"{', '.join(factor_columns)}. RF comes from the Ken French library at {frequency.lower()} frequency."
)
factor_source_description = f"Auto-downloaded: {', '.join(factor_columns)}; RF from Ken French {frequency.lower()} data."

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Observations", f"{len(frame)}", help="Number of overlapping return periods used in the regression.")
metric_col2.metric("R-squared", f"{result.r_squared:.2%}", help="Percentage of variation in excess returns explained by the selected factors.")
metric_col3.metric("Annualized Alpha", f"{result.annualized_alpha:.2%}", help="Annualized return not explained by the selected factor exposures.")
metric_col4.metric("Annualized Return", f"{annualize_return(frame[target_column], periods_per_year):.2%}", help="Compounded annualized return of the fund or portfolio.")

st.caption("Hover over the **?** beside section titles and metric labels, table column headers, or any chart point for an explanation.")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Summary", "Attribution", "Rolling Analysis", "Due Diligence", "Risk/Return Sheet", "Data Preview", "Glossary"]
)

with tab1:
    st.subheader("Factor Regression Overview", help="Core model fit, diagnostics, sample period, and alpha statistics.")
    show_table(regression_overview)


    st.subheader("Factor Analysis Results", help="Estimated factor exposures and each factor's contribution to average excess return.")
    top_left, top_right = st.columns([1.2, 1])
    with top_left:
        coefficient_display = format_dataframe(
            result.summary_table.reset_index().rename(columns={"index": "Factor"}),
            pvalue_cols=["P-Value"],
        )
        show_table(coefficient_display)
    with top_right:
        contribution_display = format_dataframe(
            contribution_table,
            percent_cols=["Average Factor Return", "Estimated Contribution"],
        )
        show_table(contribution_display)

    st.subheader("Factor Returns", help="Annualized target results and factor premiums over the matched analysis period.")
    factor_return_percent_cols = [column for column in factor_return_summary.columns if "Premium" in column or column in ["Annualized Alpha", "R-Squared"]]
    factor_return_display = format_dataframe(
        factor_return_summary,
        percent_cols=factor_return_percent_cols,
        date_cols=["First Return Date", "Last Return Date"],
    )
    show_table(factor_return_display)

    cumulative_chart = go.Figure()
    cumulative_chart.add_trace(
        go.Bar(x=frame["Date"], y=compute_cumulative_return_series(frame[target_column]), name=target_column)
    )
    cumulative_chart.update_layout(title=f"{target_column} Cumulative Return", yaxis_tickformat=".1%")
    st.plotly_chart(polish_chart(cumulative_chart), use_container_width=True)

with tab2:
    st.subheader("Risk and Return Attribution", help="Separates total return and risk into market, factor, alpha, and unexplained components.")
    attribution_percent_cols = [column for column in risk_return_attribution.columns if column not in ["Target", "First Return Date", "Last Return Date"]]
    risk_return_display = format_dataframe(
        risk_return_attribution,
        percent_cols=attribution_percent_cols,
        date_cols=["First Return Date", "Last Return Date"],
    )
    show_table(risk_return_display)

    st.subheader(
        "Cumulative Excess-Return Attribution",
        help="Shows the exact additive decomposition of realized excess return into fitted factors, alpha, and residuals.",
    )
    st.caption(
        "**Actual Excess** is the fund or portfolio return minus Ken French RF. "
        "**Fitted Excess** is alpha plus the selected factor contributions. "
        "**Residual** is Actual Excess minus Fitted Excess. These lines use arithmetic cumulative sums "
        "so the regression components reconcile exactly; the Summary tab contains the compounded total-return view."
    )
    cumulative_plot_columns = ["Actual Excess", "Fitted Excess", *factor_columns, "Alpha", "Residual"]
    cumulative_long = cumulative_attribution[["Date", *cumulative_plot_columns]].melt(id_vars="Date", var_name="Series", value_name="Return")
    cumulative_plot = px.line(cumulative_long, x="Date", y="Return", color="Series", title="Cumulative Arithmetic Excess-Return Decomposition")
    cumulative_plot.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(polish_chart(cumulative_plot), use_container_width=True)

    st.subheader("Actual vs Fitted Excess Returns", help="Compares realized return above the risk-free rate with the return predicted by the factor model each period.")
    fitted_chart = go.Figure()
    fitted_chart.add_trace(go.Scatter(x=frame["Date"], y=frame["TargetExcess"], name="Actual Excess Return", mode="lines"))
    fitted_chart.add_trace(go.Scatter(x=result.fitted_series.index, y=result.fitted_series, name="Fitted Excess Return", mode="lines"))
    fitted_chart.update_layout(title="Actual vs Fitted Excess Returns", yaxis_tickformat=".1%")
    st.plotly_chart(polish_chart(fitted_chart), use_container_width=True)

with tab3:
    st.subheader("Rolling Factor Sensitivities", help="Tracks how estimated factor loadings change across moving windows; large shifts may indicate style drift.")
    if rolling_betas.empty:
        st.info("Add more observations or reduce the rolling window to view rolling outputs.")
    else:
        rolling_long = rolling_betas.melt(id_vars="Date", var_name="Factor", value_name="Beta")
        rolling_chart = px.line(
            rolling_long,
            x="Date",
            y="Beta",
            color="Factor",
            title=f"{rolling_window}-Period Rolling Betas",
        )
        st.plotly_chart(polish_chart(rolling_chart, ".3f"), use_container_width=True)
        rolling_beta_analysis = build_rolling_beta_analysis(rolling_betas, factor_columns, rolling_window)
        if rolling_beta_analysis:
            st.info(rolling_beta_analysis)

        safe_target_name = "".join(character if character.isalnum() else "_" for character in target_column).strip("_")
        rolling_workbook = build_rolling_sensitivity_workbook(
            frame=frame,
            rolling_betas=rolling_betas,
            rolling_diag=rolling_diag,
            factor_columns=factor_columns,
            target_column=target_column,
            rolling_window=rolling_window,
            frequency=frequency,
        )
        st.download_button(
            "Download rolling factor sensitivity workbook",
            data=rolling_workbook,
            file_name=f"rolling_factor_sensitivities_{safe_target_name or 'target'}_{rolling_window}_period.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Downloads the plotted rolling betas, aligned regression inputs, and an explanation of the math and intuition behind the chart.",
        )

    st.subheader("Rolling Regression Diagnostics", help="Tracks annualized alpha and explanatory power across moving regression windows.")
    if rolling_diag.empty:
        st.info("Add more observations or reduce the rolling window to view rolling diagnostics.")
    else:
        rolling_diag_chart = go.Figure()
        rolling_diag_chart.add_trace(go.Scatter(x=rolling_diag["Date"], y=rolling_diag["Annualized Alpha"], name="Annualized Alpha", mode="lines"))
        rolling_diag_chart.add_trace(go.Scatter(x=rolling_diag["Date"], y=rolling_diag["R-Squared"], name="R-Squared", mode="lines", yaxis="y2"))
        rolling_diag_chart.update_layout(
            title=f"{rolling_window}-Period Rolling Alpha and Fit",
            yaxis=dict(title="Annualized Alpha", tickformat=".1%"),
            yaxis2=dict(title="R-Squared", overlaying="y", side="right", tickformat=".0%"),
        )
        st.plotly_chart(polish_chart(rolling_diag_chart), use_container_width=True)

    st.subheader("Rolling Window Returns", help="Compounded actual and model-fitted returns over trailing 12- and 36-period windows.")
    if rolling_return_windows:
        rolling_return_long = rolling_return_chart_data.melt(id_vars="Date", var_name="Series", value_name="Rolling Return").dropna()
        rolling_return_plot = px.line(rolling_return_long, x="Date", y="Rolling Return", color="Series", title="Rolling Compounded Returns")
        rolling_return_plot.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(polish_chart(rolling_return_plot), use_container_width=True)
    else:
        st.info("Not enough observations yet to compute 12- or 36-period rolling compounded returns.")

with tab4:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Style Consistency", help="Summarizes the stability of each rolling factor exposure; higher consistency and lower volatility indicate a steadier style.")
        if style_table.empty:
            st.info("Style consistency scores will appear once rolling exposures are available.")
        else:
            st.caption(
                "Overall style is classified as relatively stable only when the average consistency score is "
                "at least 0.80 and every individual factor score is at least 0.70."
            )
            style_display = format_dataframe(style_table)
            show_table(style_display)
    with right:
        st.subheader("Residual Diagnostics", help="Residuals are actual excess returns minus fitted excess returns; random dispersion around zero is generally desirable.")
        residual_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(result.residual_series.index),
                "Residual": pd.to_numeric(result.residual_series.to_numpy(), errors="coerce"),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if residual_df.empty:
            st.info("Residuals are unavailable because the fitted regression did not produce finite values.")
        else:
            residual_chart = go.Figure(
                go.Scatter(
                    x=residual_df["Date"],
                    y=residual_df["Residual"],
                    name="Residual",
                    mode="markers",
                    marker={"size": 7, "opacity": 0.75},
                )
            )
            residual_chart.add_hline(y=0, line_dash="dash", annotation_text="Zero residual")
            residual_chart.update_layout(title="Residuals Through Time", yaxis_tickformat=".1%")
            st.plotly_chart(polish_chart(residual_chart), use_container_width=True)

    st.subheader("Factor Significance", help="Evaluates whether each estimated exposure is statistically distinguishable from zero.")
    significance_display = format_dataframe(significance_table, pvalue_cols=["P-Value"])
    show_table(significance_display)

    with st.expander("How to interpret T-Stat, P-Value, CI Lower, and CI Upper", expanded=False):
        show_table(glossary_table)
        st.markdown(
            "A factor is usually treated as statistically significant when its `P-Value` is below `0.05` "
            "and its confidence interval does not cross zero. That means the estimated exposure is less likely "
            "to be noise in the sample you selected."
        )

    st.subheader(
        "Manager Due Diligence Notes",
        help=(
            "Plain-language signals generated from alpha, model fit, factor significance, and style stability. "
            + STYLE_CONSISTENCY_HELP
        ),
    )
    notes = []
    alpha_row = significance_table.loc[significance_table["Factor"] == "Alpha"].iloc[0]
    if alpha_row["Significant"] == "Yes":
        notes.append("Alpha is statistically significant in the selected sample, which suggests return not fully explained by the chosen factor set.")
    else:
        notes.append("Alpha is not statistically distinguishable from zero in the selected sample.")

    if result.r_squared >= 0.75:
        notes.append("The factor model explains a high share of return variation.")
    elif result.r_squared >= 0.5:
        notes.append("The factor model has moderate explanatory power.")
    else:
        notes.append("A large portion of returns remains unexplained by the selected factor set.")

    significant_non_alpha = significance_table[(significance_table["Factor"] != "Alpha") & (significance_table["Significant"] == "Yes")]
    if significant_non_alpha.empty:
        notes.append("None of the selected factor exposures are statistically significant at the 5% level.")
    else:
        factor_list = ", ".join(significant_non_alpha["Factor"].tolist())
        notes.append(f"Statistically significant factor exposures in this sample: {factor_list}.")

    if not style_table.empty:
        average_style_score = style_table["Style Consistency Score"].mean()
        weakest_style_score = style_table["Style Consistency Score"].min()
        if average_style_score >= 0.80 and weakest_style_score >= 0.70:
            style_consistency_note = (
                f"Rolling exposures suggest a relatively stable style profile "
                f"(average score {average_style_score:.2f}; weakest factor {weakest_style_score:.2f})."
            )
        else:
            style_consistency_note = (
                f"Rolling exposures show meaningful style drift "
                f"(average score {average_style_score:.2f}; weakest factor {weakest_style_score:.2f})."
            )
        notes.append(style_consistency_note)

    for note in notes:
        st.write(f"- {note}")

with tab5:
    st.subheader("bt Risk/Return Sheet", help="Performance and risk statistics calculated from the selected fund or portfolio return stream.")
    st.caption(
        f"Sharpe and Sortino use each {frequency.lower()} Ken French RF observation directly. "
        f"The {annual_risk_free_rate:.2%} annualized RF shown in the sidebar is descriptive; "
        "other performance and drawdown statistics come from bt/ffn."
    )
    if bt_stats_table.empty:
        st.info("Risk/return statistics are unavailable for the current selection.")
    else:
        display_bt_stats = bt_stats_table[["Metric", "Display"]].rename(columns={"Display": "Value"})
        show_table(display_bt_stats)

        cumulative_returns = compute_cumulative_return_series(frame[target_column]) + 1
        cumulative_chart = go.Figure()
        cumulative_chart.add_trace(go.Scatter(x=frame["Date"], y=cumulative_returns, mode="lines", name=target_column))
        cumulative_chart.update_layout(title="Cumulative Growth of $1", yaxis_tickformat=".2f")
        st.plotly_chart(polish_chart(cumulative_chart, ".3f"), use_container_width=True)

with tab6:
    preview_left, preview_right = st.columns(2)
    preview_left.subheader("Merged Analysis Data", help="The date-aligned target, factor, risk-free, and excess-return observations used by the regression.")
    with preview_left:
        show_table(frame, hide_index=False)
    preview_right.subheader("Source Data Checks", help="Observation counts and selected sources used to confirm that the inputs overlap correctly.")
    quality_checks = pd.DataFrame(
        {
            "Check": [
                "Target observations",
                "Factor observations",
                "Overlapping observations",
                "Selected factors",
                "Target source",
                "Factor source",
            ],
            "Value": [
                len(returns_df),
                len(factors_df),
                len(frame),
                ", ".join(factor_columns),
                analysis_mode,
                factor_source_description,
            ],
        }
    )
    with preview_right:
        show_table(quality_checks)


with tab7:
    st.subheader("Factor Glossary", help="Definitions and source links for the factor data used by the app.")
    st.caption(
        "Ken French and AQR data are pulled from their published data files when selected. "
        "Fixed income factors are calculated from yfinance price histories for the stated Vanguard index funds."
    )
    glossary_display = build_factor_glossary_table()
    st.dataframe(
        glossary_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source URL": st.column_config.LinkColumn("Source URL", display_text="Open source"),
            "Meaning": st.column_config.Column("Meaning", width="large"),
            "Calculation Notes": st.column_config.Column("Calculation Notes", width="large"),
        },
    )

    st.subheader(
        "Cumulative Excess-Return Attribution Math",
        help="Explains how the Attribution tab decomposes realized excess return into factor, alpha, and residual components.",
    )
    st.markdown(
        """
The attribution chart starts with the same regression used everywhere else in the app:

```text
TargetExcess_t = TargetReturn_t - RF_t

TargetExcess_t = Alpha
               + Beta_1 * Factor_1,t
               + Beta_2 * Factor_2,t
               + ...
               + Beta_k * Factor_k,t
               + Residual_t
```

For each period, the app calculates each factor's explained return contribution:

```text
FactorContribution_i,t = Beta_i * FactorReturn_i,t
AlphaContribution_t    = Alpha
FittedExcess_t         = Alpha + sum(FactorContribution_i,t)
Residual_t             = ActualExcess_t - FittedExcess_t
```

Then the plotted cumulative lines are arithmetic running totals:

```text
CumulativeFactor_i,T = sum from t=1 to T of FactorContribution_i,t
CumulativeAlpha_T    = sum from t=1 to T of Alpha
CumulativeFitted_T   = sum from t=1 to T of FittedExcess_t
CumulativeActual_T   = sum from t=1 to T of TargetExcess_t
CumulativeResidual_T = CumulativeActual_T - CumulativeFitted_T
```

The chart uses arithmetic cumulative sums rather than compounded returns because the regression equation is additive. That makes the pieces reconcile exactly:

```text
CumulativeActualExcess = CumulativeFittedExcess + CumulativeResidual

CumulativeFittedExcess = CumulativeAlpha + sum(CumulativeFactorContributions)
```

So if `Mkt-RF` has a beta of `1.10` and the market excess return in a month is `2.00%`, that month's market contribution is:

```text
1.10 * 2.00% = 2.20%
```

If monthly alpha is `0.05%`, that same `0.05%` is added as the alpha contribution for every month in the fitted excess-return line.
"""
    )

    st.subheader(
        "Style Stability Math",
        help="Explains how the app classifies rolling factor exposure as stable, watch, or style drift.",
    )
    st.markdown(
        """
The style profile is based on the rolling beta series for each selected factor. For every rolling window, the app refits the same regression using only the trailing window of observations:

```text
TargetExcess_t = Alpha_window
               + Beta_1,window * Factor_1,t
               + ...
               + Beta_k,window * Factor_k,t
               + Residual_t
```

That creates a time series of rolling betas for each factor:

```text
Beta_i,1, Beta_i,2, Beta_i,3, ..., Beta_i,T
```

For each factor, the app measures how much that rolling exposure moves around:

```text
BetaVolatility_i = standard deviation of the rolling beta series for factor i
```

Then it converts beta volatility into a style consistency score:

```text
StyleConsistencyScore_i = 1 / (1 + BetaVolatility_i)
```

This makes the score easier to read:

```text
Lower beta volatility -> score closer to 1.00 -> more stable exposure
Higher beta volatility -> score closer to 0.00 -> less stable exposure
```

The per-factor classifications are:

```text
Stable      >= 0.80
Watch       >= 0.70 and < 0.80
Style drift < 0.70
```

The manager due diligence note summarizes the full factor set. It calls the overall style profile relatively stable only when both conditions are true:

```text
Average style consistency score >= 0.80
Weakest individual factor score >= 0.70
```

So a portfolio can have a good average score but still get flagged if one selected factor has a very unstable rolling beta. That is intentional: one jumpy exposure can matter even when the broader factor profile looks calm.
"""
    )
