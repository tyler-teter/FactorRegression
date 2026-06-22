from __future__ import annotations

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
    compute_risk_return_attribution,
    compute_rolling_betas,
    compute_rolling_regression_diagnostics,
    compute_rolling_window_returns,
    compute_style_consistency,
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
    "Start Date": "First overlapping observation used in the analysis.",
    "End Date": "Last overlapping observation used in the analysis.",
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


def polish_chart(fig: go.Figure, hover_format: str = ".2%") -> go.Figure:
    """Apply consistent hover labels to each chart trace."""
    fig.update_traces(
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{fullData.name}}: %{{y:{hover_format}}}<extra></extra>"
    )
    fig.update_layout(hovermode="x unified")
    return fig


st.title("Factor Regression Lab")
st.caption(
    "Built by Tyler Teter, CFP®, CFA | "
    "[LinkedIn](https://www.linkedin.com/in/tylerteter/)"
)
st.caption(
    "Analyze a fund or portfolio with yfinance price histories, auto-downloaded "
    "Fama-French factors, rolling exposure analysis, and bt risk/return diagnostics."
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
    factor_source = "Auto-download Fama-French"
    factor_model_options = ["3 Factor"] if frequency == "Weekly" else ["3 Factor", "5 Factor"]
    default_factor_index = 0 if len(factor_model_options) == 1 else 1
    factor_model = st.selectbox("Fama-French model", factor_model_options, index=default_factor_index)
    rolling_window = st.slider("Rolling window length", min_value=12, max_value=60, value=24, step=6)
    start_date = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp.today().normalize())
    risk_free_sidebar = st.empty()
    st.link_button(
        "Ken French Data Library",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        use_container_width=True,
    )


periods_per_year = {"Monthly": 12, "Weekly": 52, "Daily": 252}[frequency]

st.info(
    "You can analyze a single fund ticker or a weighted portfolio built from holdings. "
    "Factor data is downloaded directly from the Ken French library."
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

if factor_source == "Upload factor file":
    factor_file = st.file_uploader("Upload factor returns", type=["csv", "xlsx", "xls"])
else:
    factor_file = None

try:
    if analysis_mode == "Single fund ticker":
        if not ticker_input:
            st.warning("Enter a ticker to download returns.")
            st.stop()
        price_history = fetch_price_history([ticker_input], str(start_date), str(end_date), frequency)
        price_returns = price_history.pct_change().dropna(how="all")
        returns_df = pd.DataFrame({"Date": price_returns.index, ticker_input: price_returns[ticker_input].values})
        source_description = f"Downloaded return history for `{ticker_input}` from yfinance."
    elif analysis_mode == "Portfolio holdings":
        if holdings_file is not None:
            holdings_map = load_holdings_file(holdings_file)
        else:
            holdings_map = parse_holdings_text(holdings_text)

        price_history = fetch_price_history(list(holdings_map.keys()), str(start_date), str(end_date), frequency)
        returns_df = build_portfolio_returns_from_prices(price_history, holdings_map, portfolio_name=portfolio_name)
        source_description = f"Built `{portfolio_name}` from {len(holdings_map)} holdings using yfinance price history."
    else:
        if target_file is None:
            st.warning("Upload a target returns file to begin analysis.")
            st.stop()
        returns_df = load_returns_file(target_file)
        source_description = "Loaded target returns from an uploaded file."

    if factor_source == "Auto-download Fama-French":
        factors_df = fetch_fama_french_factors(
            factor_model,
            frequency,
            start_date=str(start_date),
            end_date=str(end_date),
        )
    else:
        if factor_file is None:
            st.warning("Upload a factor returns file to begin analysis.")
            st.stop()
        factors_df = load_returns_file(factor_file)
except Exception as exc:
    st.error(str(exc))
    st.stop()

returns_df = normalize_returns_frame(returns_df)
factors_df = normalize_returns_frame(factors_df)

selected_start = pd.Timestamp(start_date)
selected_end = pd.Timestamp(end_date)
returns_df = returns_df[returns_df["Date"].between(selected_start, selected_end)].copy()
factors_df = factors_df[factors_df["Date"].between(selected_start, selected_end)].copy()

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

if factor_model == "3 Factor":
    preferred_factors = [column for column in ["Mkt-RF", "SMB", "HML"] if column in factor_candidates]
else:
    preferred_factors = [column for column in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if column in factor_candidates]

factor_columns = st.multiselect(
    "Select factor set",
    options=[column for column in factor_candidates if column != "RF"],
    default=preferred_factors or factor_candidates[: min(4, len(factor_candidates))],
)
risk_free_column = "RF"
if risk_free_column not in factor_candidates:
    st.error("The selected Fama-French dataset did not provide the required RF column.")
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
if factor_source == "Auto-download Fama-French":
    st.caption(f"Auto-downloaded `{factor_model}` data from the Ken French library at {frequency.lower()} frequency.")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Observations", f"{len(frame)}", help="Number of overlapping return periods used in the regression.")
metric_col2.metric("R-squared", f"{result.r_squared:.2%}", help="Percentage of variation in excess returns explained by the selected factors.")
metric_col3.metric("Annualized Alpha", f"{result.annualized_alpha:.2%}", help="Annualized return not explained by the selected factor exposures.")
metric_col4.metric("Annualized Return", f"{annualize_return(frame[target_column], periods_per_year):.2%}", help="Compounded annualized return of the fund or portfolio.")

st.caption("Hover over the **?** beside section titles and metric labels, table column headers, or any chart point for an explanation.")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Summary", "Attribution", "Rolling Analysis", "Due Diligence", "Risk/Return Sheet", "Data Preview"]
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
        date_cols=["Start Date", "End Date"],
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
    attribution_percent_cols = [column for column in risk_return_attribution.columns if column not in ["Target", "Start Date", "End Date"]]
    risk_return_display = format_dataframe(
        risk_return_attribution,
        percent_cols=attribution_percent_cols,
        date_cols=["Start Date", "End Date"],
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

    st.subheader("Manager Due Diligence Notes", help="Plain-language signals generated from alpha, model fit, factor significance, and style stability.")
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
            notes.append(
                f"Rolling exposures suggest a relatively stable style profile "
                f"(average score {average_style_score:.2f}; weakest factor {weakest_style_score:.2f})."
            )
        else:
            notes.append(
                f"Rolling exposures show meaningful style drift "
                f"(average score {average_style_score:.2f}; weakest factor {weakest_style_score:.2f})."
            )

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
                factor_source,
            ],
        }
    )
    with preview_right:
        show_table(quality_checks)