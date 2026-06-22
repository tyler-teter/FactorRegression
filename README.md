# Factor Regression Lab

Streamlit app for fund and portfolio analysis using automated market data, Fama-French factor downloads, multi-factor regression, rolling exposures, and manager due diligence diagnostics.

## Features

- Analyze a single fund or ETF ticker using `yfinance`
- Build portfolio returns automatically from holdings and weights
- Auto-download Fama-French 3-factor or 5-factor datasets
- Keep uploaded return files as a fallback workflow
- Estimate factor betas, alpha, t-stats, p-values, and confidence intervals
- Compare actual versus fitted returns
- Visualize rolling factor sensitivities and style consistency

## Expected input formats

Uploaded returns files should include:

- `Date`
- One or more numeric return columns

Holdings files should include:

- `Ticker`
- `Weight`

Use decimal returns such as `0.01` for 1%.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- `yfinance` is a strong default for fund, ETF, and proxy return histories.
- `bt` can still be useful later for strategy backtests and rebalancing logic, but it is not required for this regression workflow.
