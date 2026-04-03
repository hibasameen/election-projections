# UK Election Projection Tool

Interactive constituency-level seat projection model for UK general elections, built from the July 2024 baseline.

**[Live demo →](https://hibasameen.github.io/election-projections/)**

## Features

- **Swing models** — Uniform National Swing (UNS) and Proportional swing, applied at constituency level with renormalisation
- **Smoother selection** — choose between Kalman, LOWESS, or Rolling 21-day smoothers to pre-load vote share sliders; separate regional sliders for Scotland (SNP) and Wales (Plaid Cymru)
- **Tactical voting** — three configurable models (progressive consolidation, right consolidation, anti-Reform squeeze) layered on top of the swing projection
- **Scenarios** — one-click presets for Green surge, Reform decline, and Lab/Con recovery
- **Interactive map** — all 650 constituency boundaries embedded (simplified ONS Dec 2024 GeoJSON), with projected/actual/change-only views
- **Polling trends** — 409 polls (Jul 2024 – Apr 2026) from 18+ pollsters with LOWESS, Rolling 21-day, and Kalman smoothers; toggle weighted/unweighted
- **Detailed tables** — summary, marginals, seat changes, tactical impact, regional breakdown, and full seat list with search/filter

## Repository structure

```
├── index.html                     # Main projection tool (self-contained, ~5 MB)
├── notebook/
│   └── poll_data_lowess_2026.ipynb  # Poll extraction & smoothing notebook
├── data/
│   ├── uk_polling_2024_2026_national_dates_for_chart.csv  # 409 polls, raw data
│   └── Pollster_Ratings.csv         # Quality ratings for weighting
├── maps/
│   ├── uk_polls_lowess.html         # Interactive LOWESS polling chart
│   ├── uk_2019_2024_combined_map.html
│   ├── uk_2019_2024_layers.html
│   └── uk_swing_map_plain_fixed_small.html
└── README.md
```

## Data sources

| Source | Description |
|--------|-------------|
| House of Commons Library | 2024 general election results (650 constituencies) |
| Wikipedia | "Opinion polling for the next UK general election" aggregate |
| ONS Open Geography Portal | December 2024 constituency boundaries |

## Architecture

The projection tool is a **single self-contained HTML file** (~5 MB). All constituency data, boundary geometries, and polling data are embedded inline. The only external dependencies are CDN-hosted JavaScript libraries:

- [Leaflet.js](https://leafletjs.com/) — map rendering
- [Plotly.js](https://plotly.com/javascript/) — charts
- [CARTO](https://carto.com/) — base map tiles

No server, no API calls, no uploads required. Open `index.html` in any modern browser.

## Methodology

See the **About** tab within the tool for full documentation of swing models, polling smoothers, tactical voting mechanics, scenarios, and caveats.

## Author

Hiba Sameen · © 2026

## License

All rights reserved.
