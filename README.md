# UK Election Projection Tool

Interactive constituency-level seat projection model for UK general elections, built from the July 2024 baseline.

**[Live demo →](https://hibasameen.github.io/election-projections/)**

## Features

- **Swing models** — Uniform National Swing (UNS) and Proportional swing, applied at constituency level with renormalisation
- **Vote share sliders** — pre-loaded from a quality-weighted Kalman smoother; separate regional sliders for Scotland (SNP) and Wales (Plaid Cymru)
- **Tactical voting** — three configurable models (progressive consolidation, right consolidation, anti-Reform squeeze) layered on top of the swing projection
- **Scenarios** — one-click presets for Green surge, Reform decline, and Lab/Con recovery
- **Interactive map** — all 650 constituency boundaries embedded (simplified ONS Dec 2024 GeoJSON), with projected/actual/change-only views
- **Polling trends** — 406 polls (Jul 2024 – Mar 2026) from 18+ pollsters with LOWESS, Rolling 21-day, and Kalman smoothers; toggle weighted/unweighted
- **Detailed tables** — summary, marginals, seat changes, tactical impact, regional breakdown, and full seat list with search/filter

## Data sources

| Source | Description |
|--------|-------------|
| House of Commons Library | 2024 general election results (650 constituencies) |
| Wikipedia | "Opinion polling for the next UK general election" aggregate |
| ONS Open Geography Portal | December 2024 constituency boundaries |

## Architecture

The entire tool is a **single self-contained HTML file** (~5 MB). All constituency data, boundary geometries, and polling data are embedded inline. The only external dependencies are CDN-hosted JavaScript libraries:

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
