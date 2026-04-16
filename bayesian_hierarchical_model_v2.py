"""
Bayesian Hierarchical Model v2 for UK Constituency-Level Vote Shares

Improvements over v1:
  1. Cross-validated calibration of variance components (LOO-CV and k-fold)
     to detect overfitting/underfitting with only one election transition.
  2. SNP and Plaid Cymru brought into the hierarchical ALR framework
     (Scotland/Wales treated as regions with party-specific partial pooling).
  3. Empirical polling error covariance estimated from:
     (a) historical final-poll-vs-result misses (2015, 2017, 2019, 2024), and
     (b) time-series residuals from LOWESS/Kalman smoothers.

Model specification (v2):
  Level 1: δ_{c,p} ~ N(γ_{r[c],p} + β_{1,p}·ΔF_c, σ²_p)
  Level 2: γ_{r,p} ~ N(μ_p, τ²_p)
  Priors:  μ_p ~ N(0, 3), τ_p ~ HalfCauchy(1.5), σ_p ~ HalfCauchy(1.5)

  Now p ∈ {Con, RUK, LD, Green, SNP, PC} (all relative to Labour).
  SNP deltas observed only in Scotland, PC only in Wales.
  The model uses partial pooling: SNP/PC share the national hyperprior for τ
  but their γ parameters are only estimated for relevant regions.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. Load data ──────────────────────────────────────────────────────────────
BASE = '/Users/hibasameen/Library/Mobile Documents/com~apple~CloudDocs/Election_modelling/'

results_2024 = pd.read_csv(BASE + 'HoC-GE2024-results-by-constituency.csv')
results_2019 = pd.read_csv(BASE + 'HoC-GE2019-results-by-constituency.csv')
successors = pd.read_csv(BASE + 'constituency_closest_successors_final.csv')
foreign_born = pd.read_csv(BASE + 'change in foreign born.csv')
polling_data = pd.read_csv(BASE + 'uk_polling_2024_2026_national_dates_for_chart.csv')

print(f"2024 results: {len(results_2024)} constituencies")
print(f"2019 results: {len(results_2019)} constituencies")
print(f"Successors mapping: {len(successors)} rows")
print(f"Foreign born data: {len(foreign_born)} rows")
print(f"Polling data: {len(polling_data)} polls")

# ── 2. Merge 2019 and 2024 results via boundary mapping ──────────────────────
merged = successors.merge(
    results_2019[['ONS ID', 'Constituency name', 'Region name', 'Country name',
                   'Valid votes', 'Con', 'Lab', 'LD', 'BRX', 'Green', 'SNP', 'PC']],
    left_on='2019_code', right_on='ONS ID', how='left'
).merge(
    results_2024[['ONS ID', 'Constituency name', 'Region name', 'Country name',
                   'Valid votes', 'Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC']],
    left_on='2024_code', right_on='ONS ID', how='left',
    suffixes=('_19', '_24')
)

merged = merged[merged['Country name_24'].isin(['England', 'Scotland', 'Wales'])].copy()

# Compute vote shares
for p in ['Con', 'Lab', 'LD', 'BRX', 'Green', 'SNP', 'PC']:
    col = f'{p}_19' if f'{p}_19' in merged.columns else p
    if col in merged.columns:
        merged[f'share_{p}_19'] = merged[col] / merged['Valid votes_19'] * 100

if 'share_BRX_19' in merged.columns:
    merged['share_RUK_19'] = merged['share_BRX_19']

for p in ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC']:
    col_24 = f'{p}_24' if f'{p}_24' in merged.columns else p
    if col_24 in merged.columns:
        merged[f'share_{p}_24'] = merged[col_24] / merged['Valid votes_24'] * 100

# ── 3. Compute log-ratio swings ─────────────────────────────────────────────
# ALR transform with Labour as reference
# Now includes SNP and PC as modelled parties

PARTIES_GB = ['Con', 'RUK', 'LD', 'Green']      # present everywhere in GB
PARTIES_SC = ['SNP']                               # Scotland only
PARTIES_WA = ['PC']                                # Wales only
ALL_MODEL_PARTIES = PARTIES_GB + PARTIES_SC + PARTIES_WA
FLOOR = 0.5

for p in ALL_MODEL_PARTIES:
    s19 = np.maximum(merged[f'share_{p}_19'].fillna(0).values, FLOOR)
    lab19 = np.maximum(merged['share_Lab_19'].fillna(0).values, FLOOR)
    s24 = np.maximum(merged[f'share_{p}_24'].fillna(0).values, FLOOR)
    lab24 = np.maximum(merged['share_Lab_24'].fillna(0).values, FLOOR)

    merged[f'eta_{p}_19'] = np.log(s19 / lab19)
    merged[f'eta_{p}_24'] = np.log(s24 / lab24)
    merged[f'delta_{p}'] = merged[f'eta_{p}_24'] - merged[f'eta_{p}_19']

# ── 4. Prepare region indices ─────────────────────────────────────────────────
delta_cols_gb = [f'delta_{p}' for p in PARTIES_GB]
model_data = merged.dropna(subset=delta_cols_gb + ['Region name_24']).copy()

# Filter extreme outliers (> 4 sd) for GB parties
for col in delta_cols_gb:
    m, s = model_data[col].mean(), model_data[col].std()
    model_data = model_data[model_data[col].between(m - 4*s, m + 4*s)]

regions = sorted(model_data['Region name_24'].unique())
region_map = {r: i for i, r in enumerate(regions)}
model_data['region_idx'] = model_data['Region name_24'].map(region_map)

# Identify Scottish and Welsh constituencies in model_data
is_scotland = model_data['Country name_24'] == 'Scotland'
is_wales = model_data['Country name_24'] == 'Wales'

n_regions = len(regions)
n_constituencies = len(model_data)

print(f"\nModel data: {n_constituencies} constituencies, {n_regions} regions")
print(f"  Scotland: {is_scotland.sum()} seats")
print(f"  Wales: {is_wales.sum()} seats")
print(f"Regions: {regions}")

# ── 5. Merge foreign born change data ────────────────────────────────────────
fb_merged = model_data.merge(
    foreign_born[['ONSConstID_y', 'pct_point_change', 'foreign_pc_2021']],
    left_on='2024_code', right_on='ONSConstID_y', how='left'
)

has_fb = fb_merged['pct_point_change'].notna()
print(f"Foreign born data available for {has_fb.sum()} of {len(fb_merged)} constituencies")

fb_change = fb_merged['pct_point_change'].values.copy()
fb_mean = np.nanmean(fb_change)
fb_std = np.nanstd(fb_change)
fb_change_std = np.nan_to_num((fb_change - fb_mean) / fb_std, nan=0.0)
has_fb_mask = has_fb.values.astype(float)

# ── 6. Build the extended hierarchical model ─────────────────────────────────
# Key change: SNP and PC are now modelled parties with their own
# μ, τ, σ, γ, β_fb parameters. SNP deltas are only observed in
# Scottish constituencies; PC deltas only in Welsh constituencies.
# We use a masked likelihood to handle this cleanly.

# Observed data matrices
Y_gb = model_data[delta_cols_gb].values  # (N, 4) — observed everywhere
region_idx = model_data['region_idx'].values

# SNP deltas — only meaningful in Scotland
snp_deltas = model_data['delta_SNP'].values
snp_mask = is_scotland.values.astype(float)
sc_idx = np.where(is_scotland.values)[0]  # integer indices of Scottish seats
snp_deltas_sc = snp_deltas[sc_idx]         # observed SNP deltas (Scotland only)

# PC deltas — only meaningful in Wales
pc_deltas = model_data['delta_PC'].values
pc_mask = is_wales.values.astype(float)
wa_idx = np.where(is_wales.values)[0]      # integer indices of Welsh seats
pc_deltas_wa = pc_deltas[wa_idx]            # observed PC deltas (Wales only)

n_parties_gb = len(PARTIES_GB)
n_parties_all = len(ALL_MODEL_PARTIES)  # 6

print(f"\nObserved swing stats (GB parties):")
for i, p in enumerate(PARTIES_GB):
    print(f"  {p}: mean={Y_gb[:,i].mean():.3f}, sd={Y_gb[:,i].std():.3f}")
print(f"  SNP (Scotland only): mean={snp_deltas[snp_mask>0].mean():.3f}, sd={snp_deltas[snp_mask>0].std():.3f}")
print(f"  PC (Wales only): mean={pc_deltas[pc_mask>0].mean():.3f}, sd={pc_deltas[pc_mask>0].std():.3f}")

print("\nFitting extended Bayesian hierarchical model (v2)...")

with pm.Model() as hierarchical_model_v2:
    # ── Hyperpriors (all 6 parties) ──
    mu = pm.Normal('mu', mu=0, sigma=3, shape=n_parties_all)
    tau = pm.HalfCauchy('tau', beta=1.5, shape=n_parties_all)
    sigma = pm.HalfCauchy('sigma', beta=1.5, shape=n_parties_all)

    # ── Region-level means (all 6 parties × all regions) ──
    # SNP gammas will only be identified in Scotland region(s), and PC in Wales.
    # The partial pooling via the shared hyperprior (μ, τ) still applies:
    # non-observed region gammas will shrink toward μ, which is fine.
    gamma = pm.Normal('gamma', mu=mu, sigma=tau, shape=(n_regions, n_parties_all))

    # ── Foreign-born coefficient (all 6 parties) ──
    beta_fb = pm.Normal('beta_fb', mu=0, sigma=1, shape=n_parties_all)

    # ── Constituency-level expected swing ──
    region_effect = gamma[region_idx]  # (N, 6)
    fb_effect = beta_fb[None, :] * (fb_change_std[:, None] * has_fb_mask[:, None])
    expected_all = region_effect + fb_effect  # (N, 6)

    # ── Likelihood: GB parties (observed for all constituencies) ──
    obs_gb = pm.Normal('obs_gb',
                       mu=expected_all[:, :n_parties_gb],
                       sigma=sigma[:n_parties_gb],
                       observed=Y_gb)

    # ── Likelihood: SNP (observed only in Scotland) ──
    snp_expected = expected_all[sc_idx, 4]  # index 4 = SNP, Scotland only
    snp_sd = sigma[4]
    obs_snp = pm.Normal('obs_snp',
                        mu=snp_expected,
                        sigma=snp_sd,
                        observed=snp_deltas_sc)

    # ── Likelihood: PC (observed only in Wales) ──
    pc_expected = expected_all[wa_idx, 5]  # index 5 = PC, Wales only
    pc_sd = sigma[5]
    obs_pc = pm.Normal('obs_pc',
                       mu=pc_expected,
                       sigma=pc_sd,
                       observed=pc_deltas_wa)

    # ── Sample ──
    trace = pm.sample(2000, tune=1500, chains=2, cores=1,
                      random_seed=42, progressbar=True,
                      target_accept=0.9,
                      idata_kwargs={"log_likelihood": True})

print("\nSampling complete!")
summary = az.summary(trace, var_names=['mu', 'tau', 'sigma', 'beta_fb'])
print(summary)

# ── 7. Cross-validated calibration (Improvement #1) ─────────────────────────
# Use LOO-CV via PSIS to assess predictive calibration.
# Also run 5-fold spatial CV (by region) to check for overfitting.

print("\n=== Cross-validation diagnostics ===")

# 7a. LOO-CV using Pareto Smoothed Importance Sampling
print("\nLOO-CV (PSIS)...")
loo = az.loo(trace, var_name="obs_gb", pointwise=True)
print(loo)

# Check for problematic Pareto k values
k_values = loo.pareto_k.values if hasattr(loo, 'pareto_k') else None
if k_values is not None:
    n_bad = np.sum(k_values > 0.7)
    print(f"\nPareto k diagnostics:")
    print(f"  k > 0.7 (unreliable): {n_bad} / {len(k_values)} ({100*n_bad/len(k_values):.1f}%)")
    print(f"  k > 0.5 (marginal): {np.sum(k_values > 0.5)} / {len(k_values)}")
    print(f"  Mean k: {k_values.mean():.3f}, Max k: {k_values.max():.3f}")

# 7b. Region-based k-fold CV (leave-one-region-out)
print("\nLeave-one-region-out CV...")
region_cv_results = []

for hold_region, hold_idx in region_map.items():
    # Identify train/test split
    train_mask = region_idx != hold_idx
    test_mask = region_idx == hold_idx
    n_test = test_mask.sum()

    if n_test < 5:
        continue

    Y_train = Y_gb[train_mask]
    Y_test = Y_gb[test_mask]
    ridx_train = region_idx[train_mask]
    fb_train = fb_change_std[train_mask]
    fb_mask_train = has_fb_mask[train_mask]

    # Refit on training data (quick: fewer samples)
    with pm.Model() as cv_model:
        mu_cv = pm.Normal('mu', mu=0, sigma=3, shape=n_parties_gb)
        tau_cv = pm.HalfCauchy('tau', beta=1.5, shape=n_parties_gb)
        sigma_cv = pm.HalfCauchy('sigma', beta=1.5, shape=n_parties_gb)
        gamma_cv = pm.Normal('gamma', mu=mu_cv, sigma=tau_cv,
                             shape=(n_regions, n_parties_gb))
        beta_fb_cv = pm.Normal('beta_fb', mu=0, sigma=1, shape=n_parties_gb)

        re_cv = gamma_cv[ridx_train]
        fb_cv = beta_fb_cv[None, :] * (fb_train[:, None] * fb_mask_train[:, None])
        exp_cv = re_cv + fb_cv

        obs_cv = pm.Normal('obs', mu=exp_cv, sigma=sigma_cv, observed=Y_train)
        trace_cv = pm.sample(500, tune=500, chains=1, cores=1,
                             random_seed=42, progressbar=False,
                             target_accept=0.85)

    # Predict held-out region
    mu_cv_hat = trace_cv.posterior['mu'].mean(dim=['chain', 'draw']).values
    tau_cv_hat = trace_cv.posterior['tau'].mean(dim=['chain', 'draw']).values
    sigma_cv_hat = trace_cv.posterior['sigma'].mean(dim=['chain', 'draw']).values

    # For the held-out region, prediction uses the prior: μ (shrunk to national)
    # plus any FB effect
    fb_test = fb_change_std[test_mask]
    fb_mask_test = has_fb_mask[test_mask]
    beta_fb_cv_hat = trace_cv.posterior['beta_fb'].mean(dim=['chain', 'draw']).values

    pred_test = mu_cv_hat[None, :] + beta_fb_cv_hat[None, :] * (fb_test[:, None] * fb_mask_test[:, None])
    residuals = Y_test - pred_test

    # Expected variance = τ² + σ² (prior predictive for new region)
    expected_var = tau_cv_hat**2 + sigma_cv_hat**2

    # Standardised residuals
    std_resid = residuals / np.sqrt(expected_var)[None, :]

    rmse = np.sqrt(np.mean(residuals**2, axis=0))
    coverage_95 = np.mean(np.abs(std_resid) < 1.96, axis=0)
    coverage_80 = np.mean(np.abs(std_resid) < 1.28, axis=0)

    region_cv_results.append({
        'region': hold_region,
        'n_test': n_test,
        'rmse': rmse,
        'coverage_95': coverage_95,
        'coverage_80': coverage_80,
        'mean_std_resid': np.mean(np.abs(std_resid), axis=0)
    })

    print(f"  {hold_region:25s} (n={n_test:3d}): "
          f"RMSE={rmse.mean():.3f}, "
          f"95% cov={coverage_95.mean():.1%}, "
          f"80% cov={coverage_80.mean():.1%}")

# Aggregate CV metrics
all_rmse = np.array([r['rmse'] for r in region_cv_results])
all_cov95 = np.array([r['coverage_95'] for r in region_cv_results])
all_cov80 = np.array([r['coverage_80'] for r in region_cv_results])

print(f"\nAggregate LORO-CV:")
for i, p in enumerate(PARTIES_GB):
    print(f"  {p}: mean RMSE={all_rmse[:,i].mean():.3f}, "
          f"95% coverage={all_cov95[:,i].mean():.1%} (target 95%), "
          f"80% coverage={all_cov80[:,i].mean():.1%} (target 80%)")

# Flag calibration issues
for i, p in enumerate(PARTIES_GB):
    cov95 = all_cov95[:, i].mean()
    cov80 = all_cov80[:, i].mean()
    if cov95 < 0.88:
        print(f"  ⚠️ {p}: 95% interval is UNDERCONFIDENT (coverage {cov95:.1%} < 88%)")
        print(f"     → Consider tightening τ or σ priors")
    elif cov95 > 0.98:
        print(f"  ⚠️ {p}: 95% interval is OVERCONFIDENT (coverage {cov95:.1%} > 98%)")
        print(f"     → Intervals too wide; τ or σ may be inflated")

# ── 8. Empirical polling error covariance (Improvement #4) ──────────────────
# Combines:
#   (a) Historical systematic polling misses (2015, 2017, 2019, 2024)
#   (b) Time-series residuals from LOWESS smoothing

print("\n=== Empirical Polling Error Covariance ===")

# 8a. Historical polling misses (final polls vs actual results)
# Sources: UK Polling Report, Wikipedia, academic literature
# Format: {election: {party: (final_poll_avg, actual_result)}}
HISTORICAL_MISSES = {
    2015: {
        'Lab': (33.6, 30.4),   'Con': (34.0, 36.8),
        'LD':  (8.0, 7.9),     'RUK': (13.0, 12.6),
        'Green': (5.0, 3.8),   'SNP': (4.0, 4.7),
        'PC':  (0.6, 0.6),
    },
    2017: {
        'Lab': (36.0, 40.0),   'Con': (43.0, 42.3),
        'LD':  (8.0, 7.4),     'RUK': (3.0, 1.8),
        'Green': (2.0, 1.6),   'SNP': (4.0, 3.0),
        'PC':  (0.5, 0.5),
    },
    2019: {
        'Lab': (33.0, 32.1),   'Con': (43.0, 43.6),
        'LD':  (12.0, 11.6),   'RUK': (3.0, 2.0),
        'Green': (3.0, 2.7),   'SNP': (4.0, 3.9),
        'PC':  (1.0, 0.5),
    },
    2024: {
        'Lab': (37.0, 33.7),   'Con': (20.0, 23.7),
        'LD':  (11.0, 12.2),   'RUK': (17.0, 14.3),
        'Green': (6.0, 6.8),   'SNP': (3.0, 2.5),
        'PC':  (1.0, 0.7),
    },
}

POLL_PARTY_ORDER = ['Lab', 'Con', 'RUK', 'LD', 'Green', 'SNP', 'PC']
n_poll_parties = len(POLL_PARTY_ORDER)

# Compute error vectors (poll - actual) for each election
hist_errors = []
for year, misses in HISTORICAL_MISSES.items():
    err = np.array([misses[p][0] - misses[p][1] for p in POLL_PARTY_ORDER])
    hist_errors.append(err)
    print(f"  {year} errors: " +
          ", ".join(f"{p}={err[i]:+.1f}" for i, p in enumerate(POLL_PARTY_ORDER)))

hist_errors = np.array(hist_errors)  # (4, 7)

# Historical systematic covariance (small sample, so use shrinkage)
hist_mean = hist_errors.mean(axis=0)
hist_cov_raw = np.cov(hist_errors.T)

print(f"\n  Mean historical bias: " +
      ", ".join(f"{p}={hist_mean[i]:+.2f}" for i, p in enumerate(POLL_PARTY_ORDER)))

# 8b. Time-series residuals from polling data
# Fit LOWESS to each party's polling series and compute residuals
print("\n  Computing LOWESS residuals from polling time series...")

poll_df = polling_data.copy()
poll_df['date'] = pd.to_datetime(poll_df['date'])
# Exclude the election result row
poll_df = poll_df[poll_df['Pollster'] != '2024 general election'].copy()
poll_df = poll_df.sort_values('date').reset_index(drop=True)
poll_df['t'] = (poll_df['date'] - poll_df['date'].min()).dt.days

party_col_map = {'Lab': 'Lab', 'Con': 'Con', 'RUK': 'Ref', 'LD': 'LD',
                 'Green': 'Grn', 'SNP': 'SNP', 'PC': 'PC'}

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    ts_residuals = {}
    for party, col in party_col_map.items():
        if col not in poll_df.columns:
            continue
        valid = poll_df[['t', col]].dropna()
        if len(valid) < 20:
            continue
        t_vals = valid['t'].values.astype(float)
        y_vals = valid[col].values.astype(float)

        # LOWESS smooth (frac=0.15 for responsive but smooth curve)
        smoothed = sm_lowess(y_vals, t_vals, frac=0.15, return_sorted=True)
        resid = y_vals - smoothed[:, 1]
        ts_residuals[party] = resid
        print(f"    {party:5s}: {len(resid)} polls, "
              f"resid sd={resid.std():.2f}pp, "
              f"mean={resid.mean():.2f}pp")

    # Build residual matrix (align by using shortest length)
    min_len = min(len(v) for v in ts_residuals.values())
    resid_matrix = np.column_stack([
        ts_residuals[p][:min_len] for p in POLL_PARTY_ORDER
        if p in ts_residuals
    ])

    ts_cov = np.cov(resid_matrix.T)
    print(f"\n  Time-series residual covariance ({resid_matrix.shape[0]} obs):")

except ImportError:
    print("  statsmodels not available, using raw polling variance")
    ts_cov = None

# 8c. Combine: systematic (historical) + idiosyncratic (time-series)
# The total polling error is the sum of:
#   - Systematic bias (correlated across all polls in an election): hist_cov
#   - Idiosyncratic noise (varies poll-to-poll): ts_cov / n_polls_in_average
# For projection, we care about the error in the polling *average*,
# not individual polls, so the idiosyncratic component is reduced.

# Shrinkage estimator for historical covariance (4 elections is very small)
# Use Ledoit-Wolf-style shrinkage toward diagonal
alpha_shrink = 0.4  # moderate shrinkage given n=4
hist_diag = np.diag(np.diag(hist_cov_raw))
hist_cov_shrunk = (1 - alpha_shrink) * hist_cov_raw + alpha_shrink * hist_diag

# Number of recent polls in a typical average (last 2 weeks)
n_polls_avg = 15

if ts_cov is not None:
    # Scale ts_cov by 1/n_polls to get covariance of the average
    idio_cov = ts_cov / n_polls_avg

    # Ensure dimensions match (ts_cov might not include all parties)
    ts_parties = [p for p in POLL_PARTY_ORDER if p in ts_residuals]
    if len(ts_parties) == n_poll_parties:
        POLL_COV_COMBINED = hist_cov_shrunk + idio_cov
    else:
        # Map available parties into full matrix
        POLL_COV_COMBINED = hist_cov_shrunk.copy()
        for i, pi in enumerate(ts_parties):
            for j, pj in enumerate(ts_parties):
                ii = POLL_PARTY_ORDER.index(pi)
                jj = POLL_PARTY_ORDER.index(pj)
                POLL_COV_COMBINED[ii, jj] += idio_cov[i, j]
else:
    POLL_COV_COMBINED = hist_cov_shrunk

# Ensure positive semi-definite
eigvals, eigvecs = np.linalg.eigh(POLL_COV_COMBINED)
eigvals = np.maximum(eigvals, 0.01)  # floor small eigenvalues
POLL_COV_COMBINED = eigvecs @ np.diag(eigvals) @ eigvecs.T
# Symmetrise
POLL_COV_COMBINED = (POLL_COV_COMBINED + POLL_COV_COMBINED.T) / 2

print(f"\n  Combined polling error covariance matrix:")
print(f"  Parties: {POLL_PARTY_ORDER}")
print(f"  Standard deviations: " +
      ", ".join(f"{p}={np.sqrt(POLL_COV_COMBINED[i,i]):.2f}"
                for i, p in enumerate(POLL_PARTY_ORDER)))
print(f"\n  Correlation matrix:")
poll_sds = np.sqrt(np.diag(POLL_COV_COMBINED))
poll_corr = POLL_COV_COMBINED / np.outer(poll_sds, poll_sds)
for i, p in enumerate(POLL_PARTY_ORDER):
    row = "  " + f"{p:5s}" + " ".join(f"{poll_corr[i,j]:+.2f}" for j in range(n_poll_parties))
    print(row)

# ── 9. Extract posterior means for projection ────────────────────────────────
post = trace.posterior

mu_hat = post['mu'].mean(dim=['chain', 'draw']).values
tau_hat = post['tau'].mean(dim=['chain', 'draw']).values
sigma_hat = post['sigma'].mean(dim=['chain', 'draw']).values
gamma_hat = post['gamma'].mean(dim=['chain', 'draw']).values  # (n_regions, 6)
beta_fb_hat = post['beta_fb'].mean(dim=['chain', 'draw']).values

mu_sd = post['mu'].std(dim=['chain', 'draw']).values
gamma_sd = post['gamma'].std(dim=['chain', 'draw']).values

print(f"\nPosterior means (v2, 6 parties):")
for i, p in enumerate(ALL_MODEL_PARTIES):
    print(f"  {p:5s}: μ={mu_hat[i]:+.4f}, τ={tau_hat[i]:.4f}, "
          f"σ={sigma_hat[i]:.4f}, β_fb={beta_fb_hat[i]:+.4f}")

print(f"\nRegional effects (γ):")
for r, ri in sorted(region_map.items()):
    vals = ", ".join(f"{p}={gamma_hat[ri, i]:+.3f}" for i, p in enumerate(ALL_MODEL_PARTIES))
    print(f"  {r:25s}: {vals}")

# ── 10. Project forward: all 650 constituencies ──────────────────────────────
# Current polling (Kalman smoother) — will be overridden in posterior_predictive
POLLS = {"Lab": 17.08, "Con": 17.28, "RUK": 25.70, "LD": 11.60, "Green": 16.78}
NAT2024 = {"Lab": 34.64, "Con": 24.36, "RUK": 14.69, "LD": 12.56, "Green": 6.91}
SNP_POLLS = 32.53
SNP_2024 = 30.01
PC_POLLS = 19.20
PC_2024 = 14.77

# National log-ratio swing implied by polls
national_delta = {}
for p in PARTIES_GB:
    eta_polls = np.log(max(POLLS[p], 0.5) / max(POLLS['Lab'], 0.5))
    eta_2024 = np.log(max(NAT2024[p], 0.5) / max(NAT2024['Lab'], 0.5))
    national_delta[p] = eta_polls - eta_2024

# SNP and PC now use the same log-ratio framework
# Convert regional polls to GB-equivalent for the ALR transform
snp_gb = SNP_POLLS * 0.087  # Scotland share of GB electorate
pc_gb = PC_POLLS * 0.05     # Wales share of GB electorate
snp_gb_2024 = SNP_2024 * 0.087
pc_gb_2024 = PC_2024 * 0.05

# But for the ALR, we need the log-ratio relative to Labour
# SNP: log(snp_scotland / lab_scotland) → use Scotland-specific shares
# We approximate: in Scotland, Lab polled ~X%, SNP polled ~Y%
# For simplicity, use the proportional approach converted to ALR delta
eta_snp_polls = np.log(max(SNP_POLLS, 0.5) / max(POLLS['Lab'], 0.5))
eta_snp_2024 = np.log(max(SNP_2024, 0.5) / max(NAT2024['Lab'], 0.5))
national_delta['SNP'] = eta_snp_polls - eta_snp_2024

eta_pc_polls = np.log(max(PC_POLLS, 0.5) / max(POLLS['Lab'], 0.5))
eta_pc_2024 = np.log(max(PC_2024, 0.5) / max(NAT2024['Lab'], 0.5))
national_delta['PC'] = eta_pc_polls - eta_pc_2024

print(f"\nNational log-ratio swing from polls:")
for p in ALL_MODEL_PARTIES:
    print(f"  {p}: {national_delta[p]:.4f}")

# Generate constituency-level projections
all_2024 = results_2024[results_2024['Country name'].isin(['England', 'Scotland', 'Wales'])].copy()
all_2024 = all_2024.merge(
    foreign_born[['ONSConstID_y', 'pct_point_change']],
    left_on='ONS ID', right_on='ONSConstID_y', how='left'
)

projections = []
for _, row in all_2024.iterrows():
    cid = row['ONS ID']
    name = row['Constituency name']
    region = row['Region name']
    country = row['Country name']
    valid = row['Valid votes']

    shares_2024 = {}
    for p in ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC',
              'DUP', 'SF', 'SDLP', 'UUP', 'APNI']:
        shares_2024[p] = (row.get(p, 0) or 0) / valid * 100 if valid > 0 else 0

    new_shares = {}
    lab_share_2024 = max(shares_2024['Lab'], FLOOR)

    # Determine which parties to project with ALR
    parties_for_seat = list(PARTIES_GB)
    if country == 'Scotland':
        parties_for_seat.append('SNP')
    if country == 'Wales':
        parties_for_seat.append('PC')

    for p in parties_for_seat:
        i = ALL_MODEL_PARTIES.index(p)
        p_share_2024 = max(shares_2024[p], FLOOR)
        eta_2024 = np.log(p_share_2024 / lab_share_2024)

        if region in region_map:
            r_idx = region_map[region]
            delta_hat = gamma_hat[r_idx, i]
        else:
            delta_hat = mu_hat[i]

        regional_deviation = delta_hat - mu_hat[i]
        delta_projected = national_delta[p] + regional_deviation

        if pd.notna(row.get('pct_point_change')):
            fb_val = (row['pct_point_change'] - fb_mean) / fb_std
            delta_projected += beta_fb_hat[i] * fb_val

        eta_new = eta_2024 + delta_projected
        new_shares[p] = eta_new  # store log-ratio temporarily

    # Parties NOT modelled hierarchically for this seat
    if country != 'Scotland':
        new_shares['SNP'] = shares_2024['SNP']  # actual share, not log-ratio
    if country != 'Wales':
        new_shares['PC'] = shares_2024['PC']

    # Convert log-ratios back to shares
    alr_parties = parties_for_seat
    non_alr_parties = [q for q in ['SNP', 'PC'] if q not in parties_for_seat]

    exp_etas = {p: np.exp(new_shares[p]) for p in alr_parties}
    sum_exp = 1.0 + sum(exp_etas.values())

    other_share = sum(shares_2024.get(p, 0) for p in ['DUP', 'SF', 'SDLP', 'UUP', 'APNI'])
    non_alr_share = sum(new_shares.get(p, 0) for p in non_alr_parties)
    remaining = 100.0 - non_alr_share - other_share
    lab_new = max(remaining / sum_exp, 0.1)

    final_shares = {'Lab': lab_new}
    for p in alr_parties:
        final_shares[p] = lab_new * exp_etas[p]
    for p in non_alr_parties:
        final_shares[p] = new_shares.get(p, 0)
    for p in ['DUP', 'SF', 'SDLP', 'UUP', 'APNI']:
        final_shares[p] = shares_2024.get(p, 0)

    # Renormalize
    total = sum(max(v, 0) for v in final_shares.values())
    if total > 0:
        orig_total = sum(shares_2024.values())
        scale = orig_total / total if total > 0 else 1
        for p in final_shares:
            final_shares[p] = max(final_shares[p] * scale, 0)

    projections.append({
        'id': cid, 'name': name, 'region': region,
        'country': country, 'shares': final_shares
    })

# Add NI constituencies unchanged
ni_seats = results_2024[results_2024['Country name'] == 'Northern Ireland']
for _, row in ni_seats.iterrows():
    valid = row['Valid votes']
    shares = {}
    for p in ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC',
              'DUP', 'SF', 'SDLP', 'UUP', 'APNI']:
        shares[p] = (row.get(p, 0) or 0) / valid * 100 if valid > 0 else 0
    projections.append({
        'id': row['ONS ID'], 'name': row['Constituency name'],
        'region': 'Northern Ireland', 'country': 'Northern Ireland',
        'shares': shares
    })

print(f"\nGenerated projections for {len(projections)} constituencies")

# Seat counts
seat_counts = {}
for proj in projections:
    winner = max(proj['shares'], key=proj['shares'].get)
    seat_counts[winner] = seat_counts.get(winner, 0) + 1
    proj['winner'] = winner

print("\nProjected seat counts (Bayesian hierarchical v2, Kalman polls):")
for p in sorted(seat_counts, key=seat_counts.get, reverse=True):
    print(f"  {p}: {seat_counts[p]}")

# ── 11. Export ───────────────────────────────────────────────────────────────
OUT_BASE = BASE

bayes_proj = {}
for proj in projections:
    bayes_proj[proj['id']] = {p: round(v, 3) for p, v in proj['shares'].items()}

with open(OUT_BASE + 'bayes_projections_v2.json', 'w') as f:
    json.dump(bayes_proj, f)

# Export model parameters
model_params = {
    'version': 2,
    'parties': ALL_MODEL_PARTIES,
    'mu': {p: float(v) for p, v in zip(ALL_MODEL_PARTIES, mu_hat)},
    'tau': {p: float(v) for p, v in zip(ALL_MODEL_PARTIES, tau_hat)},
    'sigma': {p: float(v) for p, v in zip(ALL_MODEL_PARTIES, sigma_hat)},
    'beta_fb': {p: float(v) for p, v in zip(ALL_MODEL_PARTIES, beta_fb_hat)},
    'regions': {r: {p: float(gamma_hat[i, j]) for j, p in enumerate(ALL_MODEL_PARTIES)}
                for r, i in region_map.items()},
    'n_constituencies': int(n_constituencies),
    'n_regions': int(n_regions),
    'fb_coverage': int(has_fb.sum()),
    'fb_mean': float(fb_mean),
    'fb_std': float(fb_std),
    'poll_cov': {
        'parties': POLL_PARTY_ORDER,
        'matrix': POLL_COV_COMBINED.tolist(),
        'correlation': poll_corr.tolist(),
        'sds': poll_sds.tolist(),
        'sources': ['historical_misses_2015_2017_2019_2024', 'lowess_residuals'],
        'shrinkage_alpha': alpha_shrink,
        'n_polls_avg': n_polls_avg,
    },
    'cv_diagnostics': {
        'loo_elpd': float(loo.elpd_loo) if hasattr(loo, 'elpd_loo') else None,
        'loo_se': float(loo.se) if hasattr(loo, 'se') else None,
        'region_cv': [{
            'region': r['region'],
            'n_test': int(r['n_test']),
            'rmse': r['rmse'].tolist(),
            'coverage_95': r['coverage_95'].tolist(),
            'coverage_80': r['coverage_80'].tolist(),
        } for r in region_cv_results],
    }
}

with open(OUT_BASE + 'model_params_v2.json', 'w') as f:
    json.dump(model_params, f, indent=2)

print("\nExported v2 projections and model parameters.")
print("Done!")
