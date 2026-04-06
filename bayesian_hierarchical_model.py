"""
Bayesian Hierarchical Model for UK Constituency-Level Vote Shares

Model specification:
  Level 1: δ_{c,p} ~ N(γ_{r[c],p} + β_{1,p}·ΔF_c, σ²_p)
  Level 2: γ_{r,p} ~ N(μ_p, τ²_p)
  Priors:  μ_p ~ N(0, 5²), σ_p ~ HalfCauchy(2.5), τ_p ~ HalfCauchy(2.5)

The model is fit on 2019→2024 log-ratio swings (calibrating variance components),
then used to project forward given current polling.
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
BASE = '/sessions/sweet-amazing-dijkstra/mnt/Election_modelling/'

results_2024 = pd.read_csv(BASE + 'HoC-GE2024-results-by-constituency.csv')
results_2019 = pd.read_csv(BASE + 'HoC-GE2019-results-by-constituency.csv')
successors = pd.read_csv(BASE + 'constituency_closest_successors_final.csv')
foreign_born = pd.read_csv(BASE + 'change in foreign born.csv')

print(f"2024 results: {len(results_2024)} constituencies")
print(f"2019 results: {len(results_2019)} constituencies")
print(f"Successors mapping: {len(successors)} rows")
print(f"Foreign born data: {len(foreign_born)} rows")

# ── 2. Merge 2019 and 2024 results via boundary mapping ──────────────────────
# Map 2019 constituency to 2024 successor
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

# GB parties only (exclude NI)
merged = merged[merged['Country name_24'].isin(['England', 'Scotland', 'Wales'])].copy()

# Compute vote shares
parties_19 = ['Con', 'Lab', 'LD', 'BRX', 'Green', 'SNP', 'PC']
parties_24 = ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC']

for p in parties_19:
    col = f'{p}_19' if f'{p}_19' in merged.columns else p
    if col in merged.columns:
        merged[f'share_{p}_19'] = merged[col] / merged['Valid votes_19'] * 100

# Rename BRX -> RUK for 2019
if 'share_BRX_19' in merged.columns:
    merged['share_RUK_19'] = merged['share_BRX_19']

for p in parties_24:
    col_24 = f'{p}_24' if f'{p}_24' in merged.columns else p
    if col_24 in merged.columns:
        merged[f'share_{p}_24'] = merged[col_24] / merged['Valid votes_24'] * 100

# ── 3. Compute log-ratio swings (proportional swing in log space) ─────────────
# Reference party = Labour (present in virtually all GB seats)
# ALR transform: η_{c,p} = log(share_p / share_Lab)
# Swing in log-ratio space: δ = η_2024 - η_2019

PARTIES = ['Con', 'RUK', 'LD', 'Green']  # relative to Labour
FLOOR = 0.5  # floor at 0.5% to avoid log(0)

for p in PARTIES:
    s19 = np.maximum(merged[f'share_{p}_19'].fillna(0).values, FLOOR)
    lab19 = np.maximum(merged['share_Lab_19'].fillna(0).values, FLOOR)
    s24 = np.maximum(merged[f'share_{p}_24'].fillna(0).values, FLOOR)
    lab24 = np.maximum(merged['share_Lab_24'].fillna(0).values, FLOOR)

    merged[f'eta_{p}_19'] = np.log(s19 / lab19)
    merged[f'eta_{p}_24'] = np.log(s24 / lab24)
    merged[f'delta_{p}'] = merged[f'eta_{p}_24'] - merged[f'eta_{p}_19']

# Also compute SNP and PC log-ratios for Scotland/Wales separately
for p in ['SNP', 'PC']:
    s19 = np.maximum(merged[f'share_{p}_19'].fillna(0).values, FLOOR)
    lab19 = np.maximum(merged['share_Lab_19'].fillna(0).values, FLOOR)
    s24 = np.maximum(merged[f'share_{p}_24'].fillna(0).values, FLOOR)
    lab24 = np.maximum(merged['share_Lab_24'].fillna(0).values, FLOOR)
    merged[f'eta_{p}_19'] = np.log(s19 / lab19)
    merged[f'eta_{p}_24'] = np.log(s24 / lab24)
    merged[f'delta_{p}'] = merged[f'eta_{p}_24'] - merged[f'eta_{p}_19']

# ── 4. Prepare region indices ─────────────────────────────────────────────────
# Drop rows with NaN deltas
delta_cols = [f'delta_{p}' for p in PARTIES]
model_data = merged.dropna(subset=delta_cols + ['Region name_24']).copy()

# Filter out extreme outliers (> 4 sd)
for col in delta_cols:
    m, s = model_data[col].mean(), model_data[col].std()
    model_data = model_data[model_data[col].between(m - 4*s, m + 4*s)]

regions = model_data['Region name_24'].unique()
region_map = {r: i for i, r in enumerate(sorted(regions))}
model_data['region_idx'] = model_data['Region name_24'].map(region_map)

n_regions = len(regions)
n_constituencies = len(model_data)
n_parties = len(PARTIES)

print(f"\nModel data: {n_constituencies} constituencies, {n_regions} regions, {n_parties} parties")
print(f"Regions: {sorted(regions)}")

# ── 5. Merge foreign born change data ────────────────────────────────────────
# Map to 2024 constituencies
fb_merged = model_data.merge(
    foreign_born[['ONSConstID_y', 'pct_point_change', 'foreign_pc_2021']],
    left_on='2024_code', right_on='ONSConstID_y', how='left'
)

has_fb = fb_merged['pct_point_change'].notna()
print(f"Foreign born data available for {has_fb.sum()} of {len(fb_merged)} constituencies")

# Standardise the covariate for numerical stability
fb_change = fb_merged['pct_point_change'].values.copy()
fb_mean = np.nanmean(fb_change)
fb_std = np.nanstd(fb_change)
fb_change_std = (fb_change - fb_mean) / fb_std
# Replace NaN with 0 (will be masked out effectively by the model)
fb_change_std = np.nan_to_num(fb_change_std, nan=0.0)
has_fb_mask = has_fb.values.astype(float)

# ── 6. Fit the Bayesian hierarchical model ────────────────────────────────────
# Observed data: deltas (n_constituencies x n_parties)
Y = model_data[delta_cols].values  # (N, P)
region_idx = model_data['region_idx'].values

print(f"\nObserved swing stats:")
for i, p in enumerate(PARTIES):
    print(f"  {p}: mean={Y[:,i].mean():.3f}, sd={Y[:,i].std():.3f}")

print("\nFitting Bayesian hierarchical model...")

with pm.Model() as hierarchical_model:
    # Hyperpriors — national-level mean swing for each party
    mu = pm.Normal('mu', mu=0, sigma=3, shape=n_parties)

    # Between-region variance
    tau = pm.HalfCauchy('tau', beta=1.5, shape=n_parties)

    # Within-region (constituency) variance
    sigma = pm.HalfCauchy('sigma', beta=1.5, shape=n_parties)

    # Region-level means
    gamma = pm.Normal('gamma', mu=mu, sigma=tau, shape=(n_regions, n_parties))

    # Foreign-born change coefficient (per party)
    beta_fb = pm.Normal('beta_fb', mu=0, sigma=1, shape=n_parties)

    # Constituency-level expected swing
    # γ_{r[c],p} + β_p * ΔF_c (where ΔF is available)
    region_effect = gamma[region_idx]  # (N, P)
    fb_effect = beta_fb[None, :] * (fb_change_std[:, None] * has_fb_mask[:, None])

    expected = region_effect + fb_effect

    # Likelihood
    obs = pm.Normal('obs', mu=expected, sigma=sigma, observed=Y)

    # Sample
    trace = pm.sample(2000, tune=1500, chains=2, cores=1,
                      random_seed=42, progressbar=True,
                      target_accept=0.9)

print("\nSampling complete!")
summary = az.summary(trace, var_names=['mu', 'tau', 'sigma', 'beta_fb'])
print(summary)

# ── 7. Extract posterior means for projection ─────────────────────────────────
post = trace.posterior

mu_hat = post['mu'].mean(dim=['chain', 'draw']).values
tau_hat = post['tau'].mean(dim=['chain', 'draw']).values
sigma_hat = post['sigma'].mean(dim=['chain', 'draw']).values
gamma_hat = post['gamma'].mean(dim=['chain', 'draw']).values  # (n_regions, n_parties)
beta_fb_hat = post['beta_fb'].mean(dim=['chain', 'draw']).values

# Also get posterior standard deviations for uncertainty
mu_sd = post['mu'].std(dim=['chain', 'draw']).values
gamma_sd = post['gamma'].std(dim=['chain', 'draw']).values

print(f"\nPosterior means:")
print(f"  National mean (mu): {dict(zip(PARTIES, mu_hat))}")
print(f"  Between-region sd (tau): {dict(zip(PARTIES, tau_hat))}")
print(f"  Within-region sd (sigma): {dict(zip(PARTIES, sigma_hat))}")
print(f"  Foreign-born beta: {dict(zip(PARTIES, beta_fb_hat))}")

# ── 8. Project forward: apply to all 650 constituencies ──────────────────────
# For the projection, we need:
# - Current polling (national vote shares) -> convert to log-ratio shift relative to 2024
# - Apply hierarchical structure: use region-level posterior means
# - Constituencies without region match -> use national mean

# Current polling from smoother presets (Kalman)
POLLS = {"Lab": 15.72, "Con": 18.13, "RUK": 25.54, "LD": 10.24, "Green": 18.63}
NAT2024 = {"Lab": 34.64, "Con": 24.36, "RUK": 14.69, "LD": 12.56, "Green": 6.91}
SNP_POLLS = 34.95
SNP_2024 = 30.01
PC_POLLS = 21.85
PC_2024 = 14.77

# National log-ratio swing implied by polls (Lab is reference)
national_delta = {}
for p in PARTIES:
    eta_polls = np.log(max(POLLS[p], 0.5) / max(POLLS['Lab'], 0.5))
    eta_2024 = np.log(max(NAT2024[p], 0.5) / max(NAT2024['Lab'], 0.5))
    national_delta[p] = eta_polls - eta_2024

print(f"\nNational log-ratio swing from polls:")
for p in PARTIES:
    print(f"  {p}: {national_delta[p]:.4f}")

# Now generate constituency-level projections
# Strategy: scale the posterior regional effects by the ratio of
# (poll-implied national swing) / (estimated 2019-2024 national swing)
# This preserves the relative regional structure while shifting to current polls

# For each constituency in the 2024 results:
all_2024 = results_2024[results_2024['Country name'].isin(['England', 'Scotland', 'Wales'])].copy()

# Merge foreign born data
all_2024 = all_2024.merge(
    foreign_born[['ONSConstID_y', 'pct_point_change']],
    left_on='ONS ID', right_on='ONSConstID_y', how='left'
)

# Build projection
projections = []
for _, row in all_2024.iterrows():
    cid = row['ONS ID']
    name = row['Constituency name']
    region = row['Region name']
    country = row['Country name']
    valid = row['Valid votes']

    # Current 2024 shares
    shares_2024 = {}
    for p in ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC',
              'DUP', 'SF', 'SDLP', 'UUP', 'APNI']:
        shares_2024[p] = (row.get(p, 0) or 0) / valid * 100 if valid > 0 else 0

    # Project GB parties using hierarchical model
    new_shares = {}
    lab_share_2024 = max(shares_2024['Lab'], FLOOR)

    for i, p in enumerate(PARTIES):
        p_share_2024 = max(shares_2024[p], FLOOR)
        eta_2024 = np.log(p_share_2024 / lab_share_2024)

        # Get region-level swing estimate
        if region in region_map:
            r_idx = region_map[region]
            delta_hat = gamma_hat[r_idx, i]
        else:
            delta_hat = mu_hat[i]

        # Scale by polling: adjust the historical regional effect
        # to be centred on the poll-implied national swing
        # δ_projected = national_delta_polls + (γ_r - μ_historical)
        regional_deviation = delta_hat - mu_hat[i]
        delta_projected = national_delta[p] + regional_deviation

        # Add foreign born effect if available
        if pd.notna(row.get('pct_point_change')):
            fb_val = (row['pct_point_change'] - fb_mean) / fb_std
            delta_projected += beta_fb_hat[i] * fb_val

        # Project
        eta_new = eta_2024 + delta_projected
        # Convert back: share_p_new / share_lab_new = exp(eta_new)
        # We store the log-ratio and convert after all parties done
        new_shares[p] = eta_new

    # Handle SNP and PC
    if country == 'Scotland':
        snp_share_2024 = max(shares_2024['SNP'], FLOOR)
        snp_ratio = max(SNP_POLLS, 0.5) / max(SNP_2024, 0.5)
        new_shares['SNP'] = snp_share_2024 * snp_ratio
    else:
        new_shares['SNP'] = shares_2024['SNP']

    if country == 'Wales':
        pc_share_2024 = max(shares_2024['PC'], FLOOR)
        pc_ratio = max(PC_POLLS, 0.5) / max(PC_2024, 0.5)
        new_shares['PC'] = pc_share_2024 * pc_ratio
    else:
        new_shares['PC'] = shares_2024['PC']

    # Convert log-ratios back to shares
    # We need to solve for share_Lab_new
    # For each party p: share_p = share_Lab * exp(eta_p)
    # Sum of all shares = 100 (after including SNP/PC/others)

    # exp(eta) values for PARTIES
    exp_etas = {p: np.exp(new_shares[p]) for p in PARTIES}

    # Other shares (NI parties, Ind, etc.) stay fixed
    other_share = sum(shares_2024.get(p, 0) for p in ['DUP', 'SF', 'SDLP', 'UUP', 'APNI'])
    # Add any independent/other that might be in the original

    snp_pc_share = new_shares.get('SNP', 0) + new_shares.get('PC', 0)

    # share_Lab * (1 + sum(exp(eta_p))) + snp_pc + other = 100
    sum_exp = 1.0 + sum(exp_etas.values())
    remaining = 100.0 - snp_pc_share - other_share
    lab_new = max(remaining / sum_exp, 0.1)

    final_shares = {'Lab': lab_new}
    for p in PARTIES:
        final_shares[p] = lab_new * exp_etas[p]
    final_shares['SNP'] = new_shares.get('SNP', 0)
    final_shares['PC'] = new_shares.get('PC', 0)
    for p in ['DUP', 'SF', 'SDLP', 'UUP', 'APNI']:
        final_shares[p] = shares_2024.get(p, 0)

    # Ensure non-negative and renormalize
    total = sum(max(v, 0) for v in final_shares.values())
    if total > 0:
        orig_total = sum(shares_2024.values())
        scale = orig_total / total if total > 0 else 1
        for p in final_shares:
            final_shares[p] = max(final_shares[p] * scale, 0)

    projections.append({
        'id': cid,
        'name': name,
        'region': region,
        'country': country,
        'shares': final_shares
    })

# Also add NI constituencies unchanged
ni_seats = results_2024[results_2024['Country name'] == 'Northern Ireland']
for _, row in ni_seats.iterrows():
    valid = row['Valid votes']
    shares = {}
    for p in ['Con', 'Lab', 'LD', 'RUK', 'Green', 'SNP', 'PC',
              'DUP', 'SF', 'SDLP', 'UUP', 'APNI']:
        shares[p] = (row.get(p, 0) or 0) / valid * 100 if valid > 0 else 0
    projections.append({
        'id': row['ONS ID'],
        'name': row['Constituency name'],
        'region': 'Northern Ireland',
        'country': 'Northern Ireland',
        'shares': shares
    })

print(f"\nGenerated projections for {len(projections)} constituencies")

# ── 9. Compute seat counts ───────────────────────────────────────────────────
seat_counts = {}
for proj in projections:
    winner = max(proj['shares'], key=proj['shares'].get)
    seat_counts[winner] = seat_counts.get(winner, 0) + 1
    proj['winner'] = winner

print("\nProjected seat counts (Bayesian hierarchical, Kalman polls):")
for p in sorted(seat_counts, key=seat_counts.get, reverse=True):
    print(f"  {p}: {seat_counts[p]}")

# ── 10. Export as JSON for the dashboard ──────────────────────────────────────
# Format: {constituency_id: {party: projected_share, ...}}
bayes_proj = {}
for proj in projections:
    bayes_proj[proj['id']] = {
        p: round(v, 3) for p, v in proj['shares'].items()
    }

with open('/sessions/sweet-amazing-dijkstra/bayes_projections.json', 'w') as f:
    json.dump(bayes_proj, f)

# Also export model parameters for the methodology section
model_params = {
    'mu': {p: float(v) for p, v in zip(PARTIES, mu_hat)},
    'tau': {p: float(v) for p, v in zip(PARTIES, tau_hat)},
    'sigma': {p: float(v) for p, v in zip(PARTIES, sigma_hat)},
    'beta_fb': {p: float(v) for p, v in zip(PARTIES, beta_fb_hat)},
    'regions': {r: {p: float(gamma_hat[i, j]) for j, p in enumerate(PARTIES)}
                for r, i in region_map.items()},
    'n_constituencies': n_constituencies,
    'n_regions': n_regions,
    'fb_coverage': int(has_fb.sum()),
    'fb_mean': float(fb_mean),
    'fb_std': float(fb_std)
}

with open('/sessions/sweet-amazing-dijkstra/model_params.json', 'w') as f:
    json.dump(model_params, f, indent=2)

print("\nExported projections and model parameters.")
print("Done!")
