"""
Posterior Predictive Simulation for UK Election Projections

Uses the full MCMC posterior (not just point estimates) to generate:
1. Seat count distributions per party (with credible intervals)
2. Constituency-level win probabilities
3. Incorporates polling uncertainty on top of model parameter uncertainty
"""

import pandas as pd
import numpy as np
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

with open('/sessions/sweet-amazing-dijkstra/model_params.json') as f:
    params = json.load(f)

# ── 2. Re-fit model to get full posterior samples ─────────────────────────────
# (We need the actual trace, not just the summary stats)
print("Re-fitting Bayesian hierarchical model to extract posterior draws...")

# Prepare data (same as before)
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

PARTIES = ['Con', 'RUK', 'LD', 'Green']
FLOOR = 0.5

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

for p in PARTIES + ['SNP', 'PC']:
    s19 = np.maximum(merged[f'share_{p}_19'].fillna(0).values, FLOOR)
    lab19 = np.maximum(merged['share_Lab_19'].fillna(0).values, FLOOR)
    s24 = np.maximum(merged[f'share_{p}_24'].fillna(0).values, FLOOR)
    lab24 = np.maximum(merged['share_Lab_24'].fillna(0).values, FLOOR)
    merged[f'eta_{p}_19'] = np.log(s19 / lab19)
    merged[f'eta_{p}_24'] = np.log(s24 / lab24)
    merged[f'delta_{p}'] = merged[f'eta_{p}_24'] - merged[f'eta_{p}_19']

delta_cols = [f'delta_{p}' for p in PARTIES]
model_data = merged.dropna(subset=delta_cols + ['Region name_24']).copy()

for col in delta_cols:
    m, s = model_data[col].mean(), model_data[col].std()
    model_data = model_data[model_data[col].between(m - 4*s, m + 4*s)]

regions = sorted(model_data['Region name_24'].unique())
region_map = {r: i for i, r in enumerate(regions)}
model_data['region_idx'] = model_data['Region name_24'].map(region_map)

fb_merged = model_data.merge(
    foreign_born[['ONSConstID_y', 'pct_point_change', 'foreign_pc_2021']],
    left_on='2024_code', right_on='ONSConstID_y', how='left'
)

fb_change = fb_merged['pct_point_change'].values.copy()
fb_mean = np.nanmean(fb_change)
fb_std = np.nanstd(fb_change)
fb_change_std = np.nan_to_num((fb_change - fb_mean) / fb_std, nan=0.0)
has_fb_mask = fb_merged['pct_point_change'].notna().values.astype(float)

Y = model_data[delta_cols].values
region_idx = model_data['region_idx'].values
n_regions = len(regions)
n_parties = len(PARTIES)

import pymc as pm

with pm.Model() as hierarchical_model:
    mu = pm.Normal('mu', mu=0, sigma=3, shape=n_parties)
    tau = pm.HalfCauchy('tau', beta=1.5, shape=n_parties)
    sigma = pm.HalfCauchy('sigma', beta=1.5, shape=n_parties)
    gamma = pm.Normal('gamma', mu=mu, sigma=tau, shape=(n_regions, n_parties))
    beta_fb = pm.Normal('beta_fb', mu=0, sigma=1, shape=n_parties)
    region_effect = gamma[region_idx]
    fb_effect = beta_fb[None, :] * (fb_change_std[:, None] * has_fb_mask[:, None])
    expected = region_effect + fb_effect
    obs = pm.Normal('obs', mu=expected, sigma=sigma, observed=Y)
    trace = pm.sample(2000, tune=1500, chains=2, cores=1,
                      random_seed=42, progressbar=True, target_accept=0.9)

print("Sampling complete. Extracting posterior draws...")

# ── 3. Extract all posterior draws ────────────────────────────────────────────
post = trace.posterior
mu_draws = post['mu'].values.reshape(-1, n_parties)         # (N_draws, 4)
gamma_draws = post['gamma'].values.reshape(-1, n_regions, n_parties)  # (N_draws, 11, 4)
sigma_draws = post['sigma'].values.reshape(-1, n_parties)   # (N_draws, 4)
beta_fb_draws = post['beta_fb'].values.reshape(-1, n_parties)  # (N_draws, 4)

N_DRAWS = mu_draws.shape[0]
print(f"Total posterior draws: {N_DRAWS}")

# ── 4. Prepare constituency data for projection ──────────────────────────────
NAT2024 = {"Lab": 34.64, "Con": 24.36, "RUK": 14.69, "LD": 12.56, "Green": 6.91}
POLLS = {"Lab": 15.72, "Con": 18.13, "RUK": 25.54, "LD": 10.24, "Green": 18.63}
SNP_POLLS = 34.95
SNP_2024 = 30.01
PC_POLLS = 21.85
PC_2024 = 14.77

# Polling uncertainty: correlated perturbations
# Based on UK polling error literature: ~2.5pp sd per party, with correlation
# Parties that compete for similar voters have positive correlation
# Using a simplified correlation structure
POLL_SD = 2.5  # pp standard deviation per party
POLL_CORR = np.array([
    #        Lab   Con   RUK   LD    Green  SNP   PC
    [  1.0, -0.3, -0.2, -0.1, -0.1,  0.0,  0.0],  # Lab
    [ -0.3,  1.0, -0.4,  0.1,  0.0,  0.0,  0.0],  # Con
    [ -0.2, -0.4,  1.0, -0.1, -0.1,  0.0,  0.0],  # RUK
    [ -0.1,  0.1, -0.1,  1.0, -0.1,  0.0,  0.0],  # LD
    [ -0.1,  0.0, -0.1, -0.1,  1.0,  0.0,  0.0],  # Green
    [  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],  # SNP
    [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0],  # PC
])
POLL_SDS = np.array([POLL_SD]*5 + [3.0, 4.0])  # Slightly higher for SNP/PC (fewer polls)
POLL_COV = np.outer(POLL_SDS, POLL_SDS) * POLL_CORR
PARTY_ORDER = ['Lab', 'Con', 'RUK', 'LD', 'Green', 'SNP', 'PC']

# Generate poll perturbations for each draw
poll_perturbations = np.random.multivariate_normal(
    np.zeros(7), POLL_COV, size=N_DRAWS
)

# Build constituency arrays for vectorised projection
gb_seats = results_2024[results_2024['Country name'].isin(['England', 'Scotland', 'Wales'])].copy()
ni_seats = results_2024[results_2024['Country name'] == 'Northern Ireland'].copy()

# Merge FB data
fb_lookup = {}
for _, row in foreign_born.iterrows():
    cid = row['ONSConstID_y']
    if pd.notna(cid) and pd.notna(row['pct_point_change']):
        fb_lookup[cid] = (row['pct_point_change'] - fb_mean) / fb_std

n_gb = len(gb_seats)
n_ni = len(ni_seats)
n_total = n_gb + n_ni

print(f"Projecting {n_gb} GB + {n_ni} NI = {n_total} constituencies across {N_DRAWS} draws...")

# Pre-compute arrays for GB seats
seat_ids = gb_seats['ONS ID'].values
seat_names = gb_seats['Constituency name'].values
seat_regions = gb_seats['Region name'].values
seat_countries = gb_seats['Country name'].values
seat_valid = gb_seats['Valid votes'].values.astype(float)

# Vote shares (n_gb, n_parties_all)
ALL_PARTIES = ['Lab', 'Con', 'RUK', 'LD', 'Green', 'SNP', 'PC',
               'DUP', 'SF', 'SDLP', 'UUP', 'APNI']
shares_2024 = np.zeros((n_gb, len(ALL_PARTIES)))
for j, p in enumerate(ALL_PARTIES):
    vals = gb_seats[p].fillna(0).values.astype(float)
    shares_2024[:, j] = vals / seat_valid * 100

# Region indices for GB seats
seat_region_idx = np.array([region_map.get(r, -1) for r in seat_regions])

# FB data
seat_fb = np.array([fb_lookup.get(cid, 0.0) for cid in seat_ids])
seat_has_fb = np.array([1.0 if cid in fb_lookup else 0.0 for cid in seat_ids])

# Scotland/Wales masks
is_scotland = (seat_countries == 'Scotland')
is_wales = (seat_countries == 'Wales')

# Party index mapping for PARTIES = ['Con', 'RUK', 'LD', 'Green']
# In ALL_PARTIES: Con=1, RUK=2, LD=3, Green=4, Lab=0, SNP=5, PC=6
PARTY_IDX = {'Lab': 0, 'Con': 1, 'RUK': 2, 'LD': 3, 'Green': 4, 'SNP': 5, 'PC': 6}
NI_PARTIES = ['DUP', 'SF', 'SDLP', 'UUP', 'APNI']

# NI shares (held constant)
ni_shares = np.zeros((n_ni, len(ALL_PARTIES)))
ni_valid = ni_seats['Valid votes'].values.astype(float)
for j, p in enumerate(ALL_PARTIES):
    vals = ni_seats[p].fillna(0).values.astype(float)
    ni_shares[:, j] = vals / ni_valid * 100

ni_ids = ni_seats['ONS ID'].values
ni_names = ni_seats['Constituency name'].values

# ── 5. Run posterior predictive simulation ────────────────────────────────────
# For each draw: perturb polls, compute projections, determine winners

# Storage: seat counts per party per draw
seat_count_draws = np.zeros((N_DRAWS, len(ALL_PARTIES)))  # (N_DRAWS, 12)

# Storage: win counts per constituency per party (for win probabilities)
gb_win_counts = np.zeros((n_gb, len(ALL_PARTIES)), dtype=int)
ni_winners = np.zeros((n_ni, len(ALL_PARTIES)), dtype=int)

# Determine NI winners once (held constant)
for i in range(n_ni):
    winner_idx = np.argmax(ni_shares[i])
    ni_winners[i, winner_idx] = N_DRAWS  # Wins all draws

print("Running simulation...")
report_interval = N_DRAWS // 10

for d in range(N_DRAWS):
    if d % report_interval == 0:
        print(f"  Draw {d}/{N_DRAWS}...")

    # Perturbed poll shares for this draw
    perturbation = poll_perturbations[d]
    polls_d = {
        'Lab': max(POLLS['Lab'] + perturbation[0], 1.0),
        'Con': max(POLLS['Con'] + perturbation[1], 1.0),
        'RUK': max(POLLS['RUK'] + perturbation[2], 1.0),
        'LD':  max(POLLS['LD']  + perturbation[3], 1.0),
        'Green': max(POLLS['Green'] + perturbation[4], 1.0),
    }
    snp_d = max(SNP_POLLS + perturbation[5], 1.0)
    pc_d = max(PC_POLLS + perturbation[6], 1.0)

    # Model parameters for this draw
    mu_d = mu_draws[d]          # (4,)
    gamma_d = gamma_draws[d]    # (11, 4)
    sigma_d = sigma_draws[d]    # (4,)
    beta_fb_d = beta_fb_draws[d]  # (4,)

    # National log-ratio deltas (Lab is reference)
    lab_poll = max(polls_d['Lab'], FLOOR)
    lab_base = max(NAT2024['Lab'], FLOOR)
    nat_delta = np.zeros(4)  # Con, RUK, LD, Green
    for pi, p in enumerate(PARTIES):
        p_poll = max(polls_d[p], FLOOR)
        p_base = max(NAT2024[p], FLOOR)
        nat_delta[pi] = np.log(p_poll / lab_poll) - np.log(p_base / lab_base)

    # Project each GB constituency
    projected_shares = shares_2024.copy()  # (n_gb, 12)

    # Compute log-ratio swings for each party
    lab_shares = np.maximum(shares_2024[:, 0], FLOOR)  # Lab

    for pi, p in enumerate(PARTIES):
        p_col = PARTY_IDX[p]
        p_shares = np.maximum(shares_2024[:, p_col], FLOOR)
        eta_2024 = np.log(p_shares / lab_shares)

        # Regional deviation
        regional_dev = np.zeros(n_gb)
        for ri in range(n_regions):
            mask = (seat_region_idx == ri)
            regional_dev[mask] = gamma_d[ri, pi] - mu_d[pi]

        # FB effect
        fb_eff = beta_fb_d[pi] * seat_fb * seat_has_fb

        # Constituency-level noise (this is the key MCMC addition)
        noise = np.random.normal(0, sigma_d[pi], n_gb)

        # Total delta
        delta = nat_delta[pi] + regional_dev + fb_eff + noise
        # Store new eta for later conversion
        projected_shares[:, p_col] = eta_2024 + delta  # temporarily store eta

    # SNP: proportional swing with noise
    snp_ratio = max(snp_d, 0.5) / max(SNP_2024, 0.5)
    snp_noise = np.random.normal(0, 0.05, n_gb)  # ~5% multiplicative noise
    projected_shares[is_scotland, 5] = shares_2024[is_scotland, 5] * snp_ratio * (1 + snp_noise[is_scotland])
    projected_shares[~is_scotland, 5] = shares_2024[~is_scotland, 5]

    # PC: proportional swing with noise
    pc_ratio = max(pc_d, 0.5) / max(PC_2024, 0.5)
    pc_noise = np.random.normal(0, 0.05, n_gb)
    projected_shares[is_wales, 6] = shares_2024[is_wales, 6] * pc_ratio * (1 + pc_noise[is_wales])
    projected_shares[~is_wales, 6] = shares_2024[~is_wales, 6]

    # NI parties: unchanged (indices 7-11)
    # Already copied from shares_2024

    # Convert log-ratios back to shares for Con, RUK, LD, Green
    # projected_shares[:, 1:5] currently hold eta values
    exp_etas = np.exp(projected_shares[:, 1:5])  # (n_gb, 4)
    sum_exp = 1.0 + exp_etas.sum(axis=1)  # (n_gb,)

    # Other shares (SNP, PC, NI parties, etc.)
    other = projected_shares[:, 5:].sum(axis=1)  # SNP + PC + NI parties
    remaining = np.maximum(100.0 - other, 1.0)

    lab_new = np.maximum(remaining / sum_exp, 0.1)
    projected_shares[:, 0] = lab_new  # Lab
    for pi, p in enumerate(PARTIES):
        p_col = PARTY_IDX[p]
        projected_shares[:, p_col] = lab_new * exp_etas[:, pi]

    # Ensure non-negative
    projected_shares = np.maximum(projected_shares, 0)

    # Renormalize
    orig_totals = shares_2024.sum(axis=1)
    new_totals = projected_shares.sum(axis=1)
    scale = np.where(new_totals > 0, orig_totals / new_totals, 1.0)
    projected_shares *= scale[:, None]

    # Determine winners
    winners = np.argmax(projected_shares, axis=1)  # (n_gb,)
    for i in range(n_gb):
        gb_win_counts[i, winners[i]] += 1

    # Count seats per party
    for pi in range(len(ALL_PARTIES)):
        seat_count_draws[d, pi] = np.sum(winners == pi)

    # Add NI seats
    for i in range(n_ni):
        ni_winner = np.argmax(ni_shares[i])
        seat_count_draws[d, ni_winner] += 1

print("Simulation complete!")

# ── 6. Compute summary statistics ────────────────────────────────────────────
print("\n=== Seat Count Summary (with 95% credible intervals) ===")
seat_summary = {}
for pi, p in enumerate(ALL_PARTIES):
    counts = seat_count_draws[:, pi]
    if counts.max() > 0:
        median = np.median(counts)
        mean = np.mean(counts)
        ci_lo = np.percentile(counts, 2.5)
        ci_hi = np.percentile(counts, 97.5)
        ci_10 = np.percentile(counts, 10)
        ci_90 = np.percentile(counts, 90)
        seat_summary[p] = {
            'median': float(median),
            'mean': float(mean),
            'ci95_lo': float(ci_lo),
            'ci95_hi': float(ci_hi),
            'ci80_lo': float(ci_10),
            'ci80_hi': float(ci_90),
            'sd': float(np.std(counts))
        }
        print(f"  {p:>6s}: {median:5.0f}  (95% CI: {ci_lo:.0f}–{ci_hi:.0f})  (80% CI: {ci_10:.0f}–{ci_90:.0f})  sd={np.std(counts):.1f}")

# Majority probability
largest_per_draw = seat_count_draws.max(axis=1)
majority_party_per_draw = seat_count_draws.argmax(axis=1)
prob_majority = np.mean(largest_per_draw >= 326)
prob_hung = 1 - prob_majority

# Who gets majority when there is one?
majority_draws = largest_per_draw >= 326
if majority_draws.any():
    maj_parties = majority_party_per_draw[majority_draws]
    for pi, p in enumerate(ALL_PARTIES):
        prob = np.mean(maj_parties == pi)
        if prob > 0:
            print(f"  P({p} majority) = {prob:.1%}")
print(f"  P(hung parliament) = {prob_hung:.1%}")

# ── 7. Compute constituency win probabilities ────────────────────────────────
print("\n=== Constituency Win Probabilities ===")
win_probs = {}  # {constituency_id: {party: probability}}

for i in range(n_gb):
    cid = seat_ids[i]
    probs = {}
    for pi, p in enumerate(ALL_PARTIES):
        prob = gb_win_counts[i, pi] / N_DRAWS
        if prob > 0.001:
            probs[p] = round(prob, 3)
    win_probs[cid] = probs

for i in range(n_ni):
    cid = ni_ids[i]
    winner_idx = np.argmax(ni_shares[i])
    win_probs[cid] = {ALL_PARTIES[winner_idx]: 1.0}

# Show some interesting marginals
print("\nMost uncertain constituencies (highest entropy):")
uncertainties = []
for cid, probs in win_probs.items():
    vals = np.array(list(probs.values()))
    entropy = -np.sum(vals * np.log(vals + 1e-10))
    name = ""
    idx = np.where(seat_ids == cid)[0]
    if len(idx) > 0:
        name = seat_names[idx[0]]
    elif cid in ni_ids:
        idx2 = np.where(ni_ids == cid)[0]
        if len(idx2) > 0:
            name = ni_names[idx2[0]]
    uncertainties.append((cid, name, entropy, probs))

uncertainties.sort(key=lambda x: -x[2])
for cid, name, ent, probs in uncertainties[:15]:
    prob_str = ', '.join(f"{p}:{v:.0%}" for p, v in sorted(probs.items(), key=lambda x: -x[1]))
    print(f"  {name:40s}  {prob_str}")

# ── 8. Export for dashboard ───────────────────────────────────────────────────
output = {
    'seat_summary': seat_summary,
    'prob_majority': float(prob_majority),
    'prob_hung': float(prob_hung),
    'win_probs': win_probs,
    'seat_distributions': {},
    'majority_probs': {}
}

# Seat distributions (histogram data for chart)
for pi, p in enumerate(ALL_PARTIES):
    counts = seat_count_draws[:, pi]
    if counts.max() > 0:
        hist_min = int(max(0, counts.min() - 5))
        hist_max = int(counts.max() + 5)
        bins = np.arange(hist_min, hist_max + 1)
        hist, _ = np.histogram(counts, bins=bins)
        output['seat_distributions'][p] = {
            'bins': bins[:-1].tolist(),
            'counts': hist.tolist()
        }

# Majority probabilities per party
for pi, p in enumerate(ALL_PARTIES):
    if majority_draws.any():
        prob = float(np.mean(majority_party_per_draw[majority_draws] == pi))
        if prob > 0:
            output['majority_probs'][p] = prob

with open('/sessions/sweet-amazing-dijkstra/posterior_predictive.json', 'w') as f:
    json.dump(output, f)

print(f"\nExported posterior predictive results ({len(win_probs)} constituencies)")
print("Done!")
