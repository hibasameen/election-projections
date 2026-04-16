"""
Posterior Predictive Simulation v2 for UK Election Projections

Improvements over v1:
  1. SNP and PC projected via hierarchical ALR model (not proportional swing)
  2. Empirical polling error covariance from historical misses + LOWESS residuals
  3. Cross-validated σ/τ estimates (uses model_params_v2.json)
  4. Full MCMC posterior with 6-party model (Con, RUK, LD, Green, SNP, PC)

Uses the full MCMC posterior to generate:
  - Seat count distributions per party (with credible intervals)
  - Constituency-level win probabilities
  - Polling uncertainty on top of model parameter uncertainty
"""

import pandas as pd
import numpy as np
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

with open(BASE + 'model_params_v2.json') as f:
    params = json.load(f)

print("Loaded model_params_v2.json")
print(f"  Parties: {params['parties']}")
print(f"  Polling error SDs: {[f'{p}={s:.2f}' for p, s in zip(params['poll_cov']['parties'], params['poll_cov']['sds'])]}")

# ── 2. Re-fit model to extract full posterior draws ─────────────────────────
print("\nRe-fitting Bayesian hierarchical model v2 for posterior draws...")

# Prepare data (same pipeline as bayesian_hierarchical_model_v2.py)
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

PARTIES_GB = ['Con', 'RUK', 'LD', 'Green']
PARTIES_SC = ['SNP']
PARTIES_WA = ['PC']
ALL_MODEL_PARTIES = PARTIES_GB + PARTIES_SC + PARTIES_WA
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

for p in ALL_MODEL_PARTIES:
    s19 = np.maximum(merged[f'share_{p}_19'].fillna(0).values, FLOOR)
    lab19 = np.maximum(merged['share_Lab_19'].fillna(0).values, FLOOR)
    s24 = np.maximum(merged[f'share_{p}_24'].fillna(0).values, FLOOR)
    lab24 = np.maximum(merged['share_Lab_24'].fillna(0).values, FLOOR)
    merged[f'eta_{p}_19'] = np.log(s19 / lab19)
    merged[f'eta_{p}_24'] = np.log(s24 / lab24)
    merged[f'delta_{p}'] = merged[f'eta_{p}_24'] - merged[f'eta_{p}_19']

delta_cols_gb = [f'delta_{p}' for p in PARTIES_GB]
model_data = merged.dropna(subset=delta_cols_gb + ['Region name_24']).copy()
for col in delta_cols_gb:
    m, s = model_data[col].mean(), model_data[col].std()
    model_data = model_data[model_data[col].between(m - 4*s, m + 4*s)]

regions = sorted(model_data['Region name_24'].unique())
region_map = {r: i for i, r in enumerate(regions)}
model_data['region_idx'] = model_data['Region name_24'].map(region_map)

is_scotland = model_data['Country name_24'] == 'Scotland'
is_wales = model_data['Country name_24'] == 'Wales'

fb_merged = model_data.merge(
    foreign_born[['ONSConstID_y', 'pct_point_change', 'foreign_pc_2021']],
    left_on='2024_code', right_on='ONSConstID_y', how='left'
)
fb_change = fb_merged['pct_point_change'].values.copy()
fb_mean = np.nanmean(fb_change)
fb_std = np.nanstd(fb_change)
fb_change_std = np.nan_to_num((fb_change - fb_mean) / fb_std, nan=0.0)
has_fb_mask = fb_merged['pct_point_change'].notna().values.astype(float)

Y_gb = model_data[delta_cols_gb].values
region_idx = model_data['region_idx'].values
snp_deltas = model_data['delta_SNP'].values
snp_mask = is_scotland.values.astype(float)
sc_idx = np.where(is_scotland.values)[0]
snp_deltas_sc = snp_deltas[sc_idx]

pc_deltas = model_data['delta_PC'].values
pc_mask = is_wales.values.astype(float)
wa_idx = np.where(is_wales.values)[0]
pc_deltas_wa = pc_deltas[wa_idx]

n_regions = len(regions)
n_parties_gb = len(PARTIES_GB)
n_parties_all = len(ALL_MODEL_PARTIES)

import pymc as pm

with pm.Model() as hierarchical_model_v2:
    mu = pm.Normal('mu', mu=0, sigma=3, shape=n_parties_all)
    # τ estimated freely only for 4 GB parties; SNP/PC share mean(τ_GB)
    # because each has only one region of data (between-region variance unidentified)
    tau_gb = pm.HalfCauchy('tau_gb', beta=1.5, shape=n_parties_gb)
    tau_shared = pm.math.mean(tau_gb)
    tau = pm.math.concatenate([tau_gb,
                               pm.math.stack([tau_shared, tau_shared])])
    pm.Deterministic('tau', tau)
    sigma = pm.HalfCauchy('sigma', beta=1.5, shape=n_parties_all)
    gamma = pm.Normal('gamma', mu=mu, sigma=tau, shape=(n_regions, n_parties_all))
    beta_fb = pm.Normal('beta_fb', mu=0, sigma=1, shape=n_parties_all)

    region_effect = gamma[region_idx]
    fb_effect = beta_fb[None, :] * (fb_change_std[:, None] * has_fb_mask[:, None])
    expected_all = region_effect + fb_effect

    obs_gb = pm.Normal('obs_gb',
                       mu=expected_all[:, :n_parties_gb],
                       sigma=sigma[:n_parties_gb],
                       observed=Y_gb)

    snp_expected = expected_all[sc_idx, 4]
    obs_snp = pm.Normal('obs_snp', mu=snp_expected, sigma=sigma[4],
                        observed=snp_deltas_sc)

    pc_expected = expected_all[wa_idx, 5]
    obs_pc = pm.Normal('obs_pc', mu=pc_expected, sigma=sigma[5],
                       observed=pc_deltas_wa)

    trace = pm.sample(2000, tune=1500, chains=2, cores=1,
                      random_seed=42, progressbar=True, target_accept=0.9)

print("Sampling complete. Extracting posterior draws...")

# ── 3. Extract all posterior draws ────────────────────────────────────────────
post = trace.posterior
mu_draws = post['mu'].values.reshape(-1, n_parties_all)         # (N_draws, 6)
gamma_draws = post['gamma'].values.reshape(-1, n_regions, n_parties_all)
sigma_draws = post['sigma'].values.reshape(-1, n_parties_all)
beta_fb_draws = post['beta_fb'].values.reshape(-1, n_parties_all)

N_DRAWS = mu_draws.shape[0]
print(f"Total posterior draws: {N_DRAWS}")

# ── 4. Load empirical polling covariance ──────────────────────────────────────
POLL_COV = np.array(params['poll_cov']['matrix'])
POLL_PARTY_ORDER = params['poll_cov']['parties']
print(f"\nPolling error covariance loaded ({len(POLL_PARTY_ORDER)} parties)")
print(f"  SDs: {[f'{p}={s:.2f}' for p, s in zip(POLL_PARTY_ORDER, np.sqrt(np.diag(POLL_COV)))]}")

# ── 5. Prepare constituency data ────────────────────────────────────────────
NAT2024 = {"Lab": 34.64, "Con": 24.36, "RUK": 14.69, "LD": 12.56, "Green": 6.91}
POLLS = {"Lab": 17.08, "Con": 17.28, "RUK": 25.70, "LD": 11.60, "Green": 16.78}
SNP_POLLS = 32.53
SNP_2024 = 30.01
PC_POLLS = 19.20
PC_2024 = 14.77

# Generate poll perturbations using empirical covariance
poll_perturbations = np.random.multivariate_normal(
    np.zeros(len(POLL_PARTY_ORDER)), POLL_COV, size=N_DRAWS
)

gb_seats = results_2024[results_2024['Country name'].isin(['England', 'Scotland', 'Wales'])].copy()
ni_seats = results_2024[results_2024['Country name'] == 'Northern Ireland'].copy()

# FB lookup
fb_lookup = {}
for _, row in foreign_born.iterrows():
    cid = row['ONSConstID_y']
    if pd.notna(cid) and pd.notna(row['pct_point_change']):
        fb_lookup[cid] = (row['pct_point_change'] - fb_mean) / fb_std

n_gb = len(gb_seats)
n_ni = len(ni_seats)
n_total = n_gb + n_ni

print(f"Projecting {n_gb} GB + {n_ni} NI = {n_total} constituencies across {N_DRAWS} draws...")

# Pre-compute arrays
seat_ids = gb_seats['ONS ID'].values
seat_names = gb_seats['Constituency name'].values
seat_regions = gb_seats['Region name'].values
seat_countries = gb_seats['Country name'].values
seat_valid = gb_seats['Valid votes'].values.astype(float)

ALL_PARTIES = ['Lab', 'Con', 'RUK', 'LD', 'Green', 'SNP', 'PC',
               'DUP', 'SF', 'SDLP', 'UUP', 'APNI']
shares_2024 = np.zeros((n_gb, len(ALL_PARTIES)))
for j, p in enumerate(ALL_PARTIES):
    vals = gb_seats[p].fillna(0).values.astype(float)
    shares_2024[:, j] = vals / seat_valid * 100

seat_region_idx = np.array([region_map.get(r, -1) for r in seat_regions])
seat_fb = np.array([fb_lookup.get(cid, 0.0) for cid in seat_ids])
seat_has_fb = np.array([1.0 if cid in fb_lookup else 0.0 for cid in seat_ids])

is_scotland_seat = (seat_countries == 'Scotland')
is_wales_seat = (seat_countries == 'Wales')

PARTY_IDX = {'Lab': 0, 'Con': 1, 'RUK': 2, 'LD': 3, 'Green': 4, 'SNP': 5, 'PC': 6}

# NI
ni_shares = np.zeros((n_ni, len(ALL_PARTIES)))
ni_valid = ni_seats['Valid votes'].values.astype(float)
for j, p in enumerate(ALL_PARTIES):
    vals = ni_seats[p].fillna(0).values.astype(float)
    ni_shares[:, j] = vals / ni_valid * 100
ni_ids = ni_seats['ONS ID'].values
ni_names = ni_seats['Constituency name'].values

# ── 6. Posterior predictive simulation ────────────────────────────────────────
seat_count_draws = np.zeros((N_DRAWS, len(ALL_PARTIES)))
gb_win_counts = np.zeros((n_gb, len(ALL_PARTIES)), dtype=int)
ni_winners = np.zeros((n_ni, len(ALL_PARTIES)), dtype=int)

for i in range(n_ni):
    winner_idx = np.argmax(ni_shares[i])
    ni_winners[i, winner_idx] = N_DRAWS

print("Running simulation...")
report_interval = max(1, N_DRAWS // 10)

# Model party indices in ALL_MODEL_PARTIES: Con=0, RUK=1, LD=2, Green=3, SNP=4, PC=5
# In ALL_PARTIES: Lab=0, Con=1, RUK=2, LD=3, Green=4, SNP=5, PC=6

for d in range(N_DRAWS):
    if d % report_interval == 0:
        print(f"  Draw {d}/{N_DRAWS}...")

    # Perturbed polls (using empirical covariance)
    perturbation = poll_perturbations[d]
    # POLL_PARTY_ORDER = ['Lab', 'Con', 'RUK', 'LD', 'Green', 'SNP', 'PC']
    polls_d = {
        'Lab':   max(POLLS['Lab']   + perturbation[0], 1.0),
        'Con':   max(POLLS['Con']   + perturbation[1], 1.0),
        'RUK':   max(POLLS['RUK']   + perturbation[2], 1.0),
        'LD':    max(POLLS['LD']    + perturbation[3], 1.0),
        'Green': max(POLLS['Green'] + perturbation[4], 1.0),
    }
    snp_d = max(SNP_POLLS + perturbation[5], 1.0)
    pc_d = max(PC_POLLS + perturbation[6], 1.0)

    # Posterior draws for this iteration
    mu_d = mu_draws[d]
    gamma_d = gamma_draws[d]
    sigma_d = sigma_draws[d]
    beta_fb_d = beta_fb_draws[d]

    # National log-ratio deltas (Lab reference)
    lab_poll = max(polls_d['Lab'], FLOOR)
    lab_base = max(NAT2024['Lab'], FLOOR)

    # GB parties: national delta
    nat_delta_gb = np.zeros(n_parties_gb)
    for pi, p in enumerate(PARTIES_GB):
        p_poll = max(polls_d[p], FLOOR)
        p_base = max(NAT2024[p], FLOOR)
        nat_delta_gb[pi] = np.log(p_poll / lab_poll) - np.log(p_base / lab_base)

    # SNP national delta (Scotland-specific)
    nat_delta_snp = np.log(max(snp_d, FLOOR) / lab_poll) - np.log(max(SNP_2024, FLOOR) / lab_base)
    # PC national delta (Wales-specific)
    nat_delta_pc = np.log(max(pc_d, FLOOR) / lab_poll) - np.log(max(PC_2024, FLOOR) / lab_base)

    # Project each GB constituency
    projected_shares = shares_2024.copy()
    lab_shares = np.maximum(shares_2024[:, 0], FLOOR)

    # GB parties (Con, RUK, LD, Green)
    for pi, p in enumerate(PARTIES_GB):
        p_col = PARTY_IDX[p]
        p_shares = np.maximum(shares_2024[:, p_col], FLOOR)
        eta_2024 = np.log(p_shares / lab_shares)

        regional_dev = np.zeros(n_gb)
        for ri in range(n_regions):
            mask = (seat_region_idx == ri)
            regional_dev[mask] = gamma_d[ri, pi] - mu_d[pi]

        fb_eff = beta_fb_d[pi] * seat_fb * seat_has_fb
        noise = np.random.normal(0, sigma_d[pi], n_gb)
        delta = nat_delta_gb[pi] + regional_dev + fb_eff + noise
        projected_shares[:, p_col] = eta_2024 + delta  # store eta

    # SNP — now hierarchical for Scotland, unchanged elsewhere
    snp_col = PARTY_IDX['SNP']
    snp_shares_2024 = np.maximum(shares_2024[:, snp_col], FLOOR)
    snp_eta_2024 = np.log(snp_shares_2024 / lab_shares)

    # Scotland: hierarchical ALR projection
    snp_regional_dev = np.zeros(n_gb)
    for ri in range(n_regions):
        mask = (seat_region_idx == ri) & is_scotland_seat
        if mask.any():
            snp_regional_dev[mask] = gamma_d[ri, 4] - mu_d[4]  # index 4 = SNP

    snp_fb_eff = beta_fb_d[4] * seat_fb * seat_has_fb
    snp_noise = np.random.normal(0, sigma_d[4], n_gb)
    snp_delta = nat_delta_snp + snp_regional_dev + snp_fb_eff + snp_noise

    # Store eta for Scottish seats, keep original shares for non-Scottish
    projected_shares[is_scotland_seat, snp_col] = snp_eta_2024[is_scotland_seat] + snp_delta[is_scotland_seat]
    # Non-Scotland: keep as actual share (not log-ratio) — mark for special handling
    # We'll use a flag to distinguish
    snp_is_eta = is_scotland_seat.copy()

    # PC — hierarchical for Wales
    pc_col = PARTY_IDX['PC']
    pc_shares_2024 = np.maximum(shares_2024[:, pc_col], FLOOR)
    pc_eta_2024 = np.log(pc_shares_2024 / lab_shares)

    pc_regional_dev = np.zeros(n_gb)
    for ri in range(n_regions):
        mask = (seat_region_idx == ri) & is_wales_seat
        if mask.any():
            pc_regional_dev[mask] = gamma_d[ri, 5] - mu_d[5]  # index 5 = PC

    pc_fb_eff = beta_fb_d[5] * seat_fb * seat_has_fb
    pc_noise = np.random.normal(0, sigma_d[5], n_gb)
    pc_delta = nat_delta_pc + pc_regional_dev + pc_fb_eff + pc_noise

    projected_shares[is_wales_seat, pc_col] = pc_eta_2024[is_wales_seat] + pc_delta[is_wales_seat]
    pc_is_eta = is_wales_seat.copy()

    # Convert log-ratios back to shares
    # For each constituency, the ALR parties are in eta space, others are shares
    for c in range(n_gb):
        # Determine which parties are in ALR space for this constituency
        alr_cols = [1, 2, 3, 4]  # Con, RUK, LD, Green always
        if snp_is_eta[c]:
            alr_cols.append(5)  # SNP
        if pc_is_eta[c]:
            alr_cols.append(6)  # PC

        non_alr_share = 0.0
        # Non-ALR party shares (SNP if not Scotland, PC if not Wales, NI parties)
        for col_idx in range(len(ALL_PARTIES)):
            if col_idx == 0:  # Lab — solved from ALR
                continue
            if col_idx not in alr_cols:
                non_alr_share += max(projected_shares[c, col_idx], 0)

        exp_etas = np.exp(projected_shares[c, alr_cols])
        sum_exp = 1.0 + exp_etas.sum()
        remaining = max(100.0 - non_alr_share, 1.0)
        lab_new = max(remaining / sum_exp, 0.1)

        projected_shares[c, 0] = lab_new  # Lab
        for k, col_idx in enumerate(alr_cols):
            projected_shares[c, col_idx] = lab_new * exp_etas[k]

    # Ensure non-negative and renormalize
    projected_shares = np.maximum(projected_shares, 0)
    orig_totals = shares_2024.sum(axis=1)
    new_totals = projected_shares.sum(axis=1)
    scale = np.where(new_totals > 0, orig_totals / new_totals, 1.0)
    projected_shares *= scale[:, None]

    # Determine winners
    winners = np.argmax(projected_shares, axis=1)
    for i in range(n_gb):
        gb_win_counts[i, winners[i]] += 1

    for pi in range(len(ALL_PARTIES)):
        seat_count_draws[d, pi] = np.sum(winners == pi)

    # Add NI seats
    for i in range(n_ni):
        ni_winner = np.argmax(ni_shares[i])
        seat_count_draws[d, ni_winner] += 1

print("Simulation complete!")

# ── 7. Summary statistics ───────────────────────────────────────────────────
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
        print(f"  {p:>6s}: {median:5.0f}  "
              f"(95% CI: {ci_lo:.0f}–{ci_hi:.0f})  "
              f"(80% CI: {ci_10:.0f}–{ci_90:.0f})  "
              f"sd={np.std(counts):.1f}")

# Majority probabilities
largest_per_draw = seat_count_draws.max(axis=1)
majority_party_per_draw = seat_count_draws.argmax(axis=1)
prob_majority = np.mean(largest_per_draw >= 326)
prob_hung = 1 - prob_majority

majority_draws = largest_per_draw >= 326
if majority_draws.any():
    maj_parties = majority_party_per_draw[majority_draws]
    for pi, p in enumerate(ALL_PARTIES):
        prob = np.mean(maj_parties == pi)
        if prob > 0:
            print(f"  P({p} majority) = {prob:.1%}")
print(f"  P(hung parliament) = {prob_hung:.1%}")

# ── 8. Constituency win probabilities ───────────────────────────────────────
print("\n=== Constituency Win Probabilities ===")
win_probs = {}

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

# Most uncertain constituencies
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
for cid, name, ent, probs in uncertainties[:20]:
    prob_str = ', '.join(f"{p}:{v:.0%}" for p, v in sorted(probs.items(), key=lambda x: -x[1]))
    print(f"  {name:40s}  {prob_str}")

# ── 9. Export ───────────────────────────────────────────────────────────────
output = {
    'version': 2,
    'seat_summary': seat_summary,
    'prob_majority': float(prob_majority),
    'prob_hung': float(prob_hung),
    'win_probs': win_probs,
    'seat_distributions': {},
    'majority_probs': {},
    'model_info': {
        'n_draws': N_DRAWS,
        'parties_modelled': ALL_MODEL_PARTIES,
        'poll_cov_source': 'empirical (historical + LOWESS residuals)',
        'snp_pc_method': 'hierarchical ALR (not proportional swing)',
    }
}

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

for pi, p in enumerate(ALL_PARTIES):
    if majority_draws.any():
        prob = float(np.mean(majority_party_per_draw[majority_draws] == pi))
        if prob > 0:
            output['majority_probs'][p] = prob

with open(BASE + 'posterior_predictive_v2.json', 'w') as f:
    json.dump(output, f)

print(f"\nExported posterior predictive v2 results ({len(win_probs)} constituencies)")
print("Done!")
