#!/usr/bin/env python3
import json
import math
import numpy as np

def normal_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def welch_t_test(a, b):
    n1, n2 = len(a), len(b)
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se > 0 else 0.0
    df = (v1 / n1 + v2 / n2)**2 / ((v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1))
    
    def t_pdf(t, d):
        return (math.gamma((d + 1) / 2) / (math.sqrt(d * math.pi) * math.gamma(d / 2))) * (1 + t**2 / d)**(-(d + 1) / 2)
    
    abs_t = abs(t_stat)
    steps = 2000
    upper = max(abs_t + 10.0, 35.0)
    dt = (upper - abs_t) / steps
    tail = sum(t_pdf(abs_t + (i + 0.5) * dt, df) * dt for i in range(steps))
    p_val = min(1.0, 2.0 * tail)
    
    # 95% CI critical value
    low_t, high_t = 1.5, 4.0
    for _ in range(60):
        mid_t = (low_t + high_t) / 2.0
        s_tail = sum(t_pdf(mid_t + (i + 0.5) * ((upper - mid_t) / steps), df) * ((upper - mid_t) / steps) for i in range(steps))
        if s_tail > 0.025:
            low_t = mid_t
        else:
            high_t = mid_t
    t_crit = (low_t + high_t) / 2.0
    
    ci_low = (m1 - m2) - t_crit * se
    ci_high = (m1 - m2) + t_crit * se
    
    # Mann-Whitney U test
    combined = [(x, 1) for x in a] + [(y, 2) for y in b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(ranks[idx] for idx, item in enumerate(combined) if item[1] == 1)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)
    mu_u = n1 * n2 / 2.0
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (abs(u_stat - mu_u) - 0.5) / sigma_u if sigma_u > 0 else 0.0
    mwu_pval = 2.0 * (1.0 - normal_cdf(z))
    
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohen_d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0
    
    return {
        'diff': m1 - m2,
        't_stat': t_stat,
        'df': df,
        'p_val': p_val,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'u_stat': u_stat,
        'mwu_pval': mwu_pval,
        'cohen_d': cohen_d,
        'm1': m1, 's1': math.sqrt(v1), 'n1': n1,
        'm2': m2, 's2': math.sqrt(v2), 'n2': n2
    }

def run_tests():
    # 1. Ablation tests
    with open('results/cxl_emulation_ablation_results.json') as f:
        abl = json.load(f)
    summary = abl['summary'] if 'summary' in abl else abl
    modes = {m['mode_id']: m['raw_throughputs'] for m in summary}

    print('=' * 95)
    print('STATISTICAL SIGNIFICANCE TESTING REPORT (INDEPENDENT EVALUATION)')
    print('=' * 95)

    def compute_comp(title, g1, g2, n1, n2):
        res = welch_t_test(g1, g2)
        res['title'] = title
        res['n1_label'] = n1
        res['n2_label'] = n2
        return res

    print('\n--- 1. CXL EMULATION ABLATION (N=20 TRIALS PER MODE, RANDOMIZED INTERLEAVED) ---')
    abl_comps = [
        compute_comp('Comparison 1: Mode B vs Mode C (Latency Only vs Latency + Bandwidth)', modes['Mode B'], modes['Mode C'], 'Mode B', 'Mode C'),
        compute_comp('Comparison 2: Mode A vs Mode B (Emulation Disabled vs Latency Only)', modes['Mode A'], modes['Mode B'], 'Mode A', 'Mode B'),
        compute_comp('Comparison 3: Mode A vs Mode C (Emulation Disabled vs Latency + Bandwidth)', modes['Mode A'], modes['Mode C'], 'Mode A', 'Mode C'),
    ]

    # Multiple testing correction for 3 pairwise comparisons
    k = len(abl_comps)
    alpha = 0.05
    bonf_alpha = alpha / k
    # Holm-Bonferroni: sort by p-value ascending
    sorted_comps = sorted(abl_comps, key=lambda c: c['p_val'])
    for rank, comp in enumerate(sorted_comps):
        holm_alpha = alpha / (k - rank)
        comp['holm_alpha'] = holm_alpha
        comp['bonf_alpha'] = bonf_alpha
        comp['holm_sig'] = comp['p_val'] < holm_alpha
        comp['bonf_sig'] = comp['p_val'] < bonf_alpha
        comp['p_adj_bonf'] = min(1.0, comp['p_val'] * k)
        comp['p_adj_holm'] = min(1.0, comp['p_val'] * (k - rank))

    for res in abl_comps:
        print(f'\n* {res["title"]}')
        print(f'  Group 1 ({res["n1_label"]}): Mean = {res["m1"]:.2f}, Std = {res["s1"]:.2f} (N={res["n1"]})')
        print(f'  Group 2 ({res["n2_label"]}): Mean = {res["m2"]:.2f}, Std = {res["s2"]:.2f} (N={res["n2"]})')
        print(f'  Mean Difference ({res["n1_label"]} - {res["n2_label"]}): {res["diff"]:+.3f} tok/s')
        print(f'  95% Confidence Interval for Difference: [{res["ci_low"]:+.3f}, {res["ci_high"]:+.3f}] tok/s')
        print(f'  Welch t-test: t({res["df"]:.1f}) = {res["t_stat"]:+.3f}, uncorrected p-value = {res["p_val"]:.4f}')
        print(f'  Mann-Whitney U: U = {res["u_stat"]:.1f}, uncorrected p-value = {res["mwu_pval"]:.4f}')
        print(f'  Effect Size (Cohen\'s d): {res["cohen_d"]:+.3f}')
        print(f'  Bonferroni Adjusted p-value (k={k}): p_adj = {res["p_adj_bonf"]:.4f} (threshold alpha = {bonf_alpha:.4f})')
        print(f'  Holm-Bonferroni Adjusted p-value: p_adj = {res["p_adj_holm"]:.4f}')
        if res['bonf_sig']:
            print('  Verdict: STATISTICALLY SIGNIFICANT after multiple comparisons correction')
        else:
            print('  Verdict: NOT STATISTICALLY SIGNIFICANT after FWER control -> Substantial Distribution Overlap')

    print('\n  SUMMARY VERDICT FOR 3-MODE ABLATION:')
    all_null = all(not c['bonf_sig'] for c in abl_comps)
    if all_null:
        print('  All pairwise comparisons yield p_adj > 0.05 under Bonferroni/Holm correction.')
        print('  Scientific Conclusion: Mode A, Mode B, and Mode C are statistically indistinguishable at N=20.')
        print('  Architectural Meaning: CPU activation offloading successfully insulates end-to-end throughput')
        print('  from CXL latency and bandwidth constraints under this workload.')

    # 2. Capacity sweep
    with open('results/cxl_capacity_sweep_results.json') as f:
        cap = json.load(f)
    scens = {s['scenario']: s['trial_throughputs'] for s in cap}

    print('\n--- 2. CXL CAPACITY SWEEP (N=5 TRIALS PER SCENARIO) ---')
    cap_comps = [
        compute_comp('Comparison 4: No CXL Spill vs Extreme CXL Spill', scens['No CXL Spill (Full DRAM)'], scens['Extreme CXL Spill'], 'No Spill', 'Extreme Spill'),
        compute_comp('Comparison 5: No CXL Spill vs Moderate CXL Spill', scens['No CXL Spill (Full DRAM)'], scens['Moderate CXL Spill'], 'No Spill', 'Moderate Spill'),
        compute_comp('Comparison 6: Mild CXL Spill vs Moderate CXL Spill', scens['Mild CXL Spill'], scens['Moderate CXL Spill'], 'Mild Spill', 'Moderate Spill'),
    ]

    for res in cap_comps:
        print(f'\n* {res["title"]}')
        print(f'  Group 1 ({res["n1_label"]}): Mean = {res["m1"]:.2f}, Std = {res["s1"]:.2f} (N={res["n1"]})')
        print(f'  Group 2 ({res["n2_label"]}): Mean = {res["m2"]:.2f}, Std = {res["s2"]:.2f} (N={res["n2"]})')
        print(f'  Mean Difference ({res["n1_label"]} - {res["n2_label"]}): {res["diff"]:+.3f} tok/s')
        print(f'  95% Confidence Interval for Difference: [{res["ci_low"]:+.3f}, {res["ci_high"]:+.3f}] tok/s')
        print(f'  Welch t-test: t({res["df"]:.1f}) = {res["t_stat"]:+.3f}, p-value = {res["p_val"]:.4f}')
        print(f'  Mann-Whitney U: U = {res["u_stat"]:.1f}, p-value = {res["mwu_pval"]:.4f}')
        print(f'  Effect Size (Cohen\'s d): {res["cohen_d"]:+.3f}')
        if res['p_val'] < 0.05:
            print('  Verdict: STATISTICALLY SIGNIFICANT (p < 0.05)')
        else:
            print('  Verdict: NOT STATISTICALLY SIGNIFICANT (p >= 0.05) -> Substantial Distribution Overlap')

if __name__ == '__main__':
    run_tests()
