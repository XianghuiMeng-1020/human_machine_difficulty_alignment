# Selection-bias interpretation: 944-item subset vs 27,613-item full EeDi pool (Part J)

- 944-item content-available subset n=944; remaining pool n=26669; full verified pool n=27613 (944 + 26669 = 27613).

## Label correction (write-time documentation only)

The values previously labeled `EB_difficulty(shrunk_rate)` (subset 0.5675, rest 0.6711) are
**EB posterior correctness** (`shrunk_rate` = alpha_post / (alpha_post + beta_post)), not
difficulty. No numeric values were recomputed. Difficulty is the complementary transform
`1 - shrunk_rate`; SDs are unchanged and the SMD sign flips.

## Full comparison table

| variable | mean_subset | mean_full_rest | sd_subset | sd_full_rest | n_subset | n_full_rest | SMD | direction |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| empirical_correctness | 0.566762 | 0.672021 | 0.134457 | 0.160757 | 944 | 26669 | -0.710289 | subset lower |
| EB_posterior_correctness(shrunk_rate) | 0.567495 | 0.671071 | 0.133212 | 0.153192 | 944 | 26669 | -0.721529 | subset lower |
| EB_difficulty(1-shrunk_rate) | 0.432505 | 0.328929 | 0.133212 | 0.153192 | 944 | 26669 | +0.721529 | subset higher |
| attempt_count(n_attempts) | 2021.95 | 575.269 | 1179.92 | 655.212 | 944 | 26669 | 1.5159 | subset higher |
| empirical_bucket_category=Human Easy | 0.0360169 | 0.245229 |  |  | 34 | 6540 |  | subset 3.6% vs rest 24.5% |
| empirical_bucket_category=Human Hard | 0.103814 | 0.0555326 |  |  | 98 | 1481 |  | subset 10.4% vs rest 5.6% |
| empirical_bucket_category=Human Mid | 0.860169 | 0.699239 |  |  | 812 | 18648 |  | subset 86.0% vs rest 69.9% |
| shrunk_bucket_category=Human Easy | 0.0349576 | 0.233267 |  |  | 33 | 6221 |  | subset 3.5% vs rest 23.3% |
| shrunk_bucket_category=Human Hard | 0.103814 | 0.0476958 |  |  | 98 | 1272 |  | subset 10.4% vs rest 4.8% |
| shrunk_bucket_category=Human Mid | 0.861229 | 0.719037 |  |  | 813 | 19176 |  | subset 86.1% vs rest 71.9% |

## Which variable gives max |SMD|

**attempt_count(n_attempts)**, SMD = 1.5159 (subset mean=2021.947, rest mean=575.269).

## Manuscript-safe conclusion (do not exceed this)

> The 944-item content-available subset is not representative of the full 27,613-question EeDi pool (large SMDs on attempt count and difficulty-related measures). The alignment effect estimated on the 944-item subset must NOT be transported/generalized to the full 27,613-item EeDi universe.
