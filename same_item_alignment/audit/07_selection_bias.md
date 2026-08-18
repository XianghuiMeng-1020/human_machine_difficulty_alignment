# Selection-bias audit: 944-item same-item subset vs remaining 27,613-item EeDi pool (Section 15)

- empirical_correctness (mean_correct): same-item mean=0.5668 (sd=0.1345, n=944) vs rest mean=0.6720 (sd=0.1608, n=26669); SMD=-0.7103
- eb_difficulty (1-shrunk_rate): same-item mean=0.5675 (sd=0.1332, n=944) vs rest mean=0.6711 (sd=0.1532, n=26669); SMD=-0.7215
- n_attempts: same-item mean=2021.9470 (sd=1179.9197, n=944) vs rest mean=575.2694 (sd=655.2121, n=26669); SMD=1.5159

Outcome-category proportions, same-item subset: {'Human Mid': 0.8601694915254238, 'Human Hard': 0.1038135593220339, 'Human Easy': 0.036016949152542374}
Outcome-category proportions, remaining pool: {'Human Mid': 0.6992388166035471, 'Human Easy': 0.24522854250253104, 'Human Hard': 0.055532640893921785}

Max |SMD| across compared variables = 1.5159
Representativeness verdict: the 944-item subset differs non-trivially from the remaining pool on at least one measured dimension (threshold |SMD|<0.10 for 'small'; do NOT claim full representativeness of all 27,613 items beyond what is shown here).
