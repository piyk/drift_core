# Pre-Specification of a Feature Revision: A Behavioral Basis for DRIFT

Manuscript T-IFS-27099-2026. Written on 17 November 2026 at 23:05 local time (UTC+7),
before any code for the revision existed and before any detector was fitted on a revised
feature set. The only computation made beforehand is the descriptive check of the raw
fields reported in the section "Observation behind change 1"; it used no detector, no
score, and no label. The analysis code will be `src/features_v2.py` (feature sets and
detectors), `src/run_features.py` (injected benchmarks), and `src/features_external.py`
(sanctioned and labeled addresses). Results go to `outputs/features/`. No existing result
file is changed, and the SHA-256 digest of this file is written into every result file.

## Purpose

The audit of the revision found that the scores of DRIFT and of its components track
wallet size: against real wallets matched in count, received value, and span, injected
wallets rank close to chance; real wallets with at least 1,000 transactions outrank the
injected wallets (ROC-AUC of DRIFT 0.410); 53% of the false positives of DRIFT at the 5%
budget have at least 100 transactions; and sanctioned P2WSH addresses and their controls
fall in the corpus top 10% in 98.0% and 93.6% of cases. Three properties of the 21
features plausibly contribute, and each is a construction flaw rather than a tuning
choice. First, the transaction rate and the burst index divide a lifetime count by a span
measured over at most 50 transactions, so for busy wallets the count enters three
features. Second, the 21 features are functions of six stored quantities, several of
which enter many times (the sent fraction enters seven features), and seven one-hot
columns of script type, which Table I says expose no typology, are a third of the inputs
from which the Isolation Forest draws its split features. Third, joint anomalies such as
a large value received in two transactions are extreme in no single feature, so the
Isolation Forest cannot isolate them.

The revision is not meant to raise the PR-AUC on G2 or G3. There the injected wallets
differ from the corpus mainly in size, and a ranking by received value alone is close to
the answer key; DRIFT could beat it only by becoming a size detector. The revision is
judged where size cannot explain the result: the size-matched AUC, the verified
high-activity services, the high-activity hard negatives, and typologies that no design
choice has seen. The primary method of the manuscript, its features, and its numbers stay
unchanged. The revision is reported as a supplementary analysis whether it meets its
criteria or not.

## Observation behind change 1

The scraper requested at most 50 transactions per address from every source and measured
the active span over the confirmed ones among them. The stored rate, however, depends on
the source. In the 9,325 records (13.7%) that came from BlockCypher, the rate times the
span equals the number of transaction references in the window, which can exceed the
transaction count because one transaction may hold several references (it does in 13% of
these records). In the other 58,552 records (Esplora or blockchain.info), the rate times
the span equals the lifetime transaction count. Of the 13,079 wallets with more than 50
transactions, 10,955 therefore carry a rate of lifetime count over window span, inflated
relative to the window rate by the factor n/50 (median 3.6, 90th percentile 32.6), and
2,110 carry a window rate. A rate defined this way is itself a source artifact. Every
revised feature is therefore computed from the stored count, received value, sent value,
and span only, never from the stored rate, and the window count is defined for every
record as w = min(n, 50), the number of transactions the scraper requested from every
source.

## Feature sets

Let n be the confirmed transaction count, r and s the received and sent values in
satoshi, a the active span in days over the window, and w = min(n, 50). Five feature sets
are computed by one pipeline for real and synthetic wallets alike.

| Set | Inputs | Role |
|---|---|---|
| F21 | the 21 features of the manuscript, unchanged | current features, recomputed in the same run |
| W21 | F21 with log_tx_per_day replaced by log(1 + w / max(a, 1)) and burst_index by log(1 + w / sqrt(a + 1)) | change 1 alone |
| B | log(1 + n); log(1 + r); sent fraction q = min(s / r, 1), with q = 0 when r = 0; log(1 + a); log(1 + w / max(a, 1)) | changes 1 and 2 (five inputs) |
| BR | B plus two size-relative features, rho_r = log(1 + r) - M_r(n) and rho_a = log(1 + a) - M_a(n) | changes 1, 2, and 3 (seven inputs); the primary revised set |
| BR-u | BR without winsorization | sensitivity analysis |

Script type and the unconfirmed ratio are not inputs of B, BR, or BR-u. The optional log
value per transaction is not included, because rho_r replaces it with a nonparametric
version of the same size-relative quantity.

M_r(n) and M_a(n) are the medians of log(1 + r) and of log(1 + a) among the reference
wallets in the count bin of n. The bins divide log(1 + n) at its 2nd, 4th, ..., 98th
percentiles in the reference, taken as observed values with duplicates removed; a wallet
below the first edge belongs to the first bin. Any bin with fewer than 50 reference
wallets is merged with the bin above it (the top bin with the bin below), repeatedly,
until no such bin remains. The bins and medians are estimated on the reference only (the
fit split for the injected benchmarks; the corpus, or the four training folds under
cross-fitting, for the external checks) and applied unchanged to every scored wallet, as
are the winsorization bounds (1st and 99th percentiles) and the standardization.

Excluded on purpose, because they would make the comparison unfair: indicators of the
generator ranges or of the heuristic's flags; wallet age, recency, or time since the last
activity, which would reproduce the observation-time artifact of the sanctioned check;
the four per-transaction value aggregates, which exist only for BlockCypher records; and
any feature chosen with the labels of the validation split, the test split, H, or H'.

## Detectors and the sizing rule

Every detector receives the same feature set: the One-Class SVM, LOF, PCA, the Isolation
Forest, the autoencoder, the cascade, ECOD and COPOD with tail probabilities from the fit
split only, Deep Isolation Forest, and Deep SVDD, with the hyperparameters of the
manuscript. Two hyperparameters depend on the input dimension d and follow a fixed rule
that keeps the ratios of the manuscript: PCA keeps max(2, round(8d/21)) components and the
autoencoder bottleneck has max(2, round(6d/21)) units, with hidden layers of 32 and 16
units unchanged. For d = 21 this gives the manuscript's 8 and 6, for d = 7 it gives 3 and
2, and for d = 5 it gives 2 and 2. Deep SVDD receives n_features = d. The rule-based
heuristic is defined on its own five flags and stays unchanged across sets, and the
volume ranking stays the received value. DRIFT is the mean of the autoencoder and
Isolation Forest ranks against the reference, as in the manuscript, and the other
aggregation rules of the manuscript are computed on each set.

A secondary score, DRIFT-S, ranks the autoencoder and Isolation Forest scores of a wallet
against the reference wallets of the same canonical script type and averages the two
ranks; a type with fewer than 200 reference wallets is ranked against the whole
reference. DRIFT-S is computed for the BR set only and is not part of any criterion.

## Benchmarks

The ten seeds, the 50/15/35 split, the prevalence of 5%, and the anomaly streams of
`run_experiments.py` are used, so G2 (stream 1), G3 (stream 4), and H (stream 5) receive
the injected wallets of the manuscript and the F21 results must reproduce the stored ones.
H' is a new benchmark of four typologies, drawn from stream 90. They were written down
here before any run, by the author rather than by an independent colleague, and the
supplement will say so. Each typology is an address-level stylization; the transaction
count n, the lifetime span in days, the received value, and the sent fraction are drawn as
follows, the span is then measured over the last 50 transactions as for G2, G3, and H, and
script type and unconfirmed count follow the corpus.

Ransom collection: a few victim payments to one address, swept out soon after; n uniform
on the integers 3 to 12, span uniform on 1 to 30 days, received value 10^U(-0.5, 1.5) BTC,
sent fraction exactly 1.

Donation drive: many small contributions to one published address over months, rarely
spent; n uniform on 50 to 2,000, span uniform on 60 to 900 days, received value n times
10^U(-3.5, -2.0) BTC, sent fraction uniform on 0 to 0.5.

Escrow float: an intermediary address with many deposits and withdrawals that keeps a
float; n uniform on 60 to 800, span uniform on 20 to 365 days, received value 10^U(0.5,
2.5) BTC, sent fraction uniform on 0.6 to 0.9.

OTC settlement: few very large transfers over a long period, passed on almost in full;
n uniform on 6 to 40, span uniform on 90 to 720 days, received value 10^U(2.0, 3.5) BTC,
sent fraction uniform on 0.97 to 1.

H has been seen, and change 3 was motivated by the single-hop and dust results on H, so H
cannot confirm change 3 by itself; H' can.

## Measurements

For every feature set, detector, and aggregation rule, and for G2, G3, H, and H' on the
test split: ROC-AUC, PR-AUC, P@1%, R@5%, R@10%, and recall per typology at the 5% budget;
the matched AUC against the five real test wallets nearest in log count, log received
value, and log span (as in the manuscript); and the ROC-AUC against real test wallets with
at least 100 and at least 1,000 transactions, with the share of high-activity wallets
among the false positives at the 5% budget. Further, the noise-matched negative control
(real test wallets through the G1 noise path and through the identity-preserving noise
path, default width), the validity control (the four ADBench types generated in the
feature space of each set and the same samples reassembled into valid records, with the
share of records that break an identity of that set), and the share of unseen real test
wallets above the 95th percentile of the reference.

The external checks use five seeds with detectors fitted on the whole corpus, as in the
manuscript, for every set and every detector of the corpus-fitted suite, and additionally
the four PyOD detectors for F21 and BR. They report, for the 527 sanctioned addresses, the
median percentile, the top-10% shares of the addresses, of corpus wallets of the same
script types, and of the matched controls, the matched enrichment, and the top-10% shares
of the P2WSH addresses and their controls; the matched enrichment with five-fold
cross-fitted scores, with bootstrap intervals over addresses and over designated profiles,
under the matching of the manuscript and under the matching also on an empty balance; the
AUC of the illicit-associated against the service addresses of BABD-13 sample 1, overall
and within P2PKH and P2SH; and the AUC with which 1,250 injected G2 wallets and the
sanctioned addresses outrank the verified high-activity services of sample 2.

## Success criteria

All criteria concern DRIFT on BR (DRIFT-BR) against DRIFT on F21 (DRIFT-F21, the method of
the manuscript) or against the detectors given BR. Paired tests are two-sided Wilcoxon
signed-rank tests over the ten seeds.

S1 (size-matched detection). On each of G3, H, and H', the ten-seed mean matched AUC of
DRIFT-BR exceeds that of DRIFT-F21 by at least 0.05, and the Holm-adjusted P over these
three comparisons is below 0.05 for each.

S2 (verified services). The AUC with which DRIFT-BR ranks the injected G2 wallets above
the verified high-activity services is not lower than that of DRIFT-F21 by more than 0.03.

S3 (hard negatives). On G2, the ten-seed mean ROC-AUC of DRIFT-BR against real test
wallets with at least 100 transactions exceeds that of DRIFT-F21, with P below 0.05.

S4 (held-out typologies). On H', the ten-seed mean PR-AUC of DRIFT-BR exceeds that of each
of the nine detectors given BR (One-Class SVM, LOF, PCA, Isolation Forest, autoencoder,
ECOD, COPOD, Deep Isolation Forest, Deep SVDD), and the Holm-adjusted P over these nine
comparisons is below 0.05 for each.

The revision meets its criteria only if S1 to S4 all hold. The PR-AUC on G2 and G3 is
reported for every set and is expected to fall for BR, because the current rate carries
size that G2 rewards; it is not a criterion. Neither are the volume ranking, the
heuristic, and the aggregation rules (five-detector mean, minimum of ranks, cascade),
which are reported beside DRIFT. Every result of W21, B, BR-u, and DRIFT-S is descriptive.

## Protocol

No parameter of any set is chosen on any split: the bins, the sizing rule, the pooling
rule, and the winsorization setting are fixed above, so the validation split is not used
for any choice. Winsorization is reported in both settings (BR and BR-u), not selected.
Before the full run the code is checked once on one seed with the F21 set only, to
confirm that the stored F21 values of the manuscript are reproduced exactly, and the
revised pipeline is checked on a subsample for shapes and missing values without
computing or printing any metric. The full run on the test split, H, and H' is then
executed once. All results are reported in full, including every criterion that fails.

If the criteria are met, the manuscript gains one sentence in the Discussion stating that
a pre-specified behavioral basis met them, with the details in the supplement, and the
primary method stays as it is; if they are not met, the same sentence states that it did
not. In either case the response letter explains the analysis.
