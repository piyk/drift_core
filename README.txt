Supplementary material for manuscript T-IFS-27099-2026
"Unsupervised Deep Anomaly Detection for Bitcoin Wallet Forensics:
A Hybrid Autoencoder and Isolation Forest Framework"

All data are public on-chain records.

CONTENTS
  supplement.pdf            Supplementary document: secondary tables and figures, and the
                            details of the added analyses (Sections S1 to S10).
  blockchain-dataset/       The seven source files of the wallet corpus (10,000 records each,
                            collected on 5, 9, 11, 19, 21, 27 and 29 November 2025) and the
                            collection script Wallet3.py.
  src/              	    Code (see below).
  outputs/                  Stored results, aggregated and per seed; the feature snapshots of
                            the sanctioned and the labeled addresses; the comparison of
                            address lists.
  ofac/lists/    	    The Bitcoin addresses extracted from the OFAC SDN list
                            (advanced XML format, issued 9 November 2026).
  requirements.txt  Software versions (Python 3.11.5, CPU only).

CODE
  drift_core.py             Data consolidation and deduplication, the 21 features, the anomaly
                            generators G1, G2, G3 and H, the detectors, the aggregation rules,
                            and the metrics.
  run_experiments.py        Held-out protocol over ten seeds (main comparison, alternative
                            generators, prevalence, noise-matched negatives, evasion, contaminated
                            fit split) and the transductive protocol of the submitted version.
  run_extensions.py         Further controls with the same splits and injected wallets:
                            out-of-sample rank reference, volume baselines, matched negatives,
                            ECOD, COPOD, Deep Isolation Forest and Deep SVDD (PyOD), and the
                            four generic anomaly types of ADBench.
  stats_tests.py            Paired Wilcoxon tests over seeds, Holm correction, and
                            validation-based rule selection.
  separability.py           Supervised random-forest reference and best single feature.
  roc_recheck.py            Contribution of each generator artefact to the ROC-AUC, under the
  roc_decomposition.py      submitted and the held-out protocol; step-by-step decomposition.
  ofac_scrape.py            Retrieval of the features of the sanctioned addresses.
  ofac_compare_lists.py     Comparison of the address list with the list of the submitted version.
  ofac_external.py          External check: all detectors, bootstrap confidence intervals,
                            matched controls.
  ofac_recheck.py           Checks without matching: stratification by received value and
                            alternative matching designs.
  ofac_programs.py          Sanctions program and party type of every listed address
                            (needs SDN_ADVANCED.XML; its output is included).
  ofac_crossfit.py          External check with five-fold cross-fitted scores, bootstrap over
                            designated profiles, strata, and balance of unmatched features.
  ofac_figure_data.py       Score densities and percentile ranks for the figure of the
                            external check.
  babd_scrape.py            Draws the two samples of labeled addresses from BABD-13 and
                            retrieves their wallet fields (needs BABD-13.csv; the sampled
                            addresses and the retrieved fields are included).
  babd_attach_labels.py     Restores the BABD-13 labels of the two samples from BABD-13.csv
                            (run by run_all.py before the labeled checks).
  babd_source_check.py      Compares wallet fields retrieved from two explorer services.
  babd_external.py          Labeled external checks on the two BABD-13 samples.
  pyod_external.py          External checks (sanctioned and labeled addresses) for the four
                            PyOD detectors.
  adbench_repair.py         Repair control of the generic anomaly types: feature-space
                            anomalies reassembled into valid records.
  run_g2w.py                The primary benchmark G2 against G2L, the same draws with the
                            span over the whole activity, all detectors, ten seeds.
  features_v2.py            The pre-specified feature revision: the feature sets (the 21
  run_features.py           features, W21, B, BR, BR-u), the detectors given each set, the
  features_external.py      held-out typologies H', the injected benchmarks and the external
                            checks; the specification, frozen before the code, is
                            features_prespecification.md (SHA-256 digest in
                            features_prespecification.sha256 and in every result file).
                            The per-job percentile arrays of the external checks
                            (outputs/features/external_parts) are not included;
                            features_external.py regenerates them.
  make_tables_features.py   Tables and numeric values of the feature revision; also checks
                            that its 21-feature column equals the stored results. The
                            revision did not meet its success criteria and is not reported
                            in the article or in the supplementary document.
  make_tables.py            Write every table body and numeric value of the manuscript and
  make_tables_ext.py        of the supplementary document from the stored results.
  make_tables_text.py       Writes every number of the running text and checks the
                            qualitative statements that the text makes about them.
  make_figures.py           Draws every figure.
  run_all.py                Runs every script above in one pass with the pinned
                            floating-point settings and writes outputs/run_manifest.json.
  verify_determinism.py     Compares the main run with the verification run and with an
                            earlier run on a processor without AVX-512.
  summarize.py              Prints the aggregated results as text tables.

HOW TO REPRODUCE
  Unpack the archive, keep the folder layout, install the packages of
  requirements.txt, and run from the folder src:

    python run_all.py                (about 2 hours on an 8-core CPU)

  run_all.py sets MKL_CBWR=AVX2, ATEN_CPU_CAPABILITY=avx2 and OPENBLAS_CORETYPE=Haswell
  for every script (and 3 BLAS and OpenMP threads for the scripts of the feature
  revision, the count that the 5-process pools of the other scripts use), runs all experiments, repeats the main comparison into
  outputs/rerun_repeat, compares the two (outputs/determinism_check.json), and
  regenerates every table, numeric value and figure. To regenerate only the tables and
  figures from the stored results, run: python run_all.py --tables-only

  WHAT TO EXPECT. The training of the autoencoder amplifies differences between the
  floating-point kernels that the math libraries select for a processor. The pinned
  settings select the AVX2 kernels, which every processor with AVX2 executes identically,
  so every stored value should be reproduced exactly on such a processor. drift_core.py
  stops with an error if the settings are not in effect. The folder
  outputs/crosscheck_i7-6700K holds an earlier run of the main comparison on a processor
  without AVX-512 (made before G2 measured synthetic wallets as real ones are measured);
  the pinned run reproduced exactly every value that both runs define (G1, G3 and H).

  The SDN_ADVANCED.XML file (128 MB) is not included. It is published by the U.S. Office
  of Foreign Assets Control at https://sanctionslist.ofac.treas.gov/ . The extracted
  address list, the program of each address and the retrieved features are included, so
  the external check runs without it.

  The BABD-13 file is not included. It is distributed by its authors (Y. Xiang et al.,
  "BABD: A Bitcoin address behavior dataset for pattern analysis", IEEE Trans. Inf.
  Forensics Security, vol. 19, pp. 2171-2185, 2024) at
  https://www.kaggle.com/datasets/lemonx/babd13 under the licence CC BY-NC-ND 4.0, which
  does not permit redistributing a modified copy. The labels are therefore not included:
  babd_sample.csv and babd_sample_high.csv list only the sampled addresses, and
  babd_per_address.json holds the scores of each address without its class. The
  babd_features files hold the public on-chain fields of the addresses. To rerun the
  labeled checks, place BABD-13.csv in external/babd13/ and run
  src/babd_attach_labels.py (run_all.py does this); it restores the two sample files
  exactly as they were used. The reported results of these checks are in
  outputs/babd_external.json, which holds statistics per class and no address.
