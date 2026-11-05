# Structured Shared-Private Embedding Multiplexing: Simulation Code and Data

Reproducibility package for "Semantic Multiple Access via Structured Shared-Private
Embeddings in Mobile Networks" (Ki-Ho Lee, Hyun-Ho Choi, Jung-Ryun Lee; under review
at IEEE Transactions on Mobile Computing, major revision 2026). Every figure and every
number in the paper is regenerated from the files here.

## Layout
- `code/semantic_mac.py` — shared library: embedding model, masks, matched-filter front
  end, receivers (SimR, SimC, LMMSE, DR, DR-MMSE), spectral recovery, affinity tracker,
  conventional references (OMA, NOMA-SIC), and the main configuration block
  (`U_MAIN`, `D_SYN/DC_SYN`, `D_REAL/DC_REAL`, SNRs, pilot parameters, `TAU`, `H_MIN`, `MOB`).
- `code/exp*.py` — one script per experiment; each writes its CSV to `data/`.
- `code/replot_all.py` — regenerates every result figure in `fig/` from `data/` only.
- `code/check_*.py`, `code/lmmse_verify.py` — closed-form and optimality checks.
- `code/overhead_accounting.py` — the pilot/feedback/calibration overhead accounting.
- `data/` — raw results (CSV) and notes behind every figure and table.
- `fig/` — figure PDFs as printed.
- `docs/proof_proposition1.pdf` — complete proof of Proposition 1 (the three limits
  and the optimality property). The paper states the proposition with a short argument,
  and this note gives every step with the paper's notation and equation numbers.

## Figure to script to data
| Figure | Script | Data |
|---|---|---|
| Fig. 1 | `fig1_system_r1.tex` (TikZ) | — |
| Fig. 2 | `exp1_theory.py` | `e1_beta.csv` (the closed-form text uses `e1_dr_cf.csv` from `check_dr_closedform.py`) |
| Fig. 3 | `exp1_theory.py` | `e1_snr.csv` |
| Figs. 4–6 | `exp2_structure.py` | `e2_spectrum_multi.csv`, `e2_subspace_multi.csv`, `e2_ladder.csv` |
| Figs. 7–8 | `exp3_mobility.py` | `e3_timeseries.csv`, `e3_speed.csv` |
| Fig. 9 | `exp4_mismatch.py`, `exp4_lmmse_ref.py` | `e4_mismatch.csv`, `e4_lmmse_ref.csv` |
| Fig. 10 | `exp7_learned_sideinfo.py`, `exp7b_learned_robustness.py` | `e7_learned_sideinfo.csv`, `e7b_learned_robustness.csv` (Learned + side info, fed before the division) |
| Fig. 11 | `exp8_partial_sharing.py` | `e8_partial_sharing.csv` |
| Fig. 12 | `exp12_corpus.py` | `e12_corpus.csv` |
| Fig. 13 | `exp11_impairments.py` | `e11_imbalance.csv` |
| Table V | `exp9_retrieval.py` | `e9_retrieval.csv` |
| Table VI | `exp11b_sync.py` (symbol-level phase, CFO, timing, phase noise and gain errors) | `e11b_sync.csv`, `e11b_sync_perseed.csv` |
| E11 static phase and channel-estimation errors (text) | `exp11_impairments.py` | `e11_phase.csv`, `e11_cest.csv` |
| E10 (text), thresholds 0.35 to 0.65 | `exp10_eval_choices.py` | `e10_threshold_sweep.csv`, `e10_clip_sweep.csv` |
| E17 (text) | `exp17_pilot_budget.py` | `e17_pilot_budget.csv` |
| E17 per-run values (text) | `exp17_perrun.py` | `e17_perrun.txt` |
| E3 closed loop, masks from the fed-back estimate (text) | `exp3_closed_loop.py` | `e3_closed_loop.txt` |
| E12 at 30 dB (text) | `exp16_scene_gap.py` | `e16_scene_gap.csv` |
| E7 held-out training curves (text) | `exp7_valcurve.py` | `e7_valcurve.csv` |
| Fig. 3, 20 dB point at 4×10⁴ draws (text) | `exp1_snr20_check.py` | `e1_snr20_check.txt` |
| E11 branch mechanism and draw spread (text) | `exp11_mechanism.py`, `exp11_seedvar.py` | `e11_mechanism.txt`, `e11_seedvar.txt` |
| E18 adapter sensitivity (text) | `exp18_adapter_sensitivity.py` | `e18_adapter_sensitivity.csv`, `e18_log.txt` |
| Runtimes (text) | `runtime_check.py` | `runtimes.txt` |
| E6, DR vs. DR-MMSE on the private block, single draws and fading average (text) | `check_blmmse_optimality.py`, `check_blmmse_fading.py` | `e6_optimality.txt`, `e6_optimality_fading.txt` |
| E9 retrieval threshold bins (text) | `exp9_retrieval.py` | `e9_threshold.csv` |
| E14, E15 DR vs. DR-MMSE retrieval sweep (text, letter) | `exp14_dr_vs_subspace.py`, `exp15_adapter_receiver.py` | `e14_dr_vs_subspace.csv`, `e15_adapter_receiver.csv` |
| Shared-only receiver, SER at tau = 0.45 (text) | `check_shared_only.py` | `e10_shared_only.csv`, `e10_shared_only_real.csv` |
| Realized-mask check at two points (text) | `check_mask_realization.py` | `e_mask_check.csv` |
| Realized-mask SNR sweep, inflated-noise receivers and MSE floor (text) | `check_mask_realization.py sweep` | `e_mask_sweep.csv`, `e_mask_floor.csv` |
| E3 closed loop with masks fixed at the time average, net of pilots (text) | `exp19_static_masks.py` | `e19_static_masks.csv`, `e19_notes.txt` |
| Calibration over a noisy link (text) | `check_calibration_snr.py` | `e2_calibration_snr.csv` |
| E3 speed sweep, per-seed SER and visited affinity (letter) | `exp3_speed_stats.py` | `e3_speed_seeds.csv`, `e3_speed_stats.txt` |
| E20 covariance-aware DR and full-covariance LMMSE on BERT contents (text) | `exp20_covariance_receiver.py` | `e20_covariance_receiver.csv`, `e20_notes.txt` |
| E21 PCA-ordered OMA, covariance-aware NOMA-SIC and retrieval-candidate statistics (text) | `exp21_conventional_references.py` | `e21_conventional_references.csv`, `e21_notes.txt` |
| E20b full-covariance LMMSE variants, calibration sweep and receiver cost (text, Tables S2-S3) | `exp20b_fullcov_variants.py` | `e20b_fullcov_variants.csv`, `e20b_calib_sweep.csv`, `e20b_cost.csv`, `e20b_notes.txt` |
| E19b causal and biased affinity estimates net of pilots (text, Tables S4-S5) | `exp19b_causal_average.py` | `e19b_causal_average.csv`, `e19b_perseed.csv`, `e19b_notes.txt` |
| E8b sample-to-sample non-common fraction (Table S6) | `exp8b_varying_kappa.py` | `e8b_varying_kappa.csv`, `e8b_varying_kappa_summary.csv`, `e8b_notes.txt` |
| E11b synchronization and gain errors at symbol level (text, Table VI, Table S7) | `exp11b_sync.py` | `e11b_sync.csv`, `e11b_sync_perseed.csv`, `e11b_notes.txt` |
| E7b side-informed attention receivers (Fig. 10, Table S8) | `exp7b_learned_robustness.py`, `exp7b_summary.py` | `e7b_learned_robustness.csv`, `e7b_valcurve.csv`, `e7b_notes.txt`, `e7b_summary.txt` |
| E12b three sources of the share coefficients (text, Table S9) | `exp12b_sideinfo.py` | `e12b_sideinfo.csv`, `e12b_notes.txt` |
| E2b misspecified shared dimension (Table S10) | `exp2b_dc_misspec.py` | `e2b_dc_misspec.csv`, `e2b_notes.txt` |
| E2c adapter gain versus SNR (Table S11) | `exp2c_adapter_snr.py` | `e2c_adapter_snr.csv`, `e2c_summary.csv`, `e2c_notes.txt`, `e2c_trainer_check.txt` |
| E9b SER threshold per scheme and per block (Table S12) | `exp9b_threshold_by_scheme.py` | `e9b_threshold_by_scheme.csv`, `e9b_scheme_summary.csv`, `e9b_notes.txt` |
| E10b gain clip (text, Table S13) | `exp10b_clip_reference.py` | `e10b_clip_reference.csv`, `e10b_notes.txt` |
| E24 MS-COCO captions of one image as the users (Table S14) | `extract_coco_captions.py`, `exp24_coco_captions.py` | `e24_coco_captions.csv`, `e24_notes.txt`, `coco_pool_notes.txt` |
| Small rev5 checks and device-side operation count (text) | `rev5_small_notes.py`, `device_cost.py` | `rev5_small_notes.txt`, `device_cost.txt` |
| BERT embedding pool (input data) | `extract_bert_embeddings.py` | `bert_agnews_8000.pt`, `bert_pool_regen.txt` |

## Requirements and conventions
- CPU runs (all experiments except E7): Python 3.10, NumPy 2.2, matplotlib 3.9, PyTorch 2.5 (CPU) for E2/E5/E15.
- GPU run (E7, `exp7_learned_sideinfo.py`): Python 3.12, NumPy 2.5, PyTorch 2.11+cu128 on an NVIDIA RTX PRO 5000. The same environment ran the realized-mask sweep, E19, E20, E21, the shared-only and calibration checks and `exp11_mechanism.py`, all on CPU except the full-covariance LMMSE rows of E20. Exact versions are in `data/environment.txt`. `semantic_mac.DEVICE` selects CUDA when available.
- Seeds are fixed inside every script (`np.random.default_rng(<constant>)`,
  `torch.manual_seed`); E7 trains over seeds 0–4.
- Embeddings are unit norm; the per-user SNR is rho = 1/sigma^2 referenced to a
  unit-norm embedding; block-fading Rayleigh gains with E[h^2] = 1 are clipped at
  h_min = 0.2; the SER threshold is tau = 0.45 on the cosine recovery.
- The BERT embeddings (`data/bert_agnews_8000.pt`, 8000 x 768) are provided for the
  real-embedding experiments. Each is the last-layer bert-base-uncased output, mean pooled
  over the tokens (truncation at 64), of the text before the first period of an AG-News
  training item, kept when that text has 5 to 30 words. The pool holds the first 8000 such sentences in the
  training order, shuffled at seed 0. `code/extract_bert_embeddings.py` regenerates it
  (needs `transformers`, `datasets` and network access). Recent `datasets` releases reject
  the id `ag_news`, so pass a mirror, e.g. `--dataset wangrongsheng/ag_news`, which
  reproduced the shipped pool to 2.2e-5 per coordinate (`data/bert_pool_regen.txt`).
  The script writes `data/bert_agnews_8000_regen.pt` by default and leaves the shipped pool intact.

- E24 uses the MS-COCO 2017 caption annotations (T.-Y. Lin et al., ECCV 2014, CC BY 4.0),
  which are not redistributed here. `code/extract_coco_captions.py` downloads nothing; pass the
  annotation folder with `--ann`. It writes the BERT caption embeddings that
  `code/exp24_coco_captions.py --coco <folder>` reads.
- The rev5 experiments (E2b, E2c, E7b, E8b, E9b, E10b, E11b, E12b, E19b, E20b and E24 above) ran in the GPU environment listed in
  `data/environment.txt`.

Run `python3 code/exp1_theory.py` ... `python3 code/exp17_pilot_budget.py` to regenerate
`data/`, then `python3 code/replot_all.py` to regenerate `fig/`.
