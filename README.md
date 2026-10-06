# oneoverf-separation

Separating periodic (oscillatory) from aperiodic (1/f) activity in neural
power spectra, and the assumption hidden in that step. Part of the
sccn/OneOverF collaboration (separation pillar).

Removing the aperiodic background from a spectrum always involves an
assumption about how it combines with the oscillations. Subtracting it in
linear power (as in IRASA's oscillatory spectrum, defined as mixed minus
fractal) assumes the two are added. Subtracting it in log power (as in
specparam's peak heights) assumes the oscillation scales with the background.
Estimating the background commits to neither; the assumption comes with the
removal step. This repository treats that assumption as a parameter, the
coupling exponent λ:

    P(f) = L(f) + Σ aₙ Gₙ(f) L(f)^λ

where L is the aperiodic component and Gₙ are peak shapes. λ = 0 is additive
and λ = 1 multiplicative. For band power this reduces to log a = log c +
λ log b: in a generalized linear model of periodic power, λ is the
coefficient of the background entered as a covariate. Any effect measured
under an assumed λ is then linear in λ, s(λ) = s_a − λ s_b, and changes sign
at a single crossover λ* = s_a / s_b.

The code covers:

- a MATLAB library of aperiodic estimators (full, censored and robust
  regression, Theil–Sen, a specparam port, IRASA, a knee+plateau Whittle fit)
  and of the coupling model;
- simulations with known ground truth: an estimator benchmark, the effect
  of the assumed λ on condition contrasts, and calibration of the λ estimate;
- a Python pipeline for the Healthy Brain Network resting-state EEG
  (eyes open vs eyes closed, about 2,800 participants), including
  specificity controls, alpha power against age, and a topographic test
  with its null simulation;
- re-analyses of published claims in five more open datasets (adult aging,
  anaesthesia, propofol sedation and induction), and a coupling estimator
  that does not need positive periodic power, tested within recordings,
  between sessions, under sedation and in intracranial recordings;
- a synaptic model of the EEG showing what λ means physically, and a
  scalp-to-cortex distance analysis from HBN MRI;
- `lambdacurve/`, a small Python package (and `code/lib/oof_lambda_curve.m`
  for MATLAB) that reports how a periodic-power result depends on λ.

The accompanying paper is in preparation; a preprint will be linked here.

## Main findings so far

- λ cannot be recovered from a single spectrum, only from how the
  background varies across trials, epochs or conditions.
- Assuming the wrong λ distorts condition effects. In a simulated contrast
  in which the background steepens after the event, the recovered change
  is 69% too large when the truth is additive but the background is divided
  out (λ = 1), and 72% too small when the truth is multiplicative but the
  background is subtracted (λ = 0).
- Censored regression is the least biased of the simple aperiodic
  estimators.
- In HBN (2,396 participants aged 5-22 after quality control), eyes-closed
  alpha power decreases with age when the background is removed additively
  (λ = 0, -5.0% per year) and increases when it is removed as specparam
  does (λ = 1, +4.8% per year). The sign changes at λ ≈ 0.52.
- Of 13 published claims about periodic power re-tested in six open
  datasets (more than 7,000 participants), four depend on λ, all where the
  background itself changes (child development, propofol induction); adult
  aging (Dortmund, LEMON) and the age decline of alpha under anaesthesia
  (VitalDB) hold under both rules.
- No open design identifies λ. Estimates move with the specification
  (eyes closed vs open: 0.29-0.80), with eye state and over a session
  (within-recording fluctuations: 1.3-1.9 eyes closed, -0.1-0.7 eyes open),
  between sessions (0.33 and 0.87) and in intracranial recordings (0.93; 0.76
  over posterior contacts).
  The usual regression of log periodic on log background power, restricted
  to positive periodic power, is biased towards 1; lambda_gmm.py is not.
- λ is a property of the comparison, not of the tissue. In a synaptic model
  a gain change gives λ = 1, a change in drive 0 or 2, stronger inhibition
  about 1.3; mixtures give the background-weighted average, which exceeds 1
  when the changing source carries more of the rhythm than of the
  background.
- Scalp-to-cortex distance measured from MRI grows with age but explains
  little of the HBN background decline (share 0.02, CI -0.13 to 0.18); the
  developmental reversal is not a gain artefact.

## Layout

    code/
      lib/                       MATLAB functions (oof_*): model, synthesis,
                                 Welch PSD, aperiodic estimators, coupling fit
      sim01_estimator_benchmark.m  estimator bias, reliability, coupling estimate
      sim02_ersp_contrast.m        condition contrasts under additive vs
                                   multiplicative coupling
      test_synth.m, test_lambda_identify.m   checks of synthesis and
                                   identifiability
      hbn_extract_psd.py         download HBN-EEG and compute Welch PSDs
                                 (full, odd/even, thirds, per-segment ROI)
      hbn_fit.py                 aperiodic fits and band power per subject
      hbn_lambda.py              coupling estimate and (p, q) tests
      hbn_controls.py            window and band specificity controls
      hbn_kp.py, ap_models.py    knee+plateau aperiodic model
      hbn_age_alpha.py           alpha vs age under each separation rule
      hbn_age_alpha_robust.py    quality control; age slopes across
                                 estimators; crossover λ*
      hbn_age_alpha_bands.py     fixed band, specparam and scalp-gain checks
      hbn_age_alpha_phases.py    slopes and crossover below and above an age cut
      hbn_age_alpha_iaf.py       share of the result due to the band moving with
                                 alpha frequency
      hbn_topography*.py         topographic test (fitted and fit-free)
      spatial_neff.py            effective number of channels for map
                                 correlations
      sim_*.py                   calibration curves and null simulations
      identifiability_power.py   data needed to identify λ from one spectrum
      joint_fit.py               joint aperiodic + peak fit
      lambda_curve.py            effect as a function of λ, crossover λ*
      lambda_gmm.py              log-free estimation of λ (two conditions,
                                 or epochs within a recording)
      sim_lambda_gmm.py          simulations of the estimators
      hbn_thirds.py              HBN λ with total, background and instrument
                                 from three different thirds of the segments
      sim_lambda_thirds.py       halves vs thirds when the background differs
                                 between segments
      psd_utils.py               shared preprocessing and Welch spectra
      dortmund_extract_psd.py, lemon_extract_psd.py, vitaldb_extract.py,
      fetch_open_data.py         download and reduce the other datasets
      aging_lambda.py            Dortmund and LEMON: age and alpha under λ
      vitaldb_lambda.py          anaesthesia: age and frontal alpha under λ
      chennu_analysis.py         graded propofol sedation
      brake_analysis.py          propofol induction spectrograms
      ds003690_epochs.py, ds003690_lambda.py, dortmund_levels.py
                                 λ from fluctuations within a recording
      srm_extract_psd.py, srm_lambda.py   SRM adult EEG, test-retest λ
      ieeg_epochs.py, ieeg_lambda.py      intracranial rest recordings
      inject_calibration.py      calibration with a known rhythm added to
                                 recorded spectra, with a common drive
      sim_mechanisms.py          synaptic model: what sets λ
      hbn_age_psychopathology.py  age result with CBCL scores as covariates
      hbn_mri_fetch.py, hbn_scalp_distance.py, hbn_gain_mri.py
                                 scalp-to-cortex distance and scalp gain
      breadth_summary.py, identification_summary.py   summary tables
      figures/make_figures.py    Figures 1-5 and Supplementary Figures 1-5
    lambdacurve/                 Python package: λ-curve, crossover λ*,
                                 log-free estimator, matched null
    results/                     group-level outputs (CSV)
    archive/                     superseded first simulations (see its README)

## Running

MATLAB with the Statistics and Machine Learning, Optimization and Signal
Processing toolboxes:

    matlab -batch "run('code/test_synth.m')"
    matlab -batch "run('code/sim01_estimator_benchmark.m')"
    matlab -batch "run('code/sim02_ersp_contrast.m')"
    matlab -batch "scenario='flatten'; run('code/sim02_ersp_contrast.m')"

Python 3.10+ with numpy, scipy, pandas, statsmodels, scikit-learn, mne,
specparam, matplotlib and h5py:

    # 1. PSDs from OpenNeuro (HBN-EEG releases 1-11: ds005505-ds005512 and
    #    ds005514-ds005516; there is no ds005513);
    #    output directory set by HBN_OUT (default ./hbn_psd)
    python code/hbn_extract_psd.py ds005505 ds005506 --workers 6
    # 2. fits and coupling estimates
    python code/hbn_fit.py
    python code/hbn_lambda.py
    python code/hbn_controls.py
    python code/hbn_kp.py --workers 8
    python code/hbn_thirds.py --workers 8
    python code/hbn_age_alpha.py
    python code/hbn_age_alpha_robust.py      # writes the QC flags used below
    python code/hbn_age_alpha_bands.py
    python code/hbn_age_alpha_phases.py
    python code/hbn_age_alpha_iaf.py
    # 3. topographic analyses and their null
    python code/hbn_topography.py --workers 8
    python code/hbn_topography.py --workers 8 --fit-range 4,40 --tag _f4-40
    #    likewise 2,30 (_f2-30), and 5,40 with --censor 7,16 (_f5-40)
    python code/hbn_topography_flanks.py
    python code/spatial_neff.py
    #    null: as many simulated participants as real ones; the CSVs are
    #    appended to, so delete them first
    python code/sim_topography_leakage.py --workers 8 --n 1800 --rebuild-heights \
        --out results/sim_topography_null.csv \
        --out-flanks results/sim_topography_null_flanks.csv
    #    repeat (without --rebuild-heights) with --harmonic 0 and with --iaf-shift -0.3
    # 4. calibration and identifiability
    python code/sim_calibration.py
    python code/identifiability_power.py
    python code/sim_lambda_gmm.py --out results/sim_lambda_gmm_common.csv
    python code/sim_lambda_thirds.py
    # 5. other datasets (raw files are deleted after reduction; set
    #    DORTMUND_OUT, LEMON_OUT, VITALDB_OUT, VITALDB_META, EEG_DATA)
    python code/dortmund_extract_psd.py --workers 6
    python code/lemon_extract_psd.py --workers 4
    python code/vitaldb_extract.py --workers 6      # needs vitaldb (pip)
    python code/fetch_open_data.py                  # Chennu, Brake, ds003690
    python code/aging_lambda.py
    python code/vitaldb_lambda.py
    python code/chennu_analysis.py
    python code/brake_analysis.py
    python code/ds003690_epochs.py && python code/ds003690_lambda.py
    python code/dortmund_levels.py --cond ec_pre,eo_pre,ec_post,eo_post --no-cov
    python code/breadth_summary.py && python code/identification_summary.py
    # 6. figures
    python code/figures/make_figures.py figures

Per-subject and per-case derivatives are not included, because they contain
participant age and sex or clinical data. All of them can be regenerated from
the public data. The one per-subject file included, results/hbn_qc_flags.csv,
holds only the quality-control flags.

## Data

All open:

- Healthy Brain Network EEG (Shirazi et al., 2024), OpenNeuro ds005505-ds005512
  and ds005514-ds005516.
- Dortmund Vital Study (Getzmann et al., 2024), OpenNeuro ds005385.
- MPI-Leipzig LEMON (Babayan et al., 2019), INDI server.
- VitalDB (Lee et al., 2022), PhysioNet.
- Propofol sedation (Chennu et al., 2016), University of Cambridge Apollo,
  doi:10.17863/CAM.68959.
- Propofol induction spectrograms (Brake et al., 2024), figshare,
  doi:10.6084/m9.figshare.24777990.
- Young and older adults with pupil and EOG (Ribeiro & Castelo-Branco, 2019),
  OpenNeuro ds003690.

## Context

This work is part of the sccn/OneOverF collaboration on periodic/aperiodic
separation (https://github.com/sccn/OneOverF).

## References

Alday, P. M. (2019). How much baseline correction do we need in ERP research?
Extended GLM model can replace baseline correction while lifting its limits.
Psychophysiology, 56(12), e13451.

Donoghue, T., et al. (2020). Parameterizing neural power spectra into periodic
and aperiodic components. Nature Neuroscience, 23(12), 1655–1665.

Gyurkovics, M., Clements, G. M., Low, K. A., Fabiani, M., & Gratton, G.
(2021). The impact of 1/f activity and baseline correction on the results and
interpretation of time–frequency analyses of EEG/MEG data: A cautionary tale.
NeuroImage, 237, 118192.

Kałamała, P., Clements, G. M., Gyurkovics, M., et al. (2026). How to improve
the reliability of aperiodic parameter estimates in M/EEG: A method
comparison. Psychophysiology, 63(3), e70272.

Shirazi, S. Y., et al. (2024). HBN-EEG: The FAIR implementation of the Healthy
Brain Network (HBN) electroencephalography dataset. bioRxiv,
10.1101/2024.10.03.615261.

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in
the power spectrum of neurophysiological signal. Brain Topography, 29, 13–26.
