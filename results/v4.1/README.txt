Outputs of location_imputation.py version 4.1 (25 October 2026).

Every slot with a recorded mode of transport counts as transport (trips and, for instance,
reading on a train); transport is never imputed. The location classes are the eight places.

AGGREGATE (can be shared):
  checks.csv (data and consistency checks), run_settings.json.
  Leave-one-country-out (each donor imputed from the other 12; AT, FR and LU from all 13):
    loco_overall.csv and loco_summary.csv (metrics per country and variant, and their means),
    loco_by_class.csv (per-place scores; withheld where fewer than 50 respondents stand behind them
    in any country of the coding group, as Eurostat's publication guidelines for the SUF require),
    loco_by_class_means.csv (their means over countries), loco_home_by_slot.csv,
    loco_hours_per_day.csv (weighted hours per day by place, recorded and imputed, every variant).
  Repeated imputations of the held-out countries: loco_repeated_imputations.csv and
    loco_repeated_imputations_summary.csv (coverage of plain 95% intervals), loco_intervals_home_by_slot.csv.
  Germany: germany_shares_by_slot.csv, germany_hours_per_day.csv, germany_changes_per_diary.csv,
    germany_hours_per_imputation.csv (main variant, all 13 donors; lo_mc/hi_mc: plain 95% interval,
    Monte Carlo variation only), home_activities.csv (activities for the code-13 factor).
  Global sensitivity analysis (full factorial, exact first-order and total Sobol' indices):
    gsa_loco.csv (part A: metrics of every combination of method choices and held-out donor),
    gsa_loco_indices.csv (indices of the metrics averaged over the donors), gsa_loco_combinations.csv
    (those averages for each of the 144 combinations, with the Pareto-optimal ones flagged),
    gsa_transfer_errors.csv and gsa_transfer_errors_home_by_slot.csv.gz (imputed minus recorded for each
    donor and combination, the levels of the transfer-error factor), gsa_germany_universes.csv and
    gsa_germany_home_by_slot.csv.gz (German estimates of every combination of the method choices of part A,
    coding assumption and stream of random draws), gsa_germany_combinations.csv (the same minus each
    donor's transfer error under the same method choices), gsa_germany_indices.csv,
    gsa_germany_indices_home_by_slot.csv, gsa_germany_uncertainty.csv and
    gsa_germany_uncertainty_home_by_slot.csv (distribution over all combinations), and the same two files
    ending in _pareto.csv (over the combinations whose method choices are Pareto-optimal in part A).
MICRODATA (keep on your machine, do not send): PRIVATE_germany_imputed_codes.npz, imputation x
  diary x slot in HETUS codes (transport keeps its recorded mode; 30 = trip without a recorded mode),
  and PRIVATE_germany_ids.csv (row order of the diaries). Take uncertainty statements from the
  global sensitivity analysis, not from the spread across these imputations.
imputation_out_to_send.zip, next to this folder, holds the aggregate files only.
NOT FROM THIS RUN (left by an earlier run, not in the bundle or the export): gsa_germany_home_by_slot.csv, gsa_transfer_errors_home_by_slot.csv
