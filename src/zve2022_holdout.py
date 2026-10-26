#!/usr/bin/env python3
"""Hold-out test on the German time use survey of 2022 (ZVE 2022), whose diaries record places.

The German diaries of 2012/13 record no places, so the imputation cannot be checked on them.
The diaries of 2022 record a place or a mode of transport in every ten-minute slot. This
script sets the recorded places aside, imputes them from the 13 HETUS 2010 donors exactly as
the 2012/13 diaries were imputed, and compares the imputed with the recorded places.

  1. Variants. The nine variants of the leave-one-country-out evaluation, each estimated on
     all 13 donors, scored with the seven evaluation metrics, by place (recall and precision,
     withheld where fewer than MIN_RESPONDENTS respondents stand behind them), and as hours per
     day and share at home by slot.
  2. Repeated imputations. REPEAT_IMPUTATIONS imputations of the main specification and the
     coverage of the plain 95% intervals they give, as multiple imputation would report them.
  3. Method combinations (part A of the global analysis). Each of the 144 combinations,
     estimated on the 13 donors, imputes the 2022 diaries. Germany's seven metrics and its
     errors (imputed minus recorded) in hours per day, changes per diary and share at home by
     slot are set against those made for the 13 donors under the same combination, read from
     the results of the main run: does Germany's transfer error look like one of theirs?
  4. Uncertainty (part B). The 2,880 imputations of the 2022 diaries (144 combinations x
     code 13 x draws), each corrected by the 13 donors' transfer errors as for 2012/13, give
     37,440 values of every estimate. Does their 90% range, over all combinations and over
     those with Pareto-optimal methods, contain the value recorded in 2022?

The donors' errors, the Pareto-optimal combinations and the home activities come from the
aggregate results of the main run (results/v4.1), so the 2022 diaries play no part in choosing
or calibrating anything: they are only scored.

Inputs: the settings file of the main run (the HETUS 2010 files; only the 13 donors with full
place coding are read), the 2022 diaries in the HETUS layout written by zve2022_to_hetus.py,
and the results folder of the main run. Outputs, aggregate tables only, go to --out.

Usage:
    python src/zve2022_holdout.py data/restricted/DE22.csv --results results/v4.1 \\
        --out data/restricted/zve2022_out [/path/to/TUS_Eurostat.nml] [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import location_imputation as li  # noqa: E402

VERSION = "1.0 (25 October 2026)"
CODE = "DE22"
FACTOR_A = ["unit", "time", "age", "day_type", "assignment", "alpha"]
LABELS = li.CLASS_LABELS + [li.TRANSPORT_LABEL]
OUTPUTS_B = li.CLASS_LABELS + ["changes per diary"]       # transport is recorded, its error is zero
# streams of random numbers, distinct from those of the main run
STREAM_VARIANTS, STREAM_REPEAT, STREAM_A, STREAM_B = 21, 22, 23, 24


def combo_key(unit, time_, age, day, modal, alpha):
    return (unit, li.yes_no(time_), li.yes_no(age), li.yes_no(day), "most likely" if modal else "draw", float(alpha))


def read_inputs(nml_arg, de22_path, script_dir, only=None):
    """The HETUS donor files (those listed in `only`, else every file with places; Germany
    2012/13 is not needed) and the 2022 diaries, prepared as in the main run."""
    nml, files = li.country_files(nml_arg, script_dir)
    raw = {}
    for code in sorted(files):
        if code == li.TARGET or (only and code not in only):
            continue
        li.log(f"reading {code} ({files[code].name})")
        raw[code] = li.read_country(files[code])
    li.log(f"reading {CODE} ({Path(de22_path).name})")
    raw[CODE] = li.read_country(Path(de22_path))
    codes = np.unique(np.concatenate([r["A"].ravel() for r in raw.values()]))
    li.RADIX["a"] = len(codes)
    data, schemes = {}, {}
    for code in sorted(raw):
        r = raw.pop(code)
        C = li.slot_classes(r["A"], r["L"])
        scheme, fold, present = li.coding_scheme(C)
        if code == CODE:
            de22_checks(r, C, scheme, fold, codes)
        act_index_flat = np.searchsorted(codes, r["A"].ravel()).astype(np.int16)
        E = li.build_episodes(r["A"], C, act_index_flat)
        S = li.slot_units(C, act_index_flat)
        person, identified = li.person_codes(r["ids"], len(C))
        if code == CODE:
            li.check(code, "respondents_identified", identified, True, identified,
                     "HID and PID found" if identified else "HID or PID missing: respondents counted conservatively")
        data[code] = dict(C=C, E=E, S=S, age=r["age"], weekend=r["weekend"], weight=r["weight"], fold=fold,
                          ids=r["ids"], person=person, n=len(C),
                          A=r["A"] if code == CODE else None)
        schemes[code] = scheme
    return nml, files, data, schemes, codes


def de22_checks(r, C, scheme, fold, codes):
    """Checks of the converted 2022 file."""
    st, A, L = r["stats"], r["A"], r["L"]
    li.check(CODE, "activity_slot_columns", st["slot_columns_activity"], li.SLOTS, st["slot_columns_activity"] == li.SLOTS,
             severity="fail")
    li.check(CODE, "location_slot_columns", st["slot_columns_location"], li.SLOTS, st["slot_columns_location"] == li.SLOTS,
             severity="fail")
    li.check(CODE, "activity_code_missing_share", round(st["activity_missing_share"], 6), 0,
             st["activity_missing_share"] == 0)
    li.check(CODE, "location_code_unexpected_share", round(st["location_unexpected_share"], 6), 0,
             st["location_unexpected_share"] == 0, "codes outside 1 to 13 after conversion")
    li.check(CODE, "location_code_negative_share", round(st["location_negative_share"], 6), "<0.01",
             st["location_negative_share"] < 0.01, "slots without a place or mode")
    li.check(CODE, "weights_nonpositive", st["weight_nonpositive"], 0, st["weight_nonpositive"] == 0)
    li.check(CODE, "age_band_missing_share", round(st["age_missing_share"], 6), 0, st["age_missing_share"] == 0)
    li.check(CODE, "day_of_week_missing_share", round(st["day_missing_share"], 6), 0, st["day_missing_share"] == 0)
    li.check(CODE, "place_coding", scheme, "full", scheme == "full",
             ", ".join(li.CLASS_LABELS[k] for k in range(li.N_CLASSES) if fold[k] != k))
    trip = (A >= li.TRAVEL_ACTIVITY[0]) & (A < li.TRAVEL_ACTIVITY[1])
    mode = np.isin(L, li.TRANSPORT_CODES)
    li.check(CODE, "mode_slots_outside_trips_share", round(float((mode & ~trip).sum() / max((~trip).sum(), 1)), 6),
             "for information", True, "slots of other activities with a recorded mode, counted as transport")
    trip_at_place = float((trip & np.isin(L, li.PLACE_CODES)).sum() / max(trip.sum(), 1))
    li.check(CODE, "trip_slots_with_place_code_share", round(trip_at_place, 6), "<0.1", trip_at_place < 0.1,
             "trips recorded with a place instead of a mode")
    li.check(CODE, "diaries", int(len(A)), "for information", True)


def recorded_values(d, groups):
    """Recorded hours per day by place (transport last), changes per diary and share at home by
    slot of the 2022 diaries, defined as for the imputations."""
    full_fold = np.arange(li.N_CLASSES)
    s = li.weighted_slot_shares(d["C"], d["weight"], full_fold)
    hours = 24 * s.mean(0)
    changes = float(li.changes_per_diary(d["C"], full_fold).mean())
    home = {g: li.weighted_slot_shares(d["C"][m], d["weight"][m], full_fold)[:, li.HOME] for g, m in groups.items()}
    return hours, changes, home


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("de22", help="2022 diaries in the HETUS layout (zve2022_to_hetus.py)")
    parser.add_argument("source", nargs="?", default=None, help="path to TUS_Eurostat.nml (optional)")
    parser.add_argument("--results", required=True, help="aggregate results of the main run, e.g. results/v4.1")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--quick", action="store_true", help="fewer draws and one alpha, as a first check")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    res_dir = Path(args.results).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_start, timing = time.time(), {}
    m_rep = 4 if args.quick else li.REPEAT_IMPUTATIONS
    n_draws = 2 if args.quick else li.GSA_DRAWS
    alphas = (li.KAPPA,) if args.quick else li.GSA_ALPHAS

    # results of the main run: donors' errors and metrics per combination, Pareto flags, home activities
    need = ["run_settings.json", "gsa_transfer_errors.csv", "gsa_transfer_errors_home_by_slot.csv.gz",
            "gsa_loco.csv", "gsa_loco_combinations.csv", "home_activities.csv"]
    missing = [f for f in need if not (res_dir / f).is_file()]
    if missing:
        raise SystemExit(f"{res_dir} lacks {', '.join(missing)}: pass the results of a full run of version 4.1 or later")
    main_settings = json.loads((res_dir / "run_settings.json").read_text())
    err_tot = pd.read_csv(res_dir / "gsa_transfer_errors.csv")
    err_home = pd.read_csv(res_dir / "gsa_transfer_errors_home_by_slot.csv.gz")
    loco_g = pd.read_csv(res_dir / "gsa_loco.csv")
    combos_a = pd.read_csv(res_dir / "gsa_loco_combinations.csv")
    home_acts_main = pd.read_csv(res_dir / "home_activities.csv")
    for frame in (err_tot, err_home, loco_g, combos_a):
        frame["alpha"] = frame["alpha"].astype(float)

    t_stage = time.time()
    li.log(f"zve2022_holdout.py version {VERSION}, with location_imputation.py version {li.VERSION}")
    main_donors = main_settings.get("donors_full") or main_settings.get("global_sensitivity_analysis", {}).get("part_a_donors")
    nml, files, data, schemes, codes = read_inputs(args.source, args.de22, script_dir, only=main_donors)
    donors = [c for c, s in schemes.items() if s == "full" and c != CODE]
    li.check("run", "donors_with_full_coding", len(donors), 13, len(donors) == 13, " ".join(donors), severity="fail")
    if main_donors:
        same = sorted(main_donors) == sorted(donors)
        li.check("run", "donors_as_in_main_run", " ".join(donors), " ".join(sorted(main_donors)), same, severity="fail")
    d = data[CODE]
    groups = {g: m for g, m in li.day_groups(d).items() if m.sum() >= li.MIN_GROUP_DIARIES}
    timing["read_and_prepare"] = round(time.time() - t_stage, 1)

    # home activities (code 13 factor of part B): recomputed from the donors, compared with the main run
    acts, act_table = li.home_activities(data, donors, codes)
    flagged_now = set(codes[acts].tolist())
    flagged_main = set(home_acts_main.loc[home_acts_main.home_activity.astype(bool), "activity_code"].astype(int))
    li.check("run", "home_activities_as_in_main_run", len(flagged_now), len(flagged_main), flagged_now == flagged_main,
             "the donor files are those of the main run", severity="fail")

    # ------------------------------------------------------------------- 1. variants
    t_stage = time.time()
    tables = li.Tables(data)
    root = np.random.SeedSequence([li.SEED, STREAM_VARIANTS])
    var_rows, class_rows, home_rows, hours_rows = [], [], [], []
    for vname, spec in li.VARIANTS.items():
        model = None if spec["kind"] == "home" else tables.model(spec, donors, li.KAPPA)
        rng = np.random.default_rng(root.spawn(1)[0])
        imp = li.impute(spec, model, d, rng)
        o, bc, hm, hr = li.score(d["C"], imp, d["weight"], d["fold"], person=d["person"], count_respondents=True)
        var_rows.append(dict(country=CODE, variant=vname, diaries=d["n"], **o,
                             home_share_bias_pp=100 * float((hm.imputed - hm.observed).mean()),
                             changes_error_pct=100 * (o["changes_imputed"] / o["changes_observed"] - 1)))
        class_rows.append(bc.assign(country=CODE, variant=vname))
        home_rows.append(hm.assign(country=CODE, variant=vname))
        hours_rows.append(hr.assign(country=CODE, variant=vname))
        if vname == li.MAIN_VARIANT:
            li.imputation_checks(f"{CODE} main", d, imp)
            li.model_checks(f"{CODE} main", model, d)
    variants = pd.DataFrame(var_rows)
    variants.to_csv(out / "zve2022_variants.csv", index=False)
    by_class, _ = li.publish_by_class(pd.concat(class_rows), {CODE: "validation"})
    by_class.to_csv(out / "zve2022_by_class.csv", index=False)
    li.disclosure_checks(by_class, {CODE: "validation"})
    pd.concat(home_rows).to_csv(out / "zve2022_home_by_slot.csv", index=False)
    pd.concat(hours_rows).to_csv(out / "zve2022_hours_per_day.csv", index=False)
    show = variants.set_index("variant")[["accuracy", "kappa", "macro_f1", "home_share_mae_pp", "home_share_bias_pp",
                                          "hourly_tvd", "changes_error_pct"]]
    li.log(f"{CODE} imputed from the {len(donors)} donors, by variant:")
    print(show.round(3).to_string())
    timing["variants"] = round(time.time() - t_stage, 1)

    # ------------------------------------------------------------------- 2. repeated imputations
    t_stage = time.time()
    spec = li.VARIANTS[li.MAIN_VARIANT]
    model = tables.model(spec, donors, li.KAPPA)
    rng = np.random.default_rng(np.random.SeedSequence([li.SEED, STREAM_REPEAT]))
    lam0 = li.plain_scale(m_rep)
    so = li.weighted_slot_shares(d["C"], d["weight"], d["fold"])
    rec_hours = 24 * so.mean(0)
    scored = np.array([k for k in range(li.N_CLASSES) if d["fold"][k] == k and rec_hours[k] > 0])
    rec_changes = np.array([li.changes_per_diary(d["C"], d["fold"]).mean()])
    homes, hours, changes = [], [], []
    for _ in range(m_rep):
        imp = li.impute(spec, model, d, rng)
        si = li.weighted_slot_shares(imp, d["weight"], d["fold"])
        homes.append(si[:, li.HOME])
        hours.append(24 * si.mean(0)[scored])
        changes.append([li.changes_per_diary(imp, d["fold"]).mean()])
    mh, sh = li.mean_sd(homes)
    r_home = li.ratios(so[:, li.HOME], mh, sh)
    r_hours = li.ratios(rec_hours[scored], *li.mean_sd(hours))
    r_changes = li.ratios(rec_changes, *li.mean_sd(changes))
    rep = pd.DataFrame([dict(country=CODE, imputations=m_rep, scale_plain=lam0,
                             bias_home_pp=100 * float((mh - so[:, li.HOME]).mean()),
                             coverage_home=float(np.mean(r_home <= lam0)), width_home_pp=100 * 2 * lam0 * float(sh.mean()),
                             coverage_hours=float(np.mean(r_hours <= lam0)),
                             changes_recorded=float(rec_changes[0]), changes_imputed=float(np.mean(changes)),
                             changes_covered=bool(r_changes[0] <= lam0))])
    rep.to_csv(out / "zve2022_repeated_imputations.csv", index=False)
    li.log(f"plain 95% intervals from {m_rep} imputations contain the recorded share at home in "
           f"{100 * rep.coverage_home.iloc[0]:.0f}% of the slots")
    timing["repeated_imputations"] = round(time.time() - t_stage, 1)

    # ------------------------------------------------------------------- 3. method combinations (part A)
    t_stage = time.time()
    rec_hours_all, rec_changes_all, rec_home = recorded_values(d, groups)
    metric_rows, err_rows, err_home_rows = [], [], []
    errors22 = {}
    for unit in li.GSA_UNITS:
        for time_ in (False, True):
            for age in (False, True):
                for day in (False, True):
                    base = li.gsa_spec(unit, time_, age, day)
                    merged = tables.model(base, donors, li.KAPPA).tables
                    for alpha in alphas:
                        model_a = li.Model(base["levels"], merged, alpha)
                        for modal in (False, True):
                            rng = np.random.default_rng(np.random.SeedSequence([li.SEED, STREAM_A]))
                            imp = li.impute(dict(base, modal=modal), model_a, d, rng)
                            o, _, hm, hr = li.score(d["C"], imp, d["weight"], d["fold"])
                            key = dict(zip(FACTOR_A, combo_key(unit, time_, age, day, modal, alpha)))
                            metric_rows.append(dict(
                                key, country=CODE, accuracy=o["accuracy"], kappa=o["kappa"], macro_f1=o["macro_f1"],
                                home_share_mae_pp=o["home_share_mae_pp"],
                                home_share_bias_pp=100 * float((hm.imputed - hm.observed).mean()),
                                hourly_tvd=o["hourly_tvd"], changes_observed=o["changes_observed"],
                                changes_imputed=o["changes_imputed"],
                                changes_error_pct=100 * (o["changes_imputed"] / o["changes_observed"] - 1)))
                            e_hours = (hr.imputed_hours - hr.recorded_hours).to_numpy()
                            e_changes = float(o["changes_imputed"] - o["changes_observed"])
                            e_home = {g: li.weighted_slot_shares(imp[m], d["weight"][m], d["fold"])[:, li.HOME]
                                      - rec_home[g] for g, m in groups.items()}
                            errors22[combo_key(unit, time_, age, day, modal, alpha)] = (e_hours, e_changes, e_home)
                            for k, label in enumerate(LABELS):
                                err_rows.append(dict(key, output=label, error=e_hours[k]))
                            err_rows.append(dict(key, output="changes per diary", error=e_changes))
                            for g, e in e_home.items():
                                err_home_rows.append(dict(key, group=g, **dict(zip(li.CLOCK, e))))
            tables.cache.clear()          # the next unit and time setting needs other levels
            li.log(f"part A for {CODE}: unit {unit}, time {li.yes_no(time_)} done")
    metrics22 = pd.DataFrame(metric_rows)
    metrics22.to_csv(out / "zve2022_gsa_metrics.csv", index=False)
    errs22 = pd.DataFrame(err_rows)
    errs22.to_csv(out / "zve2022_transfer_errors.csv", index=False)
    pd.DataFrame(err_home_rows).pipe(li.alpha_text).to_csv(out / "zve2022_transfer_errors_home_by_slot.csv.gz",
                                                           index=False, compression="gzip", float_format="%.7g")

    # Germany's metrics and errors against the donors' under the same combination
    metrics = li.GSA_METRICS
    rank_rows = []
    don_m = loco_g[loco_g.country.isin(donors)]
    for _, row in metrics22.iterrows():
        sel = np.ones(len(don_m), bool)
        for f in FACTOR_A:
            sel &= (don_m[f] == row[f]).to_numpy()
        part = don_m[sel]
        for mtr in metrics:
            v = part[mtr].to_numpy(float)
            rank_rows.append(dict({f: row[f] for f in FACTOR_A}, quantity=mtr, kind="metric", germany_2022=row[mtr],
                                  donors_min=v.min(), donors_median=np.median(v), donors_max=v.max(),
                                  share_of_donors_below=float(np.mean(v < row[mtr]) + 0.5 * np.mean(v == row[mtr])),
                                  within_donor_range=bool(v.min() <= row[mtr] <= v.max())))
    don_e = err_tot[err_tot.country.isin(donors)]
    merged_e = errs22.merge(don_e, on=FACTOR_A + ["output"], suffixes=("_22", ""))
    for keys, part in merged_e.groupby(FACTOR_A + ["output"]):
        v, e22 = part.error.to_numpy(float), float(part.error_22.iloc[0])
        if keys[-1] == li.TRANSPORT_LABEL:
            continue
        rank_rows.append(dict(dict(zip(FACTOR_A, keys[:-1])), quantity=keys[-1], kind="error", germany_2022=e22,
                              donors_min=v.min(), donors_median=np.median(v), donors_max=v.max(),
                              share_of_donors_below=float(np.mean(v < e22) + 0.5 * np.mean(v == e22)),
                              within_donor_range=bool(v.min() <= e22 <= v.max())))
    ranks = pd.DataFrame(rank_rows)
    best = combos_a.loc[combos_a.pareto_optimal.astype(bool), FACTOR_A].assign(pareto_optimal=True)
    ranks = ranks.merge(best, on=FACTOR_A, how="left")
    ranks["pareto_optimal"] = ranks.pareto_optimal.fillna(False).astype(bool)
    ranks.to_csv(out / "zve2022_versus_donors.csv", index=False)
    # by slot: share of slots in which Germany's error lies within the donors' range
    clocks = list(li.CLOCK)
    don_h = err_home[err_home.country.isin(donors)]
    slot_rows = []
    for (unit, t_, a_, d_, asg, alpha), (_, _, e_home) in errors22.items():
        sel = ((don_h.unit == unit) & (don_h.time == t_) & (don_h.age == a_) & (don_h.day_type == d_)
               & (don_h.assignment == asg) & (don_h.alpha == alpha))
        for g, e in e_home.items():
            v = don_h.loc[sel & (don_h.group == g), clocks].to_numpy(float)
            if len(v) == 0:
                continue
            inside = (v.min(0) <= e) & (e <= v.max(0))
            slot_rows.append(dict(unit=unit, time=t_, age=a_, day_type=d_, assignment=asg, alpha=alpha, group=g,
                                  slots_within_donor_range=float(inside.mean()),
                                  share_of_donors_below_mean=float(((v < e).mean(0) + 0.5 * (v == e).mean(0)).mean())))
    pd.DataFrame(slot_rows).merge(best, on=FACTOR_A, how="left").fillna({"pareto_optimal": False}).to_csv(
        out / "zve2022_versus_donors_home_by_slot.csv", index=False)
    timing["part_a"] = round(time.time() - t_stage, 1)

    # ------------------------------------------------------------------- 4. uncertainty (part B)
    t_stage = time.time()
    tables13 = li.Tables(data, home_acts=acts)
    transfer = {}
    for keys, part in don_e.groupby(FACTOR_A + ["country"]):
        p = part.set_index("output").error
        transfer[keys] = (np.array([p.get(lab, 0.0) for lab in LABELS]), float(p.get("changes per diary", 0.0)))
    transfer_home = {}
    for keys, part in don_h.groupby(FACTOR_A + ["country"]):
        transfer_home[keys] = {g: row[clocks].to_numpy(float) for g, row in part.set_index("group").iterrows()}
    values, values_home, method_key = {o_: [] for o_ in OUTPUTS_B}, {g: [] for g in groups}, []
    n_imp = 0
    for unit in li.GSA_UNITS:
        for time_ in (False, True):
            for age in (False, True):
                for day in (False, True):
                    spec_b = li.gsa_spec(unit, time_, age, day)
                    for code13 in (False, True):
                        merged = (tables13 if code13 else tables).model(spec_b, donors, li.KAPPA).tables
                        for alpha in alphas:
                            model_b = li.Model(spec_b["levels"], merged, alpha)
                            for modal in (False, True):
                                res = None
                                ck = combo_key(unit, time_, age, day, modal, alpha)
                                for rep_ in range(n_draws):
                                    if res is None or not modal:
                                        rng = np.random.default_rng(np.random.SeedSequence([li.SEED, STREAM_B, rep_]))
                                        imp = li.impute(dict(spec_b, modal=modal), model_b, d, rng)
                                        summs = li.group_summaries(imp, d, groups)
                                        res = (summs["all"]["hours"], summs["all"]["changes"],
                                               {g: s_["shares"][:, li.HOME] for g, s_ in summs.items()})
                                        n_imp += 1
                                    for held in donors:
                                        e_hours, e_changes = transfer[ck + (held,)]
                                        h = np.maximum(res[0] - e_hours, 0.0)
                                        for k, lab in enumerate(li.CLASS_LABELS):
                                            values[lab].append(h[k])
                                        values["changes per diary"].append(max(res[1] - e_changes, 0.0))
                                        for g in groups:
                                            values_home[g].append(np.clip(res[2][g] - transfer_home[ck + (held,)][g], 0, 1))
                                        method_key.append(ck)
            tables.cache.clear()
            tables13.cache.clear()
            li.log(f"part B for {CODE}: unit {unit}, time {li.yes_no(time_)} done ({n_imp} imputations)")
    n_b = len(method_key)
    expected_b = len(combos_a) * 2 * n_draws * len(donors) if not args.quick else n_b
    li.check("run", "part_b_combinations", n_b, expected_b, n_b == expected_b, severity="fail")
    pareto_set = set(map(tuple, best[FACTOR_A].itertuples(index=False, name=None)))
    in_best = np.array([k in pareto_set for k in method_key])
    li.check("run", "part_b_combinations_with_pareto_optimal_method", int(in_best.sum()), ">0", in_best.any(),
             severity="fail")

    def summary(v, recorded):
        v = np.asarray(v, float)
        q = np.quantile(v, [0.05, 0.5, 0.95])
        return dict(recorded=recorded, q05=q[0], q50=q[1], q95=q[2], combinations=len(v),
                    recorded_percentile=float(np.mean(v < recorded) + 0.5 * np.mean(v == recorded)),
                    within_90=bool(q[0] <= recorded <= q[2]))

    cov_rows = []
    for name, sel in (("all", np.ones(n_b, bool)), ("pareto", in_best)):
        for k, lab in enumerate(li.CLASS_LABELS):
            cov_rows.append(dict(combinations_of=name, output=lab, **summary(np.array(values[lab])[sel], rec_hours_all[k])))
        cov_rows.append(dict(combinations_of=name, output="changes per diary",
                             **summary(np.array(values["changes per diary"])[sel], rec_changes_all)))
    coverage = pd.DataFrame(cov_rows)
    coverage.to_csv(out / "zve2022_coverage.csv", index=False)
    slot_cov, slot_sum = [], []
    for g in groups:
        H = np.vstack(values_home[g])
        for name, sel in (("all", np.ones(n_b, bool)), ("pareto", in_best)):
            q = np.quantile(H[sel], [0.05, 0.5, 0.95], axis=0)
            pct = (H[sel] < rec_home[g]).mean(0) + 0.5 * (H[sel] == rec_home[g]).mean(0)
            inside = (q[0] <= rec_home[g]) & (rec_home[g] <= q[2])
            slot_cov.append(pd.DataFrame(dict(group=g, combinations_of=name, slot=np.arange(li.SLOTS), clock=li.CLOCK,
                                              recorded=rec_home[g], q05=q[0], q50=q[1], q95=q[2],
                                              recorded_percentile=pct, within_90=inside)))
            slot_sum.append(dict(group=g, combinations_of=name, slots_within_90=float(inside.mean()),
                                 mean_width_pp=100 * float((q[2] - q[0]).mean())))
        del H
    pd.concat(slot_cov).to_csv(out / "zve2022_coverage_home_by_slot.csv", index=False)
    slot_summary = pd.DataFrame(slot_sum)
    slot_summary.to_csv(out / "zve2022_coverage_home_by_slot_summary.csv", index=False)
    li.log("recorded 2022 values against the 90% ranges of part B:")
    print(coverage[["combinations_of", "output", "recorded", "q05", "q50", "q95", "within_90"]].round(3).to_string(index=False))
    print(slot_summary.round(3).to_string(index=False))
    timing["part_b"] = round(time.time() - t_stage, 1)

    # ------------------------------------------------------------------- settings, checks, README
    rewritten_last = ("zve2022_checks.csv", "run_settings.json", "README.txt")
    stale = sorted(f.name for f in out.iterdir()
                   if f.is_file() and f.name not in rewritten_last and f.stat().st_mtime < run_start - 2)
    li.check("run", "files_of_earlier_runs_in_output_folder", len(stale), 0, not stale,
             "kept out of the export: " + ", ".join(stale) if stale else "")
    checks = pd.DataFrame(li.CHECKS)
    checks.to_csv(out / "zve2022_checks.csv", index=False)
    settings = dict(version=f"{VERSION}", pipeline_version=li.VERSION, script="zve2022_holdout.py",
                    main_run_results=res_dir.name, main_run_version=main_settings.get("version"),
                    input_files={**{c: files[c].name for c in donors}, CODE: Path(args.de22).name},
                    donors=donors, diaries_2022=int(d["n"]), repeat_imputations=m_rep, draws=n_draws,
                    alphas=list(alphas), part_b_imputations=n_imp, part_b_combinations=n_b,
                    seed=li.SEED, streams=dict(variants=STREAM_VARIANTS, repeated=STREAM_REPEAT, part_a=STREAM_A,
                                               part_b=STREAM_B),
                    min_respondents_per_place_score=li.MIN_RESPONDENTS, files_of_earlier_runs=stale,
                    checks_warnings=int((checks.status == "warn").sum()),
                    checks_failures=int((checks.status == "fail").sum()), timing_seconds=timing)
    (out / "run_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (out / "README.txt").write_text(README)
    li.log(f"checks: {len(checks)} run, {settings['checks_warnings']} warnings, {settings['checks_failures']} failures")
    li.log(f"done; aggregate outputs in {out}")


README = """Hold-out test on the German time use survey of 2022 (zve2022_holdout.py)

Every file here is an aggregate table and can be shared once the data providers' rules are met;
per-place scores resting on fewer than 50 respondents are withheld. The 2022 diaries (DE22) are
imputed from the 13 HETUS 2010 donors and compared with the places they record.

zve2022_variants.csv          seven evaluation metrics for the nine variants of the main run
zve2022_by_class.csv          recall, precision and F1 by place (withheld below 50 respondents)
zve2022_home_by_slot.csv      recorded and imputed share at home by slot, by variant
zve2022_hours_per_day.csv     recorded and imputed hours per day by place, by variant
zve2022_repeated_imputations.csv  coverage of plain 95% intervals from repeated imputations
zve2022_gsa_metrics.csv       seven metrics for each of the 144 method combinations
zve2022_transfer_errors.csv   errors (imputed minus recorded) in hours and changes, per combination
zve2022_transfer_errors_home_by_slot.csv.gz  errors in the share at home by slot, per combination
zve2022_versus_donors.csv     Germany's metrics and errors against the 13 donors' under the same combination
zve2022_versus_donors_home_by_slot.csv  share of slots where Germany's error lies within the donors' range
zve2022_coverage.csv          recorded 2022 values against the 90% ranges of part B (all, Pareto-optimal methods)
zve2022_coverage_home_by_slot.csv   the same for the share at home in each slot
zve2022_coverage_home_by_slot_summary.csv  share of slots whose recorded value lies within the range
zve2022_checks.csv            checks of the inputs and of the run
run_settings.json             versions, inputs (file names only), settings and run times
"""


if __name__ == "__main__":
    main()
