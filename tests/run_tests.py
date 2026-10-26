#!/usr/bin/env python3
"""End-to-end test on synthetic data. Needs no restricted data.

Checks the exact Sobol' indices against analytic values (tests/test_indices.py) and the
disclosure control of the per-place scores (tests/test_disclosure.py), then writes
synthetic HETUS-like country files to a temporary folder, runs the full pipeline on them,
exports the aggregate outputs and builds the figures and tables of the manuscript from the
export, and checks that
  1. the pipeline finishes and none of its checks fails (among them, that no published
     per-place score rests on fewer than 50 respondents),
  2. every aggregate output is written, including the global sensitivity analysis,
  3. the main variant beats placing everyone at home on the synthetic donors, and treating
     code 13 as missing for home activities changes the German estimates,
  4. the export holds no microdata, no local folder names and no file that an earlier run
     left in the output folder,
  5. every figure and table of the manuscript is built,
  6. synthetic files with the content of the German time use survey of 2022 are converted to the
     HETUS layout (zve2022_to_hetus.py), and the hold-out test (zve2022_holdout.py) imputes them
     from the synthetic donors with the exported results of the main run, without failed checks,
     writes every output, and its export holds no microdata.
Takes about seven minutes.

Usage:
    python tests/run_tests.py            # scale 0.05 of the real diary counts
    python tests/run_tests.py --keep     # keep the temporary folder for inspection
    python tests/run_tests.py --scale 0.1
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGGREGATES = ["checks.csv", "run_settings.json", "README.txt", "loco_overall.csv", "loco_summary.csv",
              "loco_by_class.csv", "loco_by_class_means.csv", "loco_home_by_slot.csv", "loco_hours_per_day.csv",
              "loco_repeated_imputations.csv", "loco_repeated_imputations_summary.csv",
              "loco_intervals_home_by_slot.csv",
              "germany_shares_by_slot.csv", "germany_hours_per_day.csv", "germany_changes_per_diary.csv",
              "germany_hours_per_imputation.csv", "home_activities.csv",
              "gsa_loco.csv", "gsa_loco_indices.csv", "gsa_loco_combinations.csv", "gsa_transfer_errors.csv",
              "gsa_transfer_errors_home_by_slot.csv.gz", "gsa_germany_universes.csv",
              "gsa_germany_home_by_slot.csv.gz", "gsa_germany_combinations.csv", "gsa_germany_indices.csv",
              "gsa_germany_indices_home_by_slot.csv", "gsa_germany_uncertainty.csv",
              "gsa_germany_uncertainty_home_by_slot.csv", "gsa_germany_uncertainty_pareto.csv",
              "gsa_germany_uncertainty_home_by_slot_pareto.csv"]
FIGURES = ["fig_variants", "fig_home_error", "fig_places", "fig_germany_tempogram", "fig_germany_home",
           "fig_gsa_methods", "fig_gsa_germany", "fig_gsa_home", "fig_pareto"]
TABLES = ["tab_variants", "tab_countries", "tab_germany", "tab_gsa_methods", "tab_gsa_germany"]
ZVE_OUTPUTS = ["zve2022_variants.csv", "zve2022_by_class.csv", "zve2022_home_by_slot.csv", "zve2022_hours_per_day.csv",
               "zve2022_repeated_imputations.csv", "zve2022_gsa_metrics.csv", "zve2022_transfer_errors.csv",
               "zve2022_transfer_errors_home_by_slot.csv.gz", "zve2022_versus_donors.csv",
               "zve2022_versus_donors_home_by_slot.csv", "zve2022_coverage.csv", "zve2022_coverage_home_by_slot.csv",
               "zve2022_coverage_home_by_slot_summary.csv", "zve2022_checks.csv", "run_settings.json", "README.txt"]
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", flush=True)


def run(args, log):
    with open(log, "w") as fh:
        done = subprocess.run([sys.executable, *map(str, args)], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return done.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="hetus_test_"))
    print(f"working in {tmp}", flush=True)
    try:
        ok = run([ROOT / "tests" / "test_indices.py"], tmp / "test_indices.log")
        report("Sobol' indices match analytic values", ok, "" if ok else f"see {tmp / 'test_indices.log'}")
        ok = run([ROOT / "tests" / "test_disclosure.py"], tmp / "test_disclosure.log")
        report("disclosure control of per-place scores", ok, "" if ok else f"see {tmp / 'test_disclosure.log'}")
        ok = run([ROOT / "tests" / "make_synthetic.py", tmp / "synthetic", args.scale], tmp / "make_synthetic.log")
        report("synthetic files written", ok)
        out = tmp / "restricted" / "imputation_out"
        out.mkdir(parents=True)          # an output of an earlier version that this one no longer writes
        stale = out / "loco_calibration.csv"
        stale.write_text("left by an earlier run\n")
        os.utime(stale, (stale.stat().st_atime - 3600, stale.stat().st_mtime - 3600))
        ok = run([ROOT / "src" / "location_imputation.py", tmp / "synthetic" / "TUS_Eurostat.nml", "--out", out],
                 tmp / "pipeline.log")
        report("pipeline finished", ok, "" if ok else f"see {tmp / 'pipeline.log'}")
        if not ok:
            return
        checks = pd.read_csv(out / "checks.csv")
        failed = checks[checks.status == "fail"]
        report("no failed checks", failed.empty, "; ".join(failed.scope + " " + failed.check))
        missing = [f for f in AGGREGATES if not (out / f).is_file()]
        report("all aggregate outputs written", not missing, ", ".join(missing))
        summary = pd.read_csv(out / "loco_summary.csv").set_index(["coding", "variant"])
        full = summary.loc["full"]
        main_acc, home_acc = full.loc["markov_time_age", "accuracy"], full.loc["home", "accuracy"]
        report("main variant beats all-at-home", bool(main_acc > home_acc),
               f"accuracy {main_acc:.3f} against {home_acc:.3f}")
        excluded = checks.loc[checks.check == "code13_home_activity_episodes_excluded", "value"].astype(int).sum()
        idx = pd.read_csv(out / "gsa_germany_indices.csv").set_index(["output", "factor"])
        s13 = float(idx.loc[("home", "code_13"), "first_order"])
        report("code 13 for home activities exercised", bool(excluded > 0 and s13 > 0),
               f"{excluded} donor episodes excluded, first-order index for time at home {s13:.4f}")

        export = tmp / "results"
        ok = run([ROOT / "src" / "export_results.py", out, export], tmp / "export.log")
        report("export finished", ok)
        exported = sorted(p.name for p in export.iterdir())
        leaked = [n for n in exported if n.startswith("PRIVATE_") or n.endswith(".npz")]
        report("export holds no microdata", ok and not leaked, ", ".join(leaked))
        lost = [f for f in AGGREGATES if f not in exported]
        report("export holds every aggregate output", not lost, ", ".join(lost))
        flagged = checks.loc[checks.check == "files_of_earlier_runs_in_output_folder", "value"].astype(int).sum()
        report("outputs of earlier runs not exported", flagged == 1 and stale.name not in exported,
               f"{flagged} flagged in checks.csv")
        files = json.loads((export / "run_settings.json").read_text()).get("files", {})
        absolute = [p for p in files.values() if Path(p).is_absolute() or str(tmp) in p]
        def text_of(path):
            if path.name.endswith(".gz"):
                return gzip.decompress(path.read_bytes()).decode("utf-8", errors="ignore")
            return path.read_text(errors="ignore")
        report("export holds no local folder names", not absolute and not any(
            str(tmp) in text_of(export / n) for n in exported), ", ".join(absolute))

        built = tmp / "manuscript"
        ok = run([ROOT / "src" / "make_results_figures.py", export, built], tmp / "figures.log")
        missing = [f"{f}.pdf" for f in FIGURES if not (built / "figures" / f"{f}.pdf").is_file()]
        missing += [f"{t}.tex" for t in TABLES if not (built / "tables" / f"{t}.tex").is_file()]
        report("figures and tables built", ok and not missing, ", ".join(missing))

        # hold-out test on the diaries of 2022 (synthetic files, invented variable names)
        zdir, de22 = tmp / "synthetic" / "zve2022", tmp / "restricted" / "DE22.csv"
        ok = run([ROOT / "tests" / "make_synthetic_zve2022.py", tmp / "synthetic", args.scale], tmp / "make_zve2022.log")
        ok = ok and run([ROOT / "src" / "zve2022_to_hetus.py", "--slots", zdir / "slots.csv", "--persons",
                         zdir / "persons.csv", "--columns", zdir / "columns.json", "--out", de22],
                        tmp / "zve2022_convert.log")
        summary = pd.read_csv(de22.with_name("conversion_summary.csv")) if ok else None
        unconverted = (float(summary.set_index("item").loc[
            "share of slots with an activity that could not be converted", "value"]) if summary is not None else 1)
        report("2022 diaries converted to the HETUS layout", ok and unconverted == 0,
               "" if ok else f"see {tmp / 'zve2022_convert.log'}")
        zout = tmp / "restricted" / "zve2022_out"
        ok = ok and run([ROOT / "src" / "zve2022_holdout.py", de22, tmp / "synthetic" / "TUS_Eurostat.nml",
                         "--results", export, "--out", zout], tmp / "zve2022_holdout.log")
        zchecks = pd.read_csv(zout / "zve2022_checks.csv") if ok else None
        zfailed = zchecks[zchecks.status == "fail"] if zchecks is not None else pd.DataFrame()
        report("2022 hold-out test finished without failed checks", ok and zfailed.empty,
               "; ".join(zfailed.scope + " " + zfailed.check) if ok else f"see {tmp / 'zve2022_holdout.log'}")
        missing = [f for f in ZVE_OUTPUTS if not (zout / f).is_file()]
        report("2022 hold-out outputs written", ok and not missing, ", ".join(missing))
        if ok:
            var = pd.read_csv(zout / "zve2022_variants.csv").set_index("variant")
            main_acc, home_acc = var.loc["markov_time_age", "accuracy"], var.loc["home", "accuracy"]
            report("2022: main variant beats all-at-home", bool(main_acc > home_acc),
                   f"accuracy {main_acc:.3f} against {home_acc:.3f}")
        zexport = tmp / "results_zve2022"
        ok = ok and run([ROOT / "src" / "export_results.py", zout, zexport], tmp / "zve2022_export.log")
        zexported = sorted(p.name for p in zexport.iterdir()) if ok else []
        leaked = [n for n in zexported if n.startswith("PRIVATE_") or n.endswith(".npz") or n == de22.name]
        local = [n for n in zexported if str(tmp) in text_of(zexport / n)]
        report("2022 export holds no microdata and no local folder names", ok and not leaked and not local,
               ", ".join(leaked + local))
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(RESULTS)} of {len(RESULTS)} tests passed")


if __name__ == "__main__":
    main()
    sys.exit(0 if RESULTS and all(RESULTS) else 1)
