#!/usr/bin/env python3
"""Imputation of activity locations in the HETUS 2010 round, with leave-one-country-out tests
and a global sensitivity analysis.

Version 4.1. Every slot with a recorded mode of transport counts as transport, whatever its main
activity (reading on a train is time in transport, as the trip itself is), and is kept as recorded,
like trips, in every country; the eight location classes imputed are the places of WHEREP codes 1
to 7 and 13. Per-place scores of a country (loco_by_class.csv) and location shares of an activity
(home_activities.csv) are withheld where fewer than MIN_RESPONDENTS respondents stand behind them
(Eurostat's publication guidelines for the scientific use files treat HETUS as EU-SILC: nothing on
fewer than 20 respondents, results on 20 to 49 flagged).

What it does
  1. Reads every country file named in TUS_Eurostat.nml (found automatically, as in
     hetus_inventory.py), upper-cases column names and puts Austria's 15-minute diary
     on the 10-minute grid. Each file is checked against the SUF documentation (diaries,
     slot columns, codes, weights, age, day of the week); results go to checks.csv.
  2. Splits each diary into activity episodes (runs of the same activity code, also split
     where transport starts or ends) and marks the transport episodes, which are never imputed.
  3. Estimates the probability of each place given activity, time of day, age band,
     the previous place and whether transport came in between. Sparse cells are shrunk
     toward coarser cells (hierarchical Dirichlet smoothing).
  4. Holds out each donor country in turn, trains the model on the other 12 and scores
     several variants against what the held-out diaries recorded (leave-one-country-out).
  5. Imputes each held-out country REPEAT_IMPUTATIONS times with the main variant and reports
     how often plain 95% intervals from these imputations contain the recorded values.
  6. Imputes Germany N_IMPUTATIONS times with the main variant estimated on all 13 donors.
  7. Selects the activities that donors carry out almost only at home, for which code 13 is
     treated as missing in one level of a factor of the global sensitivity analysis.
  8. Global sensitivity analysis. Every factor is discrete and uniform over its levels, so the
     full factorial is enumerated and the first-order and total Sobol' indices are computed
     exactly (functional ANOVA, no Monte Carlo error).
     Part A, method choices: unit of imputation (slot, episode, Markov chain of episodes),
     time of day, age, day type, assignment (random draw or most likely class) and the
     smoothing pseudo-count alpha (1, 5, 20), 144 combinations, each trained on 12 donors and
     validated on the 13th, for every donor. Outputs: the evaluation metrics averaged over the
     donors.
     Part B, German estimates: the 144 method combinations of part A, code 13 for home
     activities (an assumption about the coding), the random draws (GSA_DRAWS streams of random
     numbers) and the transfer error, whose 13 levels are the errors made for each donor under
     the same method combination (Germany treated as exchangeable with the donors). Outputs:
     German hours per day by place, changes per diary and the share at home in each slot (all
     days, weekdays, weekend days), each minus the transfer error, summarised over all
     combinations and over those whose method is Pareto-optimal in part A.

Usage, from anywhere inside the project folder:
    python location_imputation.py              # full run
    python location_imputation.py --quick      # fewer variants and imputations, as a first check
    python location_imputation.py --skip-gsa   # everything except the global sensitivity analysis
    python location_imputation.py /path/to/TUS_Eurostat.nml
Outputs go to imputation_out/ next to the script. README.txt there says which files are
aggregates that can be shared and which are microdata that must stay on your machine.
"""
import argparse
import json
import math
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "4.1 (25 October 2026)"

# --------------------------------------------------------------------------- settings
TARGET = "DE"
N_IMPUTATIONS = 20               # German imputations with the fixed donor pool, as in version 1
KAPPA = 5.0                      # pseudo-counts given to the coarser cell at each level (alpha in the paper)
SEED = 20261019
MIN_GROUP_DIARIES = 50           # German aggregates are written only for groups this large
MIN_CELL = 10                    # per-place scores based on fewer slots are blanked
MIN_RESPONDENTS = 50             # per-place scores of a country are published only if at least this many
                                 # respondents were recorded at (recall) or imputed to (precision) the place.
                                 # Eurostat's publication guidelines for the SUF (HETUS as EU-SILC): nothing
                                 # on fewer than 20 respondents, 20 to 49 only flagged; 50 avoids the flags.

# Repeated imputations of the held-out countries (step 5)
REPEAT_IMPUTATIONS = 20          # imputations per held-out country for the plain 95% intervals

# Home activities (step 7), for which code 13 is treated as missing in one level of a factor of part B
HOME_ACTIVITY_SHARE = 0.95       # 'home activities': at least this share of donor slots with a
HOME_ACTIVITY_MIN_SLOTS = 1000   # specified place (codes 1 to 7) at home, and at least this many such slots
HOME_ACTIVITY_MAX_CODE13 = 0.15  # and code 13 in at most this share of their slots (more often, code 13 is
                                 # a real place without a code of its own, such as a gym)

# Global sensitivity analysis (version 3): levels of each factor, all equally likely
GSA_UNITS = ("slot", "episode", "markov")   # unit of imputation and sequence model
GSA_ALPHAS = (1.0, 5.0, 20.0)               # smoothing pseudo-count, about log-uniform
GSA_GROUPS = ("all", "weekday", "weekend")   # groups for the German share at home by slot
GSA_DRAWS = 10                               # streams of random numbers for the German imputations of part B
GSA_METRICS = ["accuracy", "kappa", "macro_f1", "home_share_mae_pp", "home_share_bias_pp",
               "hourly_tvd", "changes_error_pct"]
GSA_DIRECTIONS = dict(accuracy=1, kappa=1, macro_f1=1, home_share_mae_pp=-1, home_share_bias_pp=0,
                      hourly_tvd=-1, changes_error_pct=0)   # 1 higher is better, -1 lower, 0 closer to zero

# Records (diary days) per country in Annex 1 of the HETUS 2010 SUF documentation
ANNEX1_RECORDS = {"AT": 8234, "BE": 11118, "DE": 27143, "EE": 9946, "EL": 14274,
                  "ES": 19295, "FI": 7480, "FR": 27903, "HU": 8391, "IT": 38234,
                  "LU": 4162, "NL": 14035, "NO": 7882, "PL": 78759, "RO": 56514,
                  "RS": 6835, "UK": 16118}

# WHEREP codes are Eurobase aggregates of the 17 ACL 2008 codes (HETUS 2018 Guidelines,
# 2020 re-edition, Annex VII/2; confirmed by Eurostat Unit F4): 1 home, 2 weekend home,
# 3 workplace or school, 4 other people's home, 5 restaurant, 6 shops, 7 hotel or camping,
# 8 on foot, 9 bicycle, 10 car, moped, motorcycle or motorboat, 11 public transport,
# 12 unspecified or other private mode, 13 other specified OR unspecified location.
PLACE_CODES = [1, 2, 3, 4, 5, 6, 7, 13]
TRANSPORT_CODES = [8, 9, 10, 11, 12]
TRAVEL_ACTIVITY = (900, 990)               # activity codes 900-989 are trips
CLASS_LABELS = ["home", "second home", "workplace or school", "other people's home",
                "restaurant, cafe or pub", "shops and services", "hotel or camping",
                "other or unspecified place"]
N_CLASSES = len(CLASS_LABELS)
HOME, OTHER = 0, PLACE_CODES.index(13)
TRANSPORT_LABEL = "transport"    # trips and any other activity with a recorded mode of transport
START = N_CLASSES                # 'previous place' of the first episode of a diary
TRAVEL_SLOT, UNKNOWN = -2, -1    # slot markers in class arrays: transport, place not known
OUT_TRAVEL_NO_MODE = 30          # output code of a trip without a recorded mode (not in the HETUS list)
AGE_TOP = 14                     # INC2 band 15 merged into 14 (Germany has no band 15)
SLOTS = 144

LEVELS_SLOT_TIME = [("a",), ("a", "t")]
LEVELS_SLOT_TIME_AGE = LEVELS_SLOT_TIME + [("a", "t", "g")]
LEVELS_EP_TIME_AGE = [("a",), ("a", "h"), ("a", "h", "g")]
LEVELS_MK_TIME = [("a",), ("a", "h"), ("a", "h", "p", "r")]
LEVELS_MK_TIME_AGE = LEVELS_MK_TIME + [("a", "h", "p", "r", "g")]
LEVELS_MK_TIME_AGE_DAY = LEVELS_MK_TIME_AGE + [("a", "h", "p", "r", "g", "w")]

VARIANTS = {
    "home": dict(kind="home"),
    "activity_modal": dict(kind="episode", levels=[("a",)], modal=True),
    "slot_time": dict(kind="slot", levels=LEVELS_SLOT_TIME),
    "slot_time_age": dict(kind="slot", levels=LEVELS_SLOT_TIME_AGE),
    "episode_time_age": dict(kind="episode", levels=LEVELS_EP_TIME_AGE),
    "markov_time": dict(kind="markov", levels=LEVELS_MK_TIME),
    "markov_time_age": dict(kind="markov", levels=LEVELS_MK_TIME_AGE),
    "markov_time_age_day": dict(kind="markov", levels=LEVELS_MK_TIME_AGE_DAY),
    "markov_time_age_modal": dict(kind="markov", levels=LEVELS_MK_TIME_AGE, modal=True),
}
MAIN_VARIANT = "markov_time_age"
QUICK_VARIANTS = ["home", "activity_modal", "slot_time_age", "episode_time_age", MAIN_VARIANT]

NML_COUNTRY = re.compile(r"^\s*TUS_([A-Za-z]{2})_data\s*=\s*['\"](.+?)['\"]", re.MULTILINE)
SLOT_COLUMN = re.compile(r"^(MACT|WHEREP)(\d+)$")


def log(msg, _t0=[time.time()]):
    print(f"[{time.time() - _t0[0]:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- checks
CHECKS = []


def check(scope, name, value, expected, ok, note="", severity="warn"):
    """Record one check. A failed check is a 'warn' for data issues and a 'fail' for
    anything that points to a bug; neither stops the run."""
    CHECKS.append(dict(scope=scope, check=name, value=value, expected=expected,
                       status="ok" if ok else severity, note=note))
    if not ok:
        log(f"CHECK {severity.upper()}: {scope} {name} = {value} (expected {expected}) {note}")


# --------------------------------------------------------------------------- inputs
def locate_nml(starts):
    for start in starts:
        for folder in [start, *start.parents][:6]:
            for candidate in (folder / "TUS_Eurostat.nml", folder / "Data" / "TUS_Eurostat.nml"):
                if candidate.is_file():
                    return candidate.resolve()
    return None


def resolve(written, bases):
    written = Path(written)
    if written.is_absolute():
        return written if written.is_file() else None
    for base in bases:
        candidate = Path(os.path.normpath(base / written))
        if candidate.is_file():
            return candidate
    for base in bases:
        candidate = base / written.name
        if candidate.is_file():
            return candidate
    return None


def country_files(source, script_dir):
    nml = Path(source).expanduser().resolve() if source else None
    if nml is None or not nml.is_file():
        nml = locate_nml([script_dir, Path.cwd()])
    if nml is None:
        raise SystemExit("TUS_Eurostat.nml not found. Pass its path: python location_imputation.py /path/to/TUS_Eurostat.nml")
    bases = [Path.cwd(), script_dir, nml.parent]
    files = {}
    for code, written in NML_COUNTRY.findall(nml.read_text()):
        path = resolve(written, bases)
        if path is None:
            raise SystemExit(f"File for {code.upper()} not found ({written}). Run hetus_inventory.py to see where files are.")
        files[code.upper()] = path
    return nml, files


# --------------------------------------------------------------------------- loading
def read_country(path):
    """Activity and location arrays (n x 144) plus the diary-level variables used here."""
    header = list(pd.read_csv(path, nrows=0).columns)
    by_upper = {}
    for col in header:
        by_upper.setdefault(col.strip().upper(), col)
    slots = {"MACT": [], "WHEREP": []}
    for upper in by_upper:
        match = SLOT_COLUMN.match(upper)
        if match:
            slots[match.group(1)].append((int(match.group(2)), upper))
    act_cols = [u for _, u in sorted(slots["MACT"])]
    loc_cols = [u for _, u in sorted(slots["WHEREP"])]
    extra = [c for c in ("INC2", "DDV1", "WGHT1", "HID", "PID", "DIARY") if c in by_upper]
    usecols = [by_upper[c] for c in act_cols + loc_cols + extra]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=20000, low_memory=False):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        parts.append(chunk.apply(pd.to_numeric, errors="coerce").astype("float32"))
    df = pd.concat(parts, ignore_index=True)
    A = np.nan_to_num(df[act_cols].to_numpy(), nan=-1).astype(np.int16)
    L = np.nan_to_num(df[loc_cols].to_numpy(), nan=-1).astype(np.int16)
    stats = dict(slot_columns_activity=len(act_cols), slot_columns_location=len(loc_cols),
                 activity_missing_share=float((A < 0).mean()),
                 location_negative_share=float((L < 0).mean()),
                 location_unexpected_share=float(((L == 0) | (L > 13)).mean()),
                 covariates_missing=",".join(c for c in ("INC2", "DDV1", "WGHT1") if c not in by_upper))
    if A.shape[1] != SLOTS:  # Austria: 96 fifteen-minute slots -> 144 ten-minute slots
        width = 1440 // A.shape[1]
        source_slot = (10 * np.arange(SLOTS) + 5) // width   # slot covering most of each 10 minutes
        A, L = A[:, source_slot], L[:, source_slot]
    age = np.nan_to_num(df["INC2"].to_numpy(), nan=0).astype(np.int16) if "INC2" in df else np.zeros(len(df), np.int16)
    age = np.where((age >= 1) & (age <= 15), np.minimum(age, AGE_TOP), 0).astype(np.int8)
    ddv1 = np.nan_to_num(df["DDV1"].to_numpy(), nan=0) if "DDV1" in df else np.zeros(len(df))
    weekend = np.where(np.isin(ddv1, [1, 7]), 1, np.where((ddv1 >= 2) & (ddv1 <= 6), 0, 2)).astype(np.int8)
    weight = np.nan_to_num(df["WGHT1"].to_numpy(dtype=float), nan=0.0) if "WGHT1" in df else np.ones(len(df))
    stats["weight_nonpositive"] = int((weight <= 0).sum())
    weight = np.where(weight > 0, weight, 0.0)
    if weight.sum() == 0:
        weight = np.ones(len(df))
    stats["age_missing_share"] = float((age == 0).mean())
    stats["day_missing_share"] = float((weekend == 2).mean())
    ids = df[[c for c in ("HID", "PID", "DIARY") if c in df]].astype("Int64")
    return dict(A=A, L=L, age=age, weekend=weekend, weight=weight, ids=ids, stats=stats)


def slot_classes(A, L):
    """Class of every slot: place index, TRAVEL_SLOT for transport, UNKNOWN otherwise.
    Transport is a trip (activity codes 900 to 989) or any other activity with a recorded mode
    of transport, such as reading on a train (version 4)."""
    C = np.full(A.shape, UNKNOWN, np.int8)
    for k, code in enumerate(PLACE_CODES):
        C[L == code] = k
    C[np.isin(L, TRANSPORT_CODES)] = TRAVEL_SLOT
    C[(A >= TRAVEL_ACTIVITY[0]) & (A < TRAVEL_ACTIVITY[1])] = TRAVEL_SLOT
    return C


def coding_scheme(C):
    """Place classes a country records; places it does not code are folded into code 13.

    A place counts as not coded when it holds under 0.01% of the country's recorded
    slots, or fewer than ten (the rarest coded place in any donor, second homes in
    Romania, is about 0.03%)."""
    present = np.zeros(N_CLASSES, bool)
    vals, counts = np.unique(C[C >= 0], return_counts=True)
    threshold = max(10, 1e-4 * counts.sum()) if len(counts) else 10
    present[vals] = counts >= threshold
    fold = np.arange(N_CLASSES)
    if not present[HOME]:
        return "none", fold, present
    for k in range(N_CLASSES):
        if k not in (HOME, OTHER) and not present[k]:
            fold[k] = OTHER
    return ("full" if (fold == np.arange(N_CLASSES)).all() else "partial"), fold, present


def data_checks(code, r, C, scheme, fold):
    """Checks of one country file against the SUF documentation and the coding expected."""
    n, st = len(C), r["stats"]
    expected = ANNEX1_RECORDS.get(code)
    check(code, "diaries_vs_suf_annex1", n, expected, expected is None or n == expected,
          "" if expected is None or n == expected else f"file holds {n - expected:+d} diaries; re-export before final results")
    grid = 96 if code == "AT" else SLOTS
    check(code, "activity_slot_columns", st["slot_columns_activity"], grid, st["slot_columns_activity"] == grid)
    check(code, "location_slot_columns", st["slot_columns_location"], grid, st["slot_columns_location"] == grid)
    check(code, "activity_code_missing_share", round(st["activity_missing_share"], 6), 0, st["activity_missing_share"] == 0)
    check(code, "location_code_unexpected_share", round(st["location_unexpected_share"], 6), 0,
          st["location_unexpected_share"] == 0, "codes outside 1 to 13 and the negative SUF codes")
    check(code, "location_code_negative_share", round(st["location_negative_share"], 6), "<0.01",
          st["location_negative_share"] < 0.01, "slots with no answer or not applicable")
    check(code, "weights_nonpositive", st["weight_nonpositive"], 0, st["weight_nonpositive"] == 0)
    check(code, "age_band_missing_share", round(st["age_missing_share"], 6), 0, st["age_missing_share"] == 0)
    check(code, "day_of_week_missing_share", round(st["day_missing_share"], 6), 0, st["day_missing_share"] == 0)
    if st["covariates_missing"]:
        check(code, "covariates_missing", st["covariates_missing"], "", False)
    A, L = r["A"], r["L"]
    trip = (A >= TRAVEL_ACTIVITY[0]) & (A < TRAVEL_ACTIVITY[1])
    mode_outside = np.isin(L, TRANSPORT_CODES) & ~trip
    if code == TARGET:
        check(code, "share_code_13", round(float((L == 13).mean()), 6), "about 0.95", 0.9 < (L == 13).mean() < 1)
        check(code, "place_codes_1_to_7", int(((L >= 1) & (L <= 7)).sum()), 0, not ((L >= 1) & (L <= 7)).any(),
              severity="fail")
        outside = int(mode_outside.sum())
        check(code, "mode_slots_outside_trips", outside, "<200", outside < 200,
              "German diaries record modes on trips only; slots with a mode count as transport in every country")
    else:
        check(code, "place_coding", scheme, "full or partial", scheme in ("full", "partial"),
              ", ".join(CLASS_LABELS[k] for k in range(N_CLASSES) if fold[k] != k))
        trip_at_place = float((trip & np.isin(L, PLACE_CODES)).sum() / max(trip.sum(), 1))
        check(code, "trip_slots_with_place_code_share", round(trip_at_place, 6), "<0.1", trip_at_place < 0.1,
              "trips recorded without a transport mode")
        share = float(mode_outside.sum() / max((~trip).sum(), 1))
        check(code, "mode_slots_outside_trips_share", round(share, 6), "for information", True,
              "share of the slots of other activities with a recorded mode, counted as transport (version 4)")


# --------------------------------------------------------------------------- episodes
def build_episodes(A, C, act_index_flat):
    """Episodes of one country: runs of the same activity code within a diary, split where
    transport starts or ends (version 4: slots with a recorded mode are transport, so an activity
    carried out partly at a place and partly on a train gives two episodes)."""
    n, T = A.shape
    a, c = A.ravel(), C.ravel()
    tr = c == TRAVEL_SLOT
    slot = np.tile(np.arange(T), n)
    new = np.ones(n * T, bool)
    new[1:] = (a[1:] != a[:-1]) | (tr[1:] != tr[:-1]) | (slot[1:] == 0)
    starts = np.flatnonzero(new)
    lengths = np.diff(np.append(starts, n * T))
    ep_of_slot = np.repeat(np.arange(len(starts)), lengths)
    diary, start = (starts // T).astype(np.int32), starts % T
    act_index = act_index_flat[starts].astype(np.int16)
    travel = tr[starts]                                    # transport episodes, never imputed

    # recorded class of each episode: the most frequent known class among its slots
    known = c >= 0
    keys = ep_of_slot[known].astype(np.int64) * N_CLASSES + c[known]
    uk, cnt = np.unique(keys, return_counts=True)
    ep, cl = uk // N_CLASSES, (uk % N_CLASSES).astype(np.int8)
    cls = np.full(len(starts), UNKNOWN, np.int8)
    if len(uk):  # Germany has no recorded places, so this can be empty
        order = np.lexsort((-cnt, ep))
        first = np.r_[True, ep[order][1:] != ep[order][:-1]]
        cls[ep[order][first]] = cl[order][first]
    cls[travel] = TRAVEL_SLOT

    # non-travel episodes: previous place, trip in between, rank within the diary
    nt = np.flatnonzero(~travel)
    d = diary[nt]
    first_nt = np.r_[True, d[1:] != d[:-1]]
    prev = np.full(len(nt), START, np.int8)
    prev_cls = cls[nt[:-1]]
    prev[1:] = np.where(first_nt[1:], START, np.where(prev_cls >= 0, prev_cls, START))
    first_ep_of_diary = np.searchsorted(diary, d)          # episodes are sorted by diary
    gap = np.r_[nt[0] - first_ep_of_diary[0], np.diff(nt)] if len(nt) else np.array([], int)
    trip = np.where(first_nt, nt > first_ep_of_diary, gap > 1).astype(np.int8)
    group_start = np.maximum.accumulate(np.where(first_nt, np.arange(len(nt)), 0))
    rank = (np.arange(len(nt)) - group_start).astype(np.int16)
    return dict(n=n, lengths=lengths, travel=travel, cls=cls, nt=nt, diary=d,
                a=act_index[nt], h=(start[nt] // 6).astype(np.int8), p=prev, r=trip, k=rank)


def slot_units(C, act_index_flat):
    """Non-travel slots as units for the slot-level variants."""
    T = C.shape[1]
    flat = np.flatnonzero(C.ravel() != TRAVEL_SLOT).astype(np.int32)
    return dict(diary=(flat // T).astype(np.int32), t=(flat % T).astype(np.int16),
                a=act_index_flat[flat].astype(np.int16), cls=C.ravel()[flat], flat=flat)


# --------------------------------------------------------------------------- model
RADIX = dict(a=None, t=SLOTS, h=24, g=AGE_TOP + 1, w=3, p=N_CLASSES + 1, r=2)


def keys_for(feats, level):
    key = np.zeros(len(feats["a"]), np.int64)
    for f in level:
        key = key * RADIX[f] + feats[f].astype(np.int64)
    return key


def count_table(keys, cls):
    m = cls >= 0
    kc = keys[m] * N_CLASSES + cls[m]
    u, n = np.unique(kc, return_counts=True)
    uk, inv = np.unique(u // N_CLASSES, return_inverse=True)
    M = np.bincount(inv * N_CLASSES + (u % N_CLASSES), weights=n,
                    minlength=len(uk) * N_CLASSES).reshape(len(uk), N_CLASSES)
    return uk, M


def merge_tables(tables):
    keys = np.concatenate([t[0] for t in tables])
    mats = np.concatenate([t[1] for t in tables])
    order = np.argsort(keys, kind="stable")
    keys, mats = keys[order], mats[order]
    bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[bounds], np.add.reduceat(mats, bounds, axis=0)


class Model:
    """Hierarchical Dirichlet smoothing: each level shrinks toward the level above.
    Classes listed in `banned` get probability zero (not used by the analyses of version 4)."""

    def __init__(self, levels, tables, kappa, banned=()):
        self.levels, self.tables, self.kappa, self.banned = levels, tables, kappa, list(banned)
        total = tables[0][1].sum(0)
        self.root = (total + 1.0) / (total.sum() + N_CLASSES)

    def probs(self, feats):
        n = len(feats["a"])
        p = np.tile(self.root, (n, 1))
        for level, (uk, M) in zip(self.levels, self.tables):
            q = keys_for(feats, level)
            idx = np.minimum(np.searchsorted(uk, q), len(uk) - 1)
            found = uk[idx] == q
            counts = np.zeros((n, N_CLASSES))
            counts[found] = M[idx[found]]
            p = (counts + self.kappa * p) / (counts.sum(1, keepdims=True) + self.kappa)
        if self.banned:
            p[:, self.banned] = 0.0
            p /= p.sum(1, keepdims=True)
        return p


class Pool:
    """Donor count tables of one specification, stacked once so that a model with any
    weights on the donor countries is a single weighted sum. With weight 1 for every donor in
    a list and 0 for the rest, it reproduces Tables.model for that list (used by part A of the
    global sensitivity analysis to leave out one donor at a time)."""

    def __init__(self, tables, spec, donors):
        kind = "slot" if spec["kind"] == "slot" else "episode"
        self.levels, self.donors = spec["levels"], list(donors)
        self.parts = []
        for level in self.levels:
            per = [tables.get(c, kind, level) for c in self.donors]
            keys = np.concatenate([t[0] for t in per])
            mats = np.concatenate([t[1] for t in per])
            owner = np.concatenate([np.full(len(t[0]), i, np.int16) for i, t in enumerate(per)])
            order = np.argsort(keys, kind="stable")
            keys, mats, owner = keys[order], mats[order], owner[order]
            bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
            self.parts.append((keys[bounds], mats, owner, bounds))

    def model(self, weights, kappa, banned=()):
        weights = np.asarray(weights, float)
        tables = [(uk, np.add.reduceat(mats * weights[owner][:, None], bounds, axis=0))
                  for uk, mats, owner, bounds in self.parts]
        return Model(self.levels, tables, kappa, banned)


def draw(p, rng, modal):
    if modal:
        return p.argmax(1).astype(np.int8)
    cum = np.cumsum(p, axis=1)
    r = rng.random(len(p)) * cum[:, -1]
    return np.minimum((cum < r[:, None]).sum(1), N_CLASSES - 1).astype(np.int8)


def draw_batched(model, feats, rng, modal, batch=400_000):
    n = len(feats["a"])
    out = np.empty(n, np.int8)
    for s in range(0, n, batch):
        part = {k: v[s:s + batch] for k, v in feats.items()}
        out[s:s + batch] = draw(model.probs(part), rng, modal)
    return out


# --------------------------------------------------------------------------- imputation
def impute(spec, model, data, rng, keep_known=False):
    """Imputed class array (n x 144) for one country. Trips keep TRAVEL_SLOT."""
    C, E, S = data["C"], data["E"], data["S"]
    g, w = data["age"], data["weekend"]
    if spec["kind"] == "home":
        return np.where(C == TRAVEL_SLOT, TRAVEL_SLOT, HOME).astype(np.int8)
    if spec["kind"] == "slot":
        feats = dict(a=S["a"], t=S["t"], g=g[S["diary"]], w=w[S["diary"]])
        drawn = draw_batched(model, feats, rng, spec.get("modal", False))
        if keep_known:
            drawn = np.where(S["cls"] >= 0, S["cls"], drawn)
        out = C.ravel().copy()
        out[S["flat"]] = drawn
        return out.reshape(C.shape)
    known = E["cls"][E["nt"]]
    if spec["kind"] == "episode":
        feats = dict(a=E["a"], h=E["h"], g=g[E["diary"]], w=w[E["diary"]])
        drawn = draw_batched(model, feats, rng, spec.get("modal", False))
        if keep_known:
            drawn = np.where(known >= 0, known, drawn)
    else:  # markov: episodes drawn in order, each conditioned on the place drawn before it
        drawn = np.empty(len(E["nt"]), np.int8)
        last = np.full(E["n"], START, np.int8)
        order = np.argsort(E["k"], kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(E["k"][order])) + 1, len(order)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            sel = order[s:e]
            d = E["diary"][sel]
            feats = dict(a=E["a"][sel], h=E["h"][sel], r=E["r"][sel], p=last[d], g=g[d], w=w[d])
            c = draw(model.probs(feats), rng, spec.get("modal", False))
            if keep_known:
                c = np.where(known[sel] >= 0, known[sel], c)
            drawn[sel] = c
            last[d] = c
    ep_class = np.full(len(E["lengths"]), TRAVEL_SLOT, np.int8)
    ep_class[E["nt"]] = drawn
    return np.repeat(ep_class, E["lengths"]).reshape(C.shape)


# --------------------------------------------------------------------------- scoring
def weighted_slot_shares(C, weight, fold):
    """Share of the weighted population in each class (and in travel) at each slot."""
    w = weight / weight.sum()
    shares = np.zeros((SLOTS, N_CLASSES + 1))
    folded = np.where(C >= 0, fold[np.maximum(C, 0)], C)
    for k in range(N_CLASSES):
        shares[:, k] = w @ (folded == k)
    shares[:, N_CLASSES] = w @ (C == TRAVEL_SLOT)
    return shares


def changes_per_diary(C, fold):
    n, T = C.shape
    flat = C.ravel()
    idx = np.flatnonzero(flat >= 0)
    folded = fold[flat[idx]]
    same_diary = (idx[1:] // T) == (idx[:-1] // T)
    change = same_diary & (folded[1:] != folded[:-1])
    return np.bincount(idx[1:][change] // T, minlength=n)


MAX_DIARIES_PER_RESPONDENT = 7   # HETUS 2010 designs have 1 to 7 diary days per respondent


def respondents_by_class(C, person, n_classes):
    """Number of distinct respondents with at least one slot in each class (C: diaries x slots,
    negative where a slot is not counted; person: respondent code of each diary, or None when the
    identifiers are missing, in which case the count is the smallest number of respondents that
    can have kept the diaries concerned, at MAX_DIARIES_PER_RESPONDENT diaries each)."""
    n, T = C.shape
    idx = np.flatnonzero(C.ravel() >= 0)
    has = np.zeros(n * n_classes, bool)
    has[(idx // T) * n_classes + C.ravel()[idx]] = True
    has = has.reshape(n, n_classes)
    if person is None:
        return -(-has.sum(0) // MAX_DIARIES_PER_RESPONDENT)
    return np.array([np.unique(person[has[:, k]]).size for k in range(n_classes)])


def person_codes(ids, n):
    """One integer code per respondent, from the household and person identifiers (HID, PID),
    and whether they were found. Without them respondents cannot be told apart, and the
    disclosure control counts them conservatively (respondents_by_class with person=None)."""
    if ids is not None and {"HID", "PID"} <= set(ids.columns) and ids[["HID", "PID"]].notna().all().all():
        key = ids["HID"].astype("int64").to_numpy() * 1000 + ids["PID"].astype("int64").to_numpy()
        return pd.factorize(key)[0], True
    return None, False


def score(obs, imp, weight, fold, person=None, count_respondents=False):
    valid = obs >= 0
    o, p = fold[obs[valid]], fold[imp[valid]]
    K = N_CLASSES
    conf = np.bincount(o * K + p, minlength=K * K).reshape(K, K).astype(float)
    total = conf.sum()
    acc = np.trace(conf) / total
    expected = (conf.sum(0) * conf.sum(1)).sum() / total ** 2
    rows, cols, diag = conf.sum(1), conf.sum(0), np.diag(conf)
    recall = np.divide(diag, rows, out=np.full(K, np.nan), where=rows > 0)
    precision = np.divide(diag, cols, out=np.full(K, np.nan), where=cols > 0)
    f1 = np.where(rows > 0, 2 * recall * np.nan_to_num(precision) /
                  np.maximum(recall + np.nan_to_num(precision), 1e-12), np.nan)
    so, si = weighted_slot_shares(obs, weight, fold), weighted_slot_shares(imp, weight, fold)
    hourly_o = so.reshape(24, 6, -1).mean(1)
    hourly_i = si.reshape(24, 6, -1).mean(1)
    co, ci = changes_per_diary(obs, fold), changes_per_diary(imp, fold)
    overall = dict(
        scored_slots=int(total), accuracy=acc, kappa=(acc - expected) / (1 - expected),
        macro_f1=np.nanmean(f1),
        home_share_mae_pp=100 * np.abs(so[:, HOME] - si[:, HOME]).mean(),
        home_share_max_pp=100 * np.abs(so[:, HOME] - si[:, HOME]).max(),
        hourly_tvd=0.5 * np.abs(hourly_o - hourly_i).sum(1).mean(),
        changes_observed=co.mean(), changes_imputed=ci.mean(),
        changes_mean_abs_diff=np.abs(co - ci).mean())
    by_class = pd.DataFrame(dict(
        place=CLASS_LABELS, observed_share=rows / total, imputed_share=cols / total,
        recall=recall, precision=precision, f1=f1, scored_slots=rows))
    if count_respondents:       # version 3.1: respondents behind recall (recorded) and precision (imputed)
        of = np.where(valid, fold[np.maximum(obs, 0)], -1)
        pf = np.where(valid, fold[np.maximum(imp, 0)], -1)
        by_class["respondents_recorded"] = respondents_by_class(of, person, K)
        by_class["respondents_imputed"] = respondents_by_class(pf, person, K)
    by_class = by_class[rows > 0].copy()
    small = by_class.scored_slots < MIN_CELL          # suppress rare places entirely
    by_class.loc[small, ["observed_share", "imputed_share", "recall", "precision", "f1", "scored_slots"]] = np.nan
    home = pd.DataFrame(dict(slot=np.arange(SLOTS), clock=CLOCK, observed=so[:, HOME], imputed=si[:, HOME]))
    # weighted hours per day by place (recorded and imputed), transport included
    hours = pd.DataFrame(dict(place=CLASS_LABELS + [TRANSPORT_LABEL],
                              recorded_hours=24 * so.mean(0), imputed_hours=24 * si.mean(0),
                              recorded_class=list(fold == np.arange(N_CLASSES)) + [True]))
    return overall, by_class, home, hours


def publish_by_class(bc, schemes):
    """Per-place scores that can be published, and their means over countries (version 3.1).

    The means over the countries of a coding group are taken from all scores. A recall (and the
    slot count behind it) is withheld for every country of a coding group, variant and place when
    fewer than MIN_RESPONDENTS respondents were recorded at the place in any of these countries,
    and a precision likewise when fewer respondents were imputed to it (a place never imputed has
    no precision to withhold); F1 goes with either. Withholding the whole group keeps the means
    from revealing a withheld score. Respondent counts below the threshold are blanked."""
    bc = bc.reset_index(drop=True)
    keys = [bc.country.map(schemes).rename("coding"), bc.variant, bc.place]
    scores = ["observed_share", "imputed_share", "recall", "precision", "f1"]
    means = bc.groupby(keys)[scores].mean()
    means["countries"] = bc.groupby(keys).recall.count()
    means = means.reset_index()
    if "respondents_recorded" not in bc:
        return bc, means
    few_rec = bc.respondents_recorded < MIN_RESPONDENTS
    few_imp = (bc.respondents_imputed < MIN_RESPONDENTS) & (bc.respondents_imputed > 0)
    hide_rec = few_rec.groupby(keys).transform("any")
    hide_imp = few_imp.groupby(keys).transform("any")
    bc.loc[hide_rec, ["recall", "scored_slots"]] = np.nan
    bc.loc[hide_imp, "precision"] = np.nan
    bc.loc[hide_rec | hide_imp, "f1"] = np.nan
    bc["respondents_recorded"] = bc.respondents_recorded.where(~few_rec)
    bc["respondents_imputed"] = bc.respondents_imputed.where(bc.respondents_imputed >= MIN_RESPONDENTS)
    return bc, means


def disclosure_checks(bc, schemes=None):
    """No published per-place score rests on fewer than MIN_RESPONDENTS respondents, and scores are
    published for all countries of a coding group, variant and place or for none."""
    if "respondents_recorded" not in bc:
        return
    if schemes is not None:
        keys = [bc.country.map(schemes).rename("coding"), bc.variant, bc.place]
        shown_r, shown_p = bc.recall.notna(), bc.precision.notna()
        withheld_p = ~shown_p & (bc.imputed_share.fillna(1) > 0)      # imputed somewhere, precision blank
        mixed = int((shown_r.groupby(keys).transform("any") & ~shown_r).sum()
                    + (shown_p.groupby(keys).transform("any") & withheld_p).sum())
        check("disclosure", "groups_partly_withheld", mixed, 0, mixed == 0, severity="fail",
              note="a score withheld in one country is withheld in all countries of its group")
    counts = pd.concat([bc.respondents_recorded, bc.respondents_imputed]).dropna()
    smallest = int(counts.min()) if len(counts) else 0
    check("disclosure", "smallest_published_respondent_count", smallest, f">={MIN_RESPONDENTS}",
          smallest >= MIN_RESPONDENTS, severity="fail")
    bad = int(((bc.recall.notna() & bc.respondents_recorded.isna())
               | (bc.precision.notna() & bc.respondents_imputed.isna())
               | (bc.f1.notna() & bc.recall.isna())).sum())
    check("disclosure", "scores_without_enough_respondents", bad, 0, bad == 0, severity="fail")
    withheld = bc[bc.recall.isna() & bc.observed_share.notna()].groupby(["variant", "place"]).size()
    check("disclosure", "places_with_recall_withheld", len(withheld), "for information", True,
          "variant and place pairs: " + "; ".join(f"{v} {p}" for v, p in withheld.index[:12])
          + (" ..." if len(withheld) > 12 else ""))


CLOCK = [f"{(4 + (10 * s) // 60) % 24:02d}:{(10 * s) % 60:02d}" for s in range(SLOTS)]


# --------------------------------------------------------------------------- intervals
def t975(df):
    """97.5% quantile of Student's t with df degrees of freedom (Cornish-Fisher expansion,
    within 0.001 of the exact value for df >= 4)."""
    z = 1.959963984540054
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def predictive_interval(draws):
    """Mean and 95% interval for one more draw from the distribution the imputations come
    from: mean +- t(0.975, M-1) * sd * sqrt(1 + 1/M), along axis 0. Shares are kept in [0, 1]
    by the caller where needed."""
    draws = np.asarray(draws, float)
    m = draws.shape[0]
    mean = draws.mean(0)
    if m < 2:
        return mean, mean.copy(), mean.copy()
    half = t975(m - 1) * draws.std(0, ddof=1) * math.sqrt(1 + 1 / m)
    return mean, mean - half, mean + half


def plain_scale(m):
    """Multiple of the standard deviation that gives the 95% predictive interval for m draws."""
    return t975(m - 1) * math.sqrt(1 + 1 / m) if m > 1 else 0.0


def mean_sd(draws):
    """Mean and standard deviation across imputations (axis 0)."""
    draws = np.asarray(draws, float)
    sd = draws.std(0, ddof=1) if draws.shape[0] > 1 else np.zeros(draws.shape[1:])
    return draws.mean(0), sd


def ratios(recorded, mean, sd, tiny=1e-12):
    """Distance of each recorded value from the mean of the imputations, in standard deviations.
    A spread below `tiny` (rounding noise) counts as none: the ratio is then 0 if the recorded
    value equals the mean and infinite otherwise."""
    dev = np.abs(np.asarray(recorded, float) - mean)
    spread = sd > tiny
    safe = np.where(spread, sd, 1.0)
    return np.where(spread, dev / safe, np.where(dev > tiny, np.inf, 0.0))


def coverage(R, scale):
    """Mean over countries of the share of units within `scale` standard deviations."""
    return float(np.mean([np.mean(r <= scale) for r in R if len(r)]))


# --------------------------------------------------------------------------- tables per country
class Tables:
    """Count tables per country and level, built once and merged for each fold.
    With `home_acts` (activity indices), code 13 is treated as missing for these activities: their
    slots and episodes recorded at other or unspecified places are left out of the counts, for
    slot-level and episode-level models alike (part B of the GSA)."""

    def __init__(self, data, home_acts=None):
        self.data, self.cache = data, {}
        self.home_acts = None if home_acts is None else np.asarray(home_acts)

    def get(self, country, kind, level):
        key = (country, kind, level)
        if key not in self.cache:
            d = self.data[country]
            g, w = d["age"], d["weekend"]
            if kind == "slot":
                S = d["S"]
                feats = dict(a=S["a"], t=S["t"], g=g[S["diary"]], w=w[S["diary"]])
                cls = S["cls"]
                if self.home_acts is not None:
                    cls = np.where(np.isin(S["a"], self.home_acts) & (cls == OTHER), UNKNOWN, cls).astype(np.int8)
                self.cache[key] = count_table(keys_for(feats, level), cls)
            else:
                E = d["E"]
                feats = dict(a=E["a"], h=E["h"], p=E["p"], r=E["r"], g=g[E["diary"]], w=w[E["diary"]])
                cls = E["cls"][E["nt"]]
                if self.home_acts is not None:
                    cls = np.where(np.isin(E["a"], self.home_acts) & (cls == OTHER), UNKNOWN, cls).astype(np.int8)
                self.cache[key] = count_table(keys_for(feats, level), cls)
        return self.cache[key]

    def model(self, spec, countries, kappa):
        kind = "slot" if spec["kind"] == "slot" else "episode"
        tables = [merge_tables([self.get(c, kind, lv) for c in countries]) for lv in spec["levels"]]
        return Model(spec["levels"], tables, kappa)


# --------------------------------------------------------------------------- checks and summaries
def imputation_checks(scope, d, imp):
    """Invariants every imputation must satisfy; a failure points to a bug."""
    C = d["C"]
    travel_same = bool(((imp == TRAVEL_SLOT) == (C == TRAVEL_SLOT)).all())
    check(scope, "transport_slots_unchanged", travel_same, True, travel_same, severity="fail")
    valid = bool(((imp == TRAVEL_SLOT) | ((imp >= 0) & (imp < N_CLASSES))).all())
    check(scope, "all_other_slots_imputed", valid, True, valid, severity="fail")
    shares = weighted_slot_shares(imp, d["weight"], np.arange(N_CLASSES))
    dev = float(np.abs(shares.sum(1) - 1).max())
    check(scope, "slot_shares_sum_to_one_maxdev", f"{dev:.2e}", "<1e-9", dev < 1e-9, severity="fail")


def model_checks(scope, model, d, sample=20000):
    """Probabilities of a fitted model on a sample of episodes: finite and summing to one."""
    E = d["E"]
    n = min(sample, len(E["nt"]))
    if n == 0:
        return
    sel = np.linspace(0, len(E["nt"]) - 1, n).astype(int)
    feats = dict(a=E["a"][sel], h=E["h"][sel], r=E["r"][sel], p=E["p"][sel],
                 g=d["age"][E["diary"][sel]], w=d["weekend"][E["diary"][sel]])
    p = model.probs(feats)
    ok = bool(np.isfinite(p).all() and np.abs(p.sum(1) - 1).max() < 1e-9 and (p >= 0).all())
    check(scope, "model_probabilities_valid", ok, True, ok, severity="fail")
    return feats


def group_masks(d):
    return {"all": np.ones(d["n"], bool),
            "age 10-17": d["age"] == 1, "age 18-64": (d["age"] >= 2) & (d["age"] <= 11),
            "age 65+": d["age"] >= 12, "weekday": d["weekend"] == 0, "weekend": d["weekend"] == 1}


def group_summaries(imp, d, groups):
    """Per group: weighted shares by slot (144 x 9, transport last), hours per day by place, and the
    unweighted and weighted mean number of location changes per diary."""
    changes = changes_per_diary(imp, np.arange(N_CLASSES))
    out = {}
    for gname, mask in groups.items():
        if mask.sum() < MIN_GROUP_DIARIES:
            continue
        s = weighted_slot_shares(imp[mask], d["weight"][mask], np.arange(N_CLASSES))
        w = d["weight"][mask]
        out[gname] = dict(shares=s, hours=24 * s.mean(0), changes=changes[mask].mean(),
                          changes_weighted=float(np.average(changes[mask], weights=w)) if w.sum() > 0 else np.nan)
    return out


def home_activities(data, donors, codes):
    """Activities that donors do almost only at home, among slots with a specified place
    (codes 1 to 7). For these, code 13 more likely means 'not specified' than 'elsewhere'."""
    n_act = len(codes)
    counts = np.zeros((n_act, N_CLASSES))
    respondents = np.zeros(n_act, np.int64)          # version 3.1: respondents behind each activity's shares
    for c in donors:
        S = data[c]["S"]
        m = S["cls"] >= 0
        idx = S["a"][m].astype(np.int64) * N_CLASSES + S["cls"][m]
        counts += np.bincount(idx, minlength=n_act * N_CLASSES).reshape(n_act, N_CLASSES)
        person = data[c].get("person")
        unit = (person[S["diary"][m]] if person is not None else S["diary"][m]).astype(np.int64)
        base = int(unit.max()) + 1 if len(unit) else 1
        pairs = np.unique(S["a"][m].astype(np.int64) * base + unit)
        per_act = np.bincount(pairs // base, minlength=n_act)
        respondents += per_act if person is not None else -(-per_act // MAX_DIARIES_PER_RESPONDENT)
    specified = counts[:, :OTHER].sum(1)              # places 1 to 7 (classes 0 to 6)
    home_share = np.divide(counts[:, HOME], specified, out=np.zeros(n_act), where=specified > 0)
    known = counts.sum(1)
    share13 = np.divide(counts[:, OTHER], known, out=np.zeros(n_act), where=known > 0)
    flagged = ((home_share >= HOME_ACTIVITY_SHARE) & (specified >= HOME_ACTIVITY_MIN_SLOTS)
               & (share13 <= HOME_ACTIVITY_MAX_CODE13))
    few = respondents < MIN_RESPONDENTS              # shares on fewer respondents are not published
    table = pd.DataFrame(dict(activity_code=codes,
                              donor_slots=pd.Series(known.astype(np.int64)).mask(few).astype("Int64"),
                              home_share_of_specified=np.where(few, np.nan, home_share),
                              share_code_13=np.where(few, np.nan, share13), home_activity=flagged,
                              respondents=pd.Series(respondents).mask(few).astype("Int64")))
    table = table[(known > 0) & ~((codes >= TRAVEL_ACTIVITY[0]) & (codes < TRAVEL_ACTIVITY[1]))]
    small_flagged = int((flagged & few).sum())
    check("disclosure", "home_activities_with_few_respondents", small_flagged, 0, small_flagged == 0,
          "home activities resting on fewer than MIN_RESPONDENTS respondents (their shares are blanked)")
    return np.flatnonzero(flagged), table


# --------------------------------------------------------------------------- version 3: global sensitivity analysis
def gsa_levels(unit, time_, age, day):
    """Smoothing hierarchy of one combination of choices: the activity alone, then the time
    (slot or starting hour), then for the Markov chain the previous place and the trip flag,
    then the age band and the day type, each level adding one block to the level above."""
    levels, current = [("a",)], ("a",)
    blocks = []
    if time_:
        blocks.append(("t",) if unit == "slot" else ("h",))
    if unit == "markov":
        blocks.append(("p", "r"))
    if age:
        blocks.append(("g",))
    if day:
        blocks.append(("w",))
    for block in blocks:
        current = current + block
        levels.append(current)
    return levels


def gsa_spec(unit, time_, age, day, modal=False):
    return dict(kind=unit, levels=gsa_levels(unit, time_, age, day), modal=modal)


def factorial_indices(factors, Y, tiny=1e-15):
    """Exact first-order and total Sobol' indices from a complete, balanced factorial design in
    which every factor is uniform over its levels (functional ANOVA decomposition).

    factors: DataFrame, one row per run and one column per factor (its level in that run).
    Y: array (runs x outputs). Returns (S, ST), arrays (factors x outputs), with
        S_i  = Var(E[Y | X_i]) / Var(Y)
        ST_i = E[Var(Y | X_~i)] / Var(Y),
    NaN for outputs that do not vary. Raises ValueError if the design is not a complete factorial."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    names = list(factors.columns)
    n_levels = [factors[f].nunique() for f in names]
    if len(factors) != int(np.prod(n_levels)) or factors.drop_duplicates().shape[0] != len(factors):
        raise ValueError("factorial_indices needs every combination of the factor levels exactly once")
    V = Y.var(axis=0)
    varies = V > tiny
    frame = pd.DataFrame(Y)
    S = np.full((len(names), Y.shape[1]), np.nan)
    ST = np.full((len(names), Y.shape[1]), np.nan)
    for i, f in enumerate(names):
        cond = frame.groupby(factors[f].to_numpy()).mean()          # E[Y | X_i], one row per level
        Vi = cond.var(axis=0, ddof=0).to_numpy()
        others = [factors[g].to_numpy() for g in names if g != f]
        if others:
            Vti = frame.groupby(others).var(ddof=0).mean(axis=0).to_numpy()
        else:
            Vti = V.copy()
        S[i, varies] = Vi[varies] / V[varies]
        ST[i, varies] = Vti[varies] / V[varies]
    return S, ST


def indices_frame(factors, Y, outputs):
    """Long table of the indices: one row per output and factor, plus the share of variance
    left to interactions (one minus the sum of the first-order indices)."""
    S, ST = factorial_indices(factors, Y)
    rows = []
    for j, out in enumerate(outputs):
        for i, f in enumerate(factors.columns):
            rows.append(dict(output=out, factor=f, first_order=S[i, j], total=ST[i, j]))
        rows.append(dict(output=out, factor="interactions", first_order=1 - np.nansum(S[:, j])
                         if np.isfinite(S[:, j]).any() else np.nan, total=np.nan))
    return pd.DataFrame(rows), S, ST


def indices_checks(scope, S, ST, tol=1e-9):
    """Sobol' indices of a complete factorial must satisfy 0 <= S_i <= ST_i <= 1 and sum S_i <= 1."""
    ok = np.isfinite(S)
    bounds = bool(((S[ok] >= -tol) & (S[ok] <= ST[ok] + tol) & (ST[ok] <= 1 + tol)).all())
    check(scope, "indices_within_bounds", bounds, True, bounds, "0 <= first-order <= total <= 1", severity="fail")
    sums = np.nansum(S, axis=0)
    sum_ok = bool((sums <= 1 + 1e-6).all())
    check(scope, "first_order_sum_at_most_one", f"{sums.max():.6f}" if len(sums) else "nan", "<=1", sum_ok,
          severity="fail")


def pareto_optimal(means):
    """Combinations of part A that no other combination matches or beats on every metric of
    GSA_DIRECTIONS while beating it on at least one (Pareto-optimal, non-dominated)."""
    cols = []
    for m, sense in GSA_DIRECTIONS.items():
        v = means[m].to_numpy(float)
        cols.append(v if sense > 0 else (-v if sense < 0 else -np.abs(v)))
    X = np.column_stack(cols)                       # larger is better in every column
    geq = (X[:, None, :] >= X[None, :, :]).all(2)  # geq[i, j]: i at least as good as j on every metric
    gt = (X[:, None, :] > X[None, :, :]).any(2)
    dominated = (geq & gt).any(0)
    return ~dominated


def quantile_rows(values, main):
    """Summary of the distribution of one output over the combinations."""
    v = np.asarray(values, float)
    q = np.quantile(v, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    return dict(main_imputed=main, mean=v.mean(), sd=v.std(), q025=q[0], q05=q[1], q25=q[2], q50=q[3],
                q75=q[4], q95=q[5], q975=q[6], min=v.min(), max=v.max())


def alpha_text(frame):
    """alpha written as in the other outputs (1.0, 5.0, 20.0) where a compact float format is used."""
    return frame.assign(alpha=frame["alpha"].map(lambda a: f"{a:.1f}"))


def yes_no(flag):
    return "yes" if flag else "no"


def day_groups(d):
    """All diaries, weekdays and weekend days (diaries without a day of the week only in 'all')."""
    return {"all": np.ones(d["n"], bool), "weekday": d["weekend"] == 0, "weekend": d["weekend"] == 1}


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=None, help="path to TUS_Eurostat.nml (optional)")
    parser.add_argument("--out", default=None, help="output folder (default: imputation_out next to the script)")
    parser.add_argument("--quick", action="store_true",
                        help="fewer variants and imputations, as a first check")
    parser.add_argument("--skip-gsa", action="store_true",
                        help="skip the global sensitivity analysis (about half the run time)")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    out = Path(args.out).expanduser() if args.out else script_dir / "imputation_out"
    out.mkdir(parents=True, exist_ok=True)
    run_start = time.time()
    variants = {k: VARIANTS[k] for k in (QUICK_VARIANTS if args.quick else VARIANTS)}
    m_imputations = 3 if args.quick else N_IMPUTATIONS
    m_rep = 4 if args.quick else REPEAT_IMPUTATIONS
    n_draws = 2 if args.quick else GSA_DRAWS
    gsa_alphas = (KAPPA,) if args.quick else GSA_ALPHAS
    timing = {}
    tables13 = None
    fixed, main_hours = [], {}
    t_stage = time.time()

    nml, files = country_files(args.source, script_dir)
    log(f"location_imputation.py version {VERSION}; settings file {nml}")
    raw = {}
    for code in sorted(files):
        log(f"reading {code} ({files[code].name})")
        raw[code] = read_country(files[code])
    codes = np.unique(np.concatenate([r["A"].ravel() for r in raw.values()]))
    RADIX["a"] = len(codes)

    data, schemes = {}, {}
    for code in sorted(raw):
        r = raw.pop(code)
        C = slot_classes(r["A"], r["L"])
        scheme, fold, present = coding_scheme(C)
        data_checks(code, r, C, "target" if code == TARGET else scheme, fold)
        if code == TARGET:      # in Germany code 13 marks a place that was not recorded
            C[C == OTHER] = UNKNOWN
            scheme = "target"
        act_index_flat = np.searchsorted(codes, r["A"].ravel()).astype(np.int16)
        E = build_episodes(r["A"], C, act_index_flat)
        S = slot_units(C, act_index_flat)
        person, identified = person_codes(r["ids"], len(C))
        check(code, "respondents_identified", identified, True, identified,
              "HID and PID found: respondents are counted for the disclosure control of per-place scores"
              if identified else f"HID or PID missing: respondents counted as diaries / {MAX_DIARIES_PER_RESPONDENT}, "
              "rounded up (fewest respondents that can have kept the diaries)")
        data[code] = dict(C=C, E=E, S=S, age=r["age"], weekend=r["weekend"], weight=r["weight"],
                          fold=fold, ids=r["ids"], person=person, n=len(C),
                          L=r["L"] if code == TARGET else None)
        schemes[code] = scheme
        log(f"{code}: {len(C)} diaries, {len(E['nt'])} non-travel episodes, place coding {scheme}"
            + ("" if scheme in ("full", "target") else
               f" (folded into code 13: {', '.join(CLASS_LABELS[k] for k in range(N_CLASSES) if fold[k] != k)})"))
    full = [c for c, s in schemes.items() if s == "full"]
    partial = [c for c, s in schemes.items() if s == "partial"]
    log(f"donors with full place coding: {' '.join(full)}; partial (scored only): {' '.join(partial)}")
    check("run", "donors_with_full_coding", len(full), 13, len(full) == 13, " ".join(full))
    timing["read_and_prepare"] = round(time.time() - t_stage, 1)

    # ----------------------------------------------------------------------- leave-one-country-out (step 4)
    t_stage = time.time()
    tables = Tables(data)
    rng_root = np.random.SeedSequence(SEED)
    overall_rows, class_rows, home_rows, hours_rows = [], [], [], []
    for held in full + partial:
        train = [c for c in full if c != held]          # trained on the other donors, validated on `held`
        d = data[held]
        for vname in variants:
            spec = VARIANTS[vname]
            model = None if spec["kind"] == "home" else tables.model(spec, train, KAPPA)
            rng = np.random.default_rng(rng_root.spawn(1)[0])
            imp = impute(spec, model, d, rng)
            o, bc, hm, hr = score(d["C"], imp, d["weight"], d["fold"], person=d["person"], count_respondents=True)
            overall_rows.append(dict(country=held, coding=schemes[held], variant=vname, diaries=d["n"], **o))
            class_rows.append(bc.assign(country=held, variant=vname))
            home_rows.append(hm.assign(country=held, variant=vname))
            hours_rows.append(hr.assign(country=held, coding=schemes[held], variant=vname))
            if vname == MAIN_VARIANT:
                imputation_checks(f"{held} loco", d, imp)
        best = max((r for r in overall_rows if r["country"] == held), key=lambda r: r["accuracy"])
        log(f"held out {held}: best accuracy {best['accuracy']:.3f} ({best['variant']})")

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out / "loco_overall.csv", index=False)
    by_class_pub, by_class_means = publish_by_class(pd.concat(class_rows), schemes)
    by_class_pub.to_csv(out / "loco_by_class.csv", index=False)
    by_class_means.to_csv(out / "loco_by_class_means.csv", index=False)
    disclosure_checks(by_class_pub, schemes)
    pd.concat(home_rows).to_csv(out / "loco_home_by_slot.csv", index=False)
    hours_cols = ["country", "coding", "variant", "place", "recorded_class", "recorded_hours", "imputed_hours"]
    pd.concat(hours_rows)[hours_cols].to_csv(out / "loco_hours_per_day.csv", index=False)
    metrics = ["accuracy", "kappa", "macro_f1", "home_share_mae_pp", "hourly_tvd",
               "changes_observed", "changes_imputed", "changes_mean_abs_diff"]
    summary = overall.groupby(["coding", "variant"])[metrics].mean().reset_index()
    summary.to_csv(out / "loco_summary.csv", index=False)
    for coding, title in (("full", "held-out countries with full place coding"),
                          ("partial", "AT, FR, LU, scored on the places they record")):
        part = summary[summary.coding == coding].drop(columns="coding")
        if len(part):
            log(f"leave-one-country-out means, {title}:")
            print(part.round(3).to_string(index=False))
    timing["loco_variants"] = round(time.time() - t_stage, 1)

    # ----------------------------------------------------------------------- repeated imputations (step 5)
    spec = VARIANTS[MAIN_VARIANT]
    if full:   # the stacked pool of part A of the GSA must reproduce the fixed model exactly
        pool = Pool(tables, spec, full)
        held = full[0]
        mask = np.array([c != held for c in full])
        feats = model_checks(f"{held} pool", pool.model(mask.astype(float), KAPPA), data[held])
        if feats is not None:
            diff = float(np.abs(pool.model(mask.astype(float), KAPPA).probs(feats)
                                - tables.model(spec, [c for c in full if c != held], KAPPA).probs(feats)).max())
            check(f"{held} pool", "pool_equals_fixed_model_maxdiff", f"{diff:.1e}", "0", diff == 0, severity="fail")
        del pool

    t_stage = time.time()
    rng_rep = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    lam0 = plain_scale(m_rep)                  # plain 95% predictive interval for one more draw
    countries = full + partial
    log(f"repeated imputations: {len(countries)} held-out countries x {m_rep} imputations of the main variant")
    rep_rows, int_rows, R = [], [], {"home": [], "hours": [], "changes": []}
    for held in countries:
        d = data[held]
        model = tables.model(spec, [c for c in full if c != held], KAPPA)
        so = weighted_slot_shares(d["C"], d["weight"], d["fold"])
        rec_hours = 24 * so.mean(0)
        scored = np.array([k for k in range(N_CLASSES) if d["fold"][k] == k and rec_hours[k] > 0])
        rec_changes = np.array([changes_per_diary(d["C"], d["fold"]).mean()])
        homes, hours, changes = [], [], []
        for _ in range(m_rep):
            imp = impute(spec, model, d, rng_rep)
            si = weighted_slot_shares(imp, d["weight"], d["fold"])
            homes.append(si[:, HOME])
            hours.append(24 * si.mean(0)[scored])
            changes.append([changes_per_diary(imp, d["fold"]).mean()])
        mh, sh = mean_sd(homes)
        r = dict(home=ratios(so[:, HOME], mh, sh), hours=ratios(rec_hours[scored], *mean_sd(hours)),
                 changes=ratios(rec_changes, *mean_sd(changes)))
        for q in R:
            R[q].append(r[q])
        rep_rows.append(dict(
            country=held, coding=schemes[held], imputations=m_rep, bias_home_pp=100 * float((mh - so[:, HOME]).mean()),
            coverage_home=float(np.mean(r["home"] <= lam0)), width_home_pp=100 * 2 * lam0 * float(sh.mean()),
            max_width_home_pp=100 * 2 * lam0 * float(sh.max()),
            coverage_hours=float(np.mean(r["hours"] <= lam0)), changes_recorded=float(rec_changes[0]),
            changes_imputed=float(np.mean(changes)), changes_covered=bool(r["changes"][0] <= lam0)))
        int_rows.append(pd.DataFrame(dict(country=held, slot=np.arange(SLOTS), clock=CLOCK, recorded=so[:, HOME],
                                          mean=mh, sd=sh, lo=np.clip(mh - lam0 * sh, 0, 1),
                                          hi=np.clip(mh + lam0 * sh, 0, 1))))
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(out / "loco_repeated_imputations.csv", index=False)
    pd.concat(int_rows).to_csv(out / "loco_intervals_home_by_slot.csv", index=False)
    rep_summary = []
    for name, sel in (("all held-out countries", rep.country.isin(countries)), ("donors", rep.country.isin(full))):
        idx = [j for j, c in enumerate(countries) if sel.iloc[j]]
        rep_summary.append(dict(
            countries=name, n_countries=len(idx), imputations=m_rep, scale_plain=lam0,
            coverage_home=coverage([R["home"][j] for j in idx], lam0),
            coverage_hours=coverage([R["hours"][j] for j in idx], lam0),
            coverage_changes=coverage([R["changes"][j] for j in idx], lam0),
            width_home_pp=float(rep[sel].width_home_pp.mean()),
            home_units_without_spread=float(np.mean(np.isinf(np.concatenate([R["home"][j] for j in idx]))))))
    rep_summary = pd.DataFrame(rep_summary)
    rep_summary.to_csv(out / "loco_repeated_imputations_summary.csv", index=False)
    log("coverage of plain 95% intervals from repeated imputations (recorded values inside):")
    print(rep_summary.round(3).to_string(index=False))
    check("run", "coverage_home_repeated_imputations", round(float(rep_summary.coverage_home.iloc[0]), 3),
          "for information", True, f"plain 95% intervals from {m_rep} imputations per held-out country")
    timing["repeated_imputations"] = round(time.time() - t_stage, 1)

    # ----------------------------------------------------------------------- Germany (step 6)
    if TARGET in data:
        t_stage = time.time()
        d = data[TARGET]
        groups = group_masks(d)
        L = d["L"]
        transport_codes = np.where(np.isin(L, TRANSPORT_CODES), L, OUT_TRAVEL_NO_MODE).astype(np.int8)
        class_codes = np.array(PLACE_CODES, np.int8)

        def codes_of(imp):
            return np.where(imp == TRAVEL_SLOT, transport_codes, class_codes[np.maximum(imp, 0)])

        model = tables.model(spec, full, KAPPA)
        model_checks(f"{TARGET} fixed", model, d)
        stack = np.empty((m_imputations, d["n"], SLOTS), np.int8)
        for m in range(m_imputations):
            rng = np.random.default_rng(rng_root.spawn(1)[0])
            imp = impute(spec, model, d, rng, keep_known=True)
            if m == 0:
                imputation_checks(f"{TARGET} fixed", d, imp)
            stack[m] = codes_of(imp)
            fixed.append(group_summaries(imp, d, groups))
            if (m + 1) % 5 == 0 or m + 1 == m_imputations:
                log(f"Germany imputation {m + 1}/{m_imputations}")
        np.savez_compressed(out / "PRIVATE_germany_imputed_codes.npz", codes=stack)
        del stack
        d["ids"].to_csv(out / "PRIVATE_germany_ids.csv", index=False)

        labels = CLASS_LABELS + [TRANSPORT_LABEL]
        share_rows, hour_rows, change_rows, per_imp_rows = [], [], [], []
        for gname in fixed[0]:
            arr = np.stack([f[gname]["shares"] for f in fixed])           # m x slot x category
            _, lo_mc, hi_mc = predictive_interval(arr)                    # plain, Monte Carlo variation only
            for k, label in enumerate(labels):
                for slot in range(SLOTS):
                    v = arr[:, slot, k]
                    share_rows.append(dict(group=gname, diaries=int(groups[gname].sum()), slot=slot,
                                           clock=CLOCK[slot], place=label, mean=v.mean(),
                                           p025=np.quantile(v, 0.025), p975=np.quantile(v, 0.975),
                                           lo_mc=max(lo_mc[slot, k], 0), hi_mc=min(hi_mc[slot, k], 1)))
            hrs = np.stack([f[gname]["hours"] for f in fixed])
            _, hlo, hhi = predictive_interval(hrs)
            for k, label in enumerate(labels):
                hour_rows.append(dict(group=gname, place=label, lo_mc=max(hlo[k], 0), hi_mc=hhi[k]))
            for m, f in enumerate(fixed):
                for k, label in enumerate(labels):
                    per_imp_rows.append(dict(imputation=m + 1, group=gname, place=label, hours=f[gname]["hours"][k]))
            ch = np.array([f[gname]["changes"] for f in fixed])
            cm, clo, chi = predictive_interval(ch)
            change_rows.append(dict(group=gname, diaries=int(groups[gname].sum()), changes=cm, lo_mc=clo, hi_mc=chi,
                                    changes_weighted_by_wght1=np.mean([f[gname]["changes_weighted"] for f in fixed])))
        shares_df = pd.DataFrame(share_rows)
        shares_df.to_csv(out / "germany_shares_by_slot.csv", index=False)
        day = shares_df.groupby(["group", "place"])["mean"].mean().mul(24).rename("hours_per_day").reset_index()
        day = day.merge(pd.DataFrame(hour_rows), on=["group", "place"], how="left")
        day.to_csv(out / "germany_hours_per_day.csv", index=False)
        pd.DataFrame(change_rows).to_csv(out / "germany_changes_per_diary.csv", index=False)
        pd.DataFrame(per_imp_rows).to_csv(out / "germany_hours_per_imputation.csv", index=False)
        main_hours = {g: np.mean([f[g]["hours"] for f in fixed], axis=0) for g in fixed[0]}
        timing["germany"] = round(time.time() - t_stage, 1)

        # ------------------------------------------------------------------- home activities (step 7)
        t_stage = time.time()
        acts, act_table = home_activities(data, full, codes)
        act_table.to_csv(out / "home_activities.csv", index=False)
        log(f"{len(acts)} home activities, for which code 13 is treated as missing in part B of the GSA: "
            + ", ".join(str(codes[a]) for a in acts))
        for c in full:
            E, S = data[c]["E"], data[c]["S"]
            n_ep = int((np.isin(E["a"], acts) & (E["cls"][E["nt"]] == OTHER)).sum())
            n_sl = int((np.isin(S["a"], acts) & (S["cls"] == OTHER)).sum())
            check(c, "code13_home_activity_episodes_excluded", n_ep, ">=0", True)
            check(c, "code13_home_activity_slots_excluded", n_sl, ">=0", True)
        tables13 = Tables(data, home_acts=acts)
        timing["home_activities"] = round(time.time() - t_stage, 1)

    # ----------------------------------------------------------------------- global sensitivity analysis (step 8)
    gsa_settings = None
    if not args.skip_gsa and full:
        # Part A: method choices, each combination trained on 12 donors and validated on the 13th, for every donor
        t_stage = time.time()
        factor_a = ["unit", "time", "age", "day_type", "assignment", "alpha"]
        n_specs = len(GSA_UNITS) * 2 * 2 * 2
        n_comb_a = n_specs * 2 * len(gsa_alphas)
        n_runs = n_comb_a * len(full)
        log(f"global sensitivity analysis, part A: {n_comb_a} combinations of unit, time, age, day type, "
            f"assignment and alpha, each scored on {len(full)} held-out donors ({n_runs} imputations)")
        tables_gsa = Tables(data)
        labels = CLASS_LABELS + [TRANSPORT_LABEL]
        loco_rows, err_rows, err_home_rows = [], [], []
        transfer = {}        # (unit, time, age, day, modal, alpha, donor) -> errors of that combination for that donor
        groups_h, obs_home = {}, {}   # recorded share at home by slot of each donor, all days, weekdays, weekend days
        for held in full:
            d_h = data[held]
            groups_h[held] = {g: m for g, m in day_groups(d_h).items() if m.sum() > 0}
            obs_home[held] = {g: weighted_slot_shares(d_h["C"][m], d_h["weight"][m], d_h["fold"])[:, HOME]
                              for g, m in groups_h[held].items()}
        done = 0
        for unit in GSA_UNITS:
            for time_ in (False, True):
                for age in (False, True):
                    for day in (False, True):
                        base = gsa_spec(unit, time_, age, day)
                        pool_g = Pool(tables_gsa, base, full)
                        for j, held in enumerate(full):
                            d_h = data[held]
                            mask = np.array([c != held for c in full], float)
                            for alpha in gsa_alphas:
                                model_g = pool_g.model(mask, alpha)
                                for modal in (False, True):
                                    # the same random numbers for every combination scored on this donor
                                    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, j]))
                                    imp = impute(dict(base, modal=modal), model_g, d_h, rng)
                                    o, _, hm, hr = score(d_h["C"], imp, d_h["weight"], d_h["fold"])
                                    key = dict(unit=unit, time=yes_no(time_), age=yes_no(age), day_type=yes_no(day),
                                               assignment="most likely" if modal else "draw", alpha=alpha, country=held)
                                    loco_rows.append(dict(
                                        key, accuracy=o["accuracy"], kappa=o["kappa"], macro_f1=o["macro_f1"],
                                        home_share_mae_pp=o["home_share_mae_pp"],
                                        home_share_bias_pp=100 * float((hm.imputed - hm.observed).mean()),
                                        hourly_tvd=o["hourly_tvd"], changes_observed=o["changes_observed"],
                                        changes_imputed=o["changes_imputed"],
                                        changes_error_pct=100 * (o["changes_imputed"] / o["changes_observed"] - 1)))
                                    # errors in the quantities estimated for Germany (part B), for every combination
                                    e_hours = (hr.imputed_hours - hr.recorded_hours).to_numpy()
                                    e_changes = float(o["changes_imputed"] - o["changes_observed"])
                                    e_home = {g: weighted_slot_shares(imp[m], d_h["weight"][m], d_h["fold"])[:, HOME]
                                              - obs_home[held][g] for g, m in groups_h[held].items()}
                                    transfer[unit, time_, age, day, modal, alpha, held] = (e_hours, e_changes, e_home)
                                    for k, label in enumerate(labels):
                                        err_rows.append(dict(key, output=label, error=e_hours[k]))
                                    err_rows.append(dict(key, output="changes per diary", error=e_changes))
                                    for g, e in e_home.items():
                                        err_home_rows.append(dict(key, group=g, **dict(zip(CLOCK, e))))
                                    done += 1
                        del pool_g
                        elapsed = time.time() - t_stage
                        log(f"part A: unit {unit}, time {yes_no(time_)}, age {yes_no(age)}, day type {yes_no(day)} "
                            f"done ({done}/{n_runs} imputations, about {elapsed / done * (n_runs - done) / 60:.0f} min left)")
                tables_gsa.cache.clear()      # the next unit and time setting needs other levels
        del tables_gsa
        loco_g = pd.DataFrame(loco_rows)
        loco_g.to_csv(out / "gsa_loco.csv", index=False)
        means = loco_g.groupby(factor_a)[GSA_METRICS].mean().reset_index()
        check("gsa", "part_a_combinations", len(means), n_comb_a, len(means) == n_comb_a, severity="fail")
        idx_a, S_a, ST_a = indices_frame(means[factor_a], means[GSA_METRICS].to_numpy(), GSA_METRICS)
        indices_checks("gsa part A", S_a, ST_a)
        idx_a.to_csv(out / "gsa_loco_indices.csv", index=False)
        means["pareto_optimal"] = pareto_optimal(means)
        means.to_csv(out / "gsa_loco_combinations.csv", index=False)
        n_pareto = int(means.pareto_optimal.sum())
        check("gsa", "part_a_pareto_optimal_combinations", n_pareto, ">=1", n_pareto >= 1,
              "combinations that no other one matches or beats on every evaluation metric")
        log(f"part A: {n_pareto} of {len(means)} combinations are Pareto-optimal over the {len(GSA_METRICS)} metrics")
        main_a = means[(means.unit == "markov") & (means.time == "yes") & (means.age == "yes")
                       & (means.day_type == "no") & (means.assignment == "draw") & (means.alpha == KAPPA)]
        loco_main = overall[(overall.coding == "full") & (overall.variant == MAIN_VARIANT)]["accuracy"].mean()
        if len(main_a) == 1 and np.isfinite(loco_main):   # same model as the main variant, other random draws
            diff = abs(float(main_a.accuracy.iloc[0]) - float(loco_main))
            check("gsa", "main_combination_vs_loco_accuracy_absdiff", round(diff, 5), "<0.005", diff < 0.005,
                  "same model as the main variant of step 4, other random draws")
        pd.DataFrame(err_rows).to_csv(out / "gsa_transfer_errors.csv", index=False)
        pd.DataFrame(err_home_rows).pipe(alpha_text).to_csv(out / "gsa_transfer_errors_home_by_slot.csv.gz",
                                                             index=False, compression="gzip", float_format="%.7g")
        del err_home_rows
        log("part A, share of variance (first-order / total Sobol' index) of the evaluation metrics:")
        print(idx_a.pivot(index="factor", columns="output", values="first_order").round(3).to_string())
        timing["gsa_part_a"] = round(time.time() - t_stage, 1)
        gsa_settings = dict(units=list(GSA_UNITS), alphas=list(gsa_alphas), part_a_factors=factor_a,
                            part_a_combinations=n_comb_a, part_a_donors=full)

        # Part B: German estimates over the same 144 method combinations as part A, crossed with the
        # coding assumption on code 13, the random draws and the transfer error of each donor under the
        # same method combination (Germany treated as exchangeable with the donors)
        if TARGET in data and tables13 is not None:
            t_stage = time.time()
            d = data[TARGET]
            factor_b = factor_a + ["code_13", "draws", "transfer_error"]
            de_groups = day_groups(d)
            german, uni_rows, uni_home = {}, [], []
            n_imp = 0
            for unit in GSA_UNITS:
                for time_ in (False, True):
                    for age in (False, True):
                        for day in (False, True):
                            spec_b = gsa_spec(unit, time_, age, day)
                            for code13 in (False, True):
                                merged = (tables13 if code13 else tables).model(spec_b, full, KAPPA).tables
                                for alpha in gsa_alphas:
                                    model_b = Model(spec_b["levels"], merged, alpha)
                                    for modal in (False, True):
                                        res = None
                                        for rep in range(n_draws):
                                            # stream `rep` of random numbers, the same for every combination of the
                                            # other factors; the most likely class does not depend on the draws
                                            if res is None or not modal:
                                                rng = np.random.default_rng(np.random.SeedSequence([SEED, 4, rep]))
                                                imp = impute(dict(spec_b, modal=modal), model_b, d, rng, keep_known=True)
                                                summs = group_summaries(imp, d, de_groups)
                                                res = (summs["all"]["hours"], summs["all"]["changes"],
                                                       {g: s_["shares"][:, HOME] for g, s_ in summs.items()})
                                                n_imp += 1
                                            german[unit, time_, age, day, modal, alpha, code13, rep] = res
                                            row = dict(unit=unit, time=yes_no(time_), age=yes_no(age),
                                                       day_type=yes_no(day), assignment="most likely" if modal else "draw",
                                                       alpha=alpha,
                                                       code_13="missing for home activities" if code13 else "as recorded",
                                                       draws=rep + 1)
                                            uni_rows.append(dict(row, **{lab: res[0][k] for k, lab in enumerate(labels)},
                                                                 **{"changes per diary": res[1]}))
                                            for g, sh in res[2].items():
                                                uni_home.append(dict(row, group=g, **dict(zip(CLOCK, sh))))
                    tables.cache.clear()          # the next unit and time setting needs other levels
                    tables13.cache.clear()
                    log(f"part B: unit {unit}, time {yes_no(time_)} done ({n_imp} German imputations, "
                        f"{(time.time() - t_stage) / 60:.0f} min)")
            n_uni = n_comb_a * 2 * n_draws
            check("gsa", "part_b_german_combinations", len(german), n_uni, len(german) == n_uni, severity="fail")
            pd.DataFrame(uni_rows).to_csv(out / "gsa_germany_universes.csv", index=False)
            pd.DataFrame(uni_home).pipe(alpha_text).to_csv(out / "gsa_germany_home_by_slot.csv.gz", index=False,
                                                          compression="gzip", float_format="%.7g")
            del uni_home

            groups_b = [g for g in GSA_GROUPS if all(g in v[2] for v in german.values())
                        and all(g in v[2] for v in transfer.values())]
            combo_rows, home_rows_b = [], {g: [] for g in groups_b}
            for (unit, time_, age, day, modal, alpha, code13, rep), (hours, changes, home) in german.items():
                for held in full:
                    e_hours, e_changes, e_home = transfer[unit, time_, age, day, modal, alpha, held]
                    h = np.maximum(hours - e_hours, 0.0)       # transport is recorded: its error is zero
                    combo_rows.append(dict(unit=unit, time=yes_no(time_), age=yes_no(age), day_type=yes_no(day),
                                           assignment="most likely" if modal else "draw", alpha=alpha,
                                           code_13="missing for home activities" if code13 else "as recorded",
                                           draws=rep + 1, transfer_error=held,
                                           **{lab: h[k] for k, lab in enumerate(labels)},
                                           **{"changes per diary": max(changes - e_changes, 0.0)}))
                    for g in groups_b:
                        home_rows_b[g].append(np.clip(home[g] - e_home[g], 0.0, 1.0))
            combos = pd.DataFrame(combo_rows)
            del combo_rows
            n_comb_b = n_uni * len(full)
            check("gsa", "part_b_combinations", len(combos), n_comb_b, len(combos) == n_comb_b, severity="fail")
            combos.pipe(alpha_text).to_csv(out / "gsa_germany_combinations.csv", index=False, float_format="%.7g")
            # the combinations whose method is Pareto-optimal in part A
            best = means.loc[means.pareto_optimal, factor_a].assign(_best=True)
            in_best = combos[factor_a].merge(best, on=factor_a, how="left")["_best"].fillna(False).to_numpy(bool)
            check("gsa", "part_b_combinations_with_pareto_optimal_method", int(in_best.sum()),
                  n_pareto * 2 * n_draws * len(full), int(in_best.sum()) == n_pareto * 2 * n_draws * len(full),
                  severity="fail")
            outputs_b = CLASS_LABELS + ["changes per diary"]       # transport is recorded, so it does not vary
            idx_b, S_b, ST_b = indices_frame(combos[factor_b], combos[outputs_b].to_numpy(), outputs_b)
            indices_checks("gsa part B", S_b, ST_b)
            idx_b.to_csv(out / "gsa_germany_indices.csv", index=False)
            idx_rows, unc_rows, unc_best_rows = [], [], []
            # the main specification (Markov chain, time, age, random draw, alpha 5), first stream of draws
            main_key = ("markov", True, True, False, False, KAPPA, False, 0)
            for g in groups_b:
                H = np.array(home_rows_b.pop(g))
                S_h, ST_h = factorial_indices(combos[factor_b], H)
                indices_checks(f"gsa part B home by slot, {g}", S_h, ST_h)
                idx_rows += [dict(group=g, slot=s_, clock=CLOCK[s_], factor=f, first_order=S_h[i, s_], total=ST_h[i, s_])
                             for s_ in range(SLOTS) for i, f in enumerate(factor_b)]
                unc_rows += [dict(group=g, slot=s_, clock=CLOCK[s_], **quantile_rows(H[:, s_], german[main_key][2][g][s_]))
                             for s_ in range(SLOTS)]
                unc_best_rows += [dict(group=g, slot=s_, clock=CLOCK[s_],
                                       **quantile_rows(H[in_best, s_], german[main_key][2][g][s_]))
                                  for s_ in range(SLOTS)]
                del H
            pd.DataFrame(idx_rows).to_csv(out / "gsa_germany_indices_home_by_slot.csv", index=False)
            pd.DataFrame(unc_rows).to_csv(out / "gsa_germany_uncertainty_home_by_slot.csv", index=False)
            pd.DataFrame(unc_best_rows).to_csv(out / "gsa_germany_uncertainty_home_by_slot_pareto.csv", index=False)

            for sel, name in ((np.ones(len(combos), bool), "gsa_germany_uncertainty.csv"),
                              (in_best, "gsa_germany_uncertainty_pareto.csv")):
                unc = [dict(output=lab, **quantile_rows(combos.loc[sel, lab], german[main_key][0][k]))
                       for k, lab in enumerate(labels)]
                unc.append(dict(output="changes per diary",
                                **quantile_rows(combos.loc[sel, "changes per diary"], german[main_key][1])))
                pd.DataFrame(unc).to_csv(out / name, index=False)
            if main_hours:
                diff_home = abs(float(german[main_key][0][HOME]) - float(main_hours["all"][HOME]))
                sd_home = float(np.std([f["all"]["hours"][HOME] for f in fixed], ddof=1)) if len(fixed) > 1 else 0.0
                limit = max(4 * sd_home, 0.01)
                check("gsa", "main_combination_vs_fixed_pool_home_hours_absdiff", round(diff_home, 4),
                      f"<{limit:.3f}", diff_home < limit,
                      "same model as the German imputations of step 6, other random draws "
                      "(limit: four standard deviations across those imputations)")
            log("part B, share of variance (first-order Sobol' index) of the German estimates:")
            print(idx_b.pivot(index="factor", columns="output", values="first_order").round(3).to_string())
            timing["gsa_part_b"] = round(time.time() - t_stage, 1)
            gsa_settings.update(part_b_factors=factor_b, part_b_draws=n_draws, part_b_german_combinations=n_uni,
                                part_b_german_imputations=n_imp, part_b_combinations=n_comb_b,
                                part_a_pareto_optimal_combinations=n_pareto)

    # ----------------------------------------------------------------------- settings, checks, README, bundle
    # Files that an earlier run left in the output folder and this run did not overwrite, such as
    # outputs that this version no longer writes: kept out of the bundle and of the export by
    # export_results.py, so that no output of an earlier run passes for one of this run.
    rewritten_last = ("checks.csv", "run_settings.json", "README.txt")
    stale = sorted(f.name for f in out.iterdir()
                   if f.is_file() and f.name not in rewritten_last and f.stat().st_mtime < run_start - 2)
    check("run", "files_of_earlier_runs_in_output_folder", len(stale), 0, not stale,
          "kept out of the bundle and the export: " + ", ".join(stale) if stale else "")
    checks = pd.DataFrame(CHECKS)
    checks.to_csv(out / "checks.csv", index=False)
    n_warn, n_fail = int((checks.status == "warn").sum()), int((checks.status == "fail").sum())
    log(f"checks: {len(checks)} run, {n_warn} warnings, {n_fail} failures (see checks.csv)")
    settings = dict(version=VERSION, target=TARGET, n_imputations=m_imputations, kappa=KAPPA, seed=SEED,
                    place_codes=PLACE_CODES, transport_codes=TRANSPORT_CODES, travel_activity=TRAVEL_ACTIVITY,
                    transport_rule="a slot is transport if its activity is a trip or a mode of transport is recorded",
                    age_top=AGE_TOP, main_variant=MAIN_VARIANT,
                    variants={k: {kk: (vv if kk != "levels" else [list(x) for x in vv])
                                  for kk, vv in v.items()} for k, v in variants.items()},
                    donors_full=full, scored_partial=partial, files={k: str(v) for k, v in files.items()},
                    repeat_imputations=m_rep,
                    home_activity_share=HOME_ACTIVITY_SHARE, home_activity_min_slots=HOME_ACTIVITY_MIN_SLOTS,
                    home_activity_max_code13=HOME_ACTIVITY_MAX_CODE13, global_sensitivity_analysis=gsa_settings,
                    min_respondents_per_place_score=MIN_RESPONDENTS, files_of_earlier_runs=stale,
                    checks_warnings=n_warn, checks_failures=n_fail, timing_seconds=timing)
    (out / "run_settings.json").write_text(json.dumps(settings, indent=2))
    (out / "README.txt").write_text(
        f"Outputs of location_imputation.py version {VERSION}.\n\n"
        "Every slot with a recorded mode of transport counts as transport (trips and, for instance,\n"
        "reading on a train); transport is never imputed. The location classes are the eight places.\n\n"
        "AGGREGATE (can be shared):\n"
        "  checks.csv (data and consistency checks), run_settings.json.\n"
        "  Leave-one-country-out (each donor imputed from the other 12; AT, FR and LU from all 13):\n"
        "    loco_overall.csv and loco_summary.csv (metrics per country and variant, and their means),\n"
        "    loco_by_class.csv (per-place scores; withheld where fewer than "
        f"{MIN_RESPONDENTS} respondents stand behind them\n"
        "    in any country of the coding group, as Eurostat's publication guidelines for the SUF require),\n"
        "    loco_by_class_means.csv (their means over countries), loco_home_by_slot.csv,\n"
        "    loco_hours_per_day.csv (weighted hours per day by place, recorded and imputed, every variant).\n"
        "  Repeated imputations of the held-out countries: loco_repeated_imputations.csv and\n"
        "    loco_repeated_imputations_summary.csv (coverage of plain 95% intervals), loco_intervals_home_by_slot.csv.\n"
        "  Germany: germany_shares_by_slot.csv, germany_hours_per_day.csv, germany_changes_per_diary.csv,\n"
        "    germany_hours_per_imputation.csv (main variant, all 13 donors; lo_mc/hi_mc: plain 95% interval,\n"
        "    Monte Carlo variation only), home_activities.csv (activities for the code-13 factor).\n"
        "  Global sensitivity analysis (full factorial, exact first-order and total Sobol' indices):\n"
        "    gsa_loco.csv (part A: metrics of every combination of method choices and held-out donor),\n"
        "    gsa_loco_indices.csv (indices of the metrics averaged over the donors), gsa_loco_combinations.csv\n"
        "    (those averages for each of the 144 combinations, with the Pareto-optimal ones flagged),\n"
        "    gsa_transfer_errors.csv and gsa_transfer_errors_home_by_slot.csv.gz (imputed minus recorded for each\n"
        "    donor and combination, the levels of the transfer-error factor), gsa_germany_universes.csv and\n"
        "    gsa_germany_home_by_slot.csv.gz (German estimates of every combination of the method choices of part A,\n"
        "    coding assumption and stream of random draws), gsa_germany_combinations.csv (the same minus each\n"
        "    donor's transfer error under the same method choices), gsa_germany_indices.csv,\n"
        "    gsa_germany_indices_home_by_slot.csv, gsa_germany_uncertainty.csv and\n"
        "    gsa_germany_uncertainty_home_by_slot.csv (distribution over all combinations), and the same two files\n"
        "    ending in _pareto.csv (over the combinations whose method choices are Pareto-optimal in part A).\n"
        "MICRODATA (keep on your machine, do not send): PRIVATE_germany_imputed_codes.npz, imputation x\n"
        "  diary x slot in HETUS codes (transport keeps its recorded mode; 30 = trip without a recorded mode),\n"
        "  and PRIVATE_germany_ids.csv (row order of the diaries). Take uncertainty statements from the\n"
        "  global sensitivity analysis, not from the spread across these imputations.\n"
        "imputation_out_to_send.zip, next to this folder, holds the aggregate files only.\n"
        + ("NOT FROM THIS RUN (left by an earlier run, not in the bundle or the export): "
           + ", ".join(stale) + "\n" if stale else ""))
    bundle = out.parent / "imputation_out_to_send.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.iterdir()):
            if f.is_file() and not f.name.startswith("PRIVATE_") and f.name not in stale:
                zf.write(f, arcname=f"imputation_out/{f.name}")
    log(f"done, outputs in {out}")
    log(f"send this file (aggregates only, no microdata): {bundle}")


if __name__ == "__main__":
    main()
