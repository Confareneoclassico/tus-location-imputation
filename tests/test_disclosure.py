#!/usr/bin/env python3
"""Checks of the disclosure control of the per-place scores (src/location_imputation.py).

Run with  python tests/test_disclosure.py  (no data needed, a few seconds). Exits with 1 on failure.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import location_imputation as li  # noqa: E402

FAILED = []


def expect(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": {detail}"))
    if not ok:
        FAILED.append(name)


# 1. Respondents, not diaries or slots, are counted: 40 respondents with two diaries each; the
#    first 15 respondents spend slots 10 to 19 of both diaries at place 1, the rest stay at home.
n_resp, T = 40, 144
person = np.repeat(np.arange(n_resp), 2)
C = np.zeros((2 * n_resp, T), np.int8)
C[:30, 10:20] = 1
C[:, :2] = li.TRAVEL_SLOT                      # travel slots are not counted
counts = li.respondents_by_class(C, person, li.N_CLASSES)
expect("respondents counted once however many diaries and slots", counts[0] == 40 and counts[1] == 15,
       f"got {counts[:2]}")
expect("places without slots have no respondents", counts[2:].sum() == 0, f"got {counts}")

# 2. person_codes uses HID and PID; without them respondents are counted conservatively
ids = pd.DataFrame({"HID": [7, 7, 7, 9], "PID": [1, 1, 2, 1], "DIARY": [1, 2, 1, 1]}).astype("Int64")
codes, found = li.person_codes(ids, 4)
expect("household and person identifiers define respondents", found and len(set(codes)) == 3, f"got {codes}")
codes, found = li.person_codes(ids.drop(columns="PID"), 4)
expect("without PID no respondent codes", (not found) and codes is None, f"got {codes}")
counts = li.respondents_by_class(C, None, li.N_CLASSES)
expect("without identifiers, diaries divided by 7 and rounded up", counts[0] == 12 and counts[1] == 5,
       f"got {counts[:2]}")

# 3. score() reports respondents recorded at and imputed to each place
imp = np.where(C >= 0, 0, C).astype(np.int8)
imp[:4, 10:20] = 1                             # place 1 imputed for the first two respondents only
_, bc, _, _ = li.score(C, imp, np.ones(len(C)), np.arange(li.N_CLASSES), person=person, count_respondents=True)
row = bc.set_index("place").loc[li.CLASS_LABELS[1]]
expect("score() counts respondents behind recall and precision",
       row.respondents_recorded == 15 and row.respondents_imputed == 2,
       f"got {row.respondents_recorded}, {row.respondents_imputed}")

# 4. publish_by_class withholds a score for every country of the group when any country has too
#    few respondents, keeps the means over all countries, and blanks small respondent counts
rows = []
for country, few in (("AA", False), ("BB", True), ("CC", False)):
    for place, recall, precision in (("home", 0.9, 0.8), ("second home", 0.1, 0.2)):
        small = few and place == "second home"        # recorded for 20 respondents, never imputed
        rows.append(dict(place=place, observed_share=0.5, imputed_share=0.0 if small else 0.5,
                         recall=recall + 0.01 * len(rows),
                         precision=np.nan if small else precision, f1=0.5, scored_slots=1000, country=country,
                         variant="main", respondents_recorded=20 if small else 500,
                         respondents_imputed=0 if small else 400))
bc = pd.DataFrame(rows)
schemes = {"AA": "full", "BB": "full", "CC": "full"}
pub, means = li.publish_by_class(bc, schemes)
second = pub[pub.place == "second home"]
home = pub[pub.place == "home"]
expect("recall withheld in every country of the group", second.recall.isna().all() and second.f1.isna().all(),
       second.to_string())
expect("slot counts withheld with the recall", second.scored_slots.isna().all())
expect("precision kept where no country has 1 to 49 respondents imputed",
       second.precision.notna().sum() == 2, second.precision.to_string())
pub2, _ = li.publish_by_class(bc.assign(respondents_imputed=np.where(
    (bc.country == "CC") & (bc.place == "home"), 30, bc.respondents_imputed)), schemes)
expect("precision withheld in every country when one has 1 to 49 respondents imputed",
       pub2[pub2.place == "home"].precision.isna().all() and pub2[pub2.place == "home"].recall.notna().all())
expect("other places untouched", home.recall.notna().all() and home.precision.notna().all())
expect("respondent counts below the threshold blanked",
       np.isnan(second.respondents_recorded.iloc[1]) and second.respondents_recorded.iloc[[0, 2]].eq(500).all())
m = means.set_index("place").loc["second home"]
expect("means over countries computed from all scores",
       np.isclose(m.recall, bc[bc.place == "second home"].recall.mean()) and m.countries == 3, m.to_string())
li.CHECKS.clear()
li.disclosure_checks(pub, schemes)
status = {c["check"]: c["status"] for c in li.CHECKS}
expect("disclosure checks pass on published scores",
       status.get("smallest_published_respondent_count") == "ok"
       and status.get("scores_without_enough_respondents") == "ok"
       and status.get("groups_partly_withheld") == "ok", str(status))
li.CHECKS.clear()
partial = pub.copy()
partial.loc[(partial.place == "second home") & (partial.country == "AA"), "recall"] = 0.1
li.disclosure_checks(partial, schemes)
status = {c["check"]: c["status"] for c in li.CHECKS}
expect("disclosure check catches a group withheld in one country only",
       status.get("groups_partly_withheld") == "fail" and status.get("scores_without_enough_respondents") == "ok",
       str(status))

# 5. ... and fail when a small score slips through
leak = pub.copy()
leak.loc[leak.place == "second home", "recall"] = 0.1
li.CHECKS.clear()
li.disclosure_checks(leak)
status = {c["check"]: c["status"] for c in li.CHECKS}
expect("disclosure check catches a published score on too few respondents",
       status.get("scores_without_enough_respondents") == "fail", str(status))

print(f"{'all' if not FAILED else len(FAILED)} disclosure tests {'passed' if not FAILED else 'failed'}")
sys.exit(1 if FAILED else 0)
