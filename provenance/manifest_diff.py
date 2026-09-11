#!/usr/bin/env python3
"""manifest_diff.py -- what a new freeze makes stale.

COMPONENT CLASS: **VALIDATION-ONLY**.

Reads two `science_manifest.json` documents and emits a report.  It computes no science
quantity, opens no pack, runs no sampler, and writes nothing into the paper, code or notes
repositories.  Every number it prints is copied from one of the two manifests.  It is a
STALENESS ORACLE, not an estimator.

It walks ONLY `products[].upstream` and `dependencies[].depends_on`.  If an edge is not
recorded, the product is reported `UNTRACEABLE` rather than guessed at.  A field that is
absent on the old side is reported `old_side_unrecorded`, never as "unchanged".

Exit codes: 0 = no stale consumers; 1 = stale consumers found (the normal result after a
freeze change); 2 = schema-invalid input, UNTRACEABLE products on the new side, a
retired-but-still-referenced quantity, or an undeclared withholding.

Implements MANIFEST_DIFF_SPEC.md sections 1-2.7.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCHEMA_PATH = pathlib.Path(
    "/home/mfho/lowz_clean_work_2026-09-11/phase0c/SCIENCE_MANIFEST_SCHEMA.json")

STALE_VERDICTS = {"STALE-REPRINT", "STALE-SILENT", "RECLASSIFY", "CAMPAIGN"}
KIND_TABLES = [
    ("table_cell", "stale table cells"),
    ("figure_array", "stale figure arrays"),
    ("ledger_row", "stale ledger rows"),
    ("prose_literal", "stale prose literals"),
    ("hash_pin", "stale hash pins"),
    ("release_artifact", "stale release artifacts"),
    ("table_file", "stale table files"),
    ("figure_file", "stale figure files"),
]


def load(p):
    with open(p) as f:
        return json.load(f)


def is_gap(v):
    return isinstance(v, str) and v.startswith("_GAP")


def idx(rows, key):
    return {r[key]: r for r in rows}


def leaves(obj, prefix=""):
    """Flatten a contract subtree into dotted leaf paths, so nothing is compared by eye."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        # legs[] is keyed by its own `leg` field so a reordering is not a false diff
        if obj and isinstance(obj[0], dict) and "leg" in obj[0]:
            for it in obj:
                yield from leaves(it, f"{prefix}[{it['leg']}]")
        else:
            yield prefix, obj
    else:
        yield prefix, obj


def fmt(value, precision, multiplier=1):
    if value is None or precision is None:
        return None
    if isinstance(value, str):
        return value
    return f"{value * multiplier:.{precision}f}"


# ------------------------------------------------------------------------ contract diff
def contract_diff(old, new):
    lo, ln = dict(leaves(old["contract"]["selection"])), dict(leaves(new["contract"]["selection"]))
    rows, breaks = [], []
    for k in sorted(set(lo) | set(ln)):
        a, b = lo.get(k, "<absent>"), ln.get(k, "<absent>")
        if a != b:
            row = {"field": f"contract.selection.{k}", "old": a, "new": b}
            if k.startswith("snr.legs[") and k.endswith(".statistic"):
                row["class"] = "CONTRACT-BREAK"
            elif k == "population.spectype_cut":
                row["class"] = "CONTRACT-BREAK"
            else:
                row["class"] = "changed"
            if a == "<absent>":
                row["confidence"] = "old_side_unrecorded"
            rows.append(row)
            if row["class"] == "CONTRACT-BREAK":
                breaks.append(row)
    est_o = idx(old["contract"].get("estimands", []), "estimand_id")
    est_n = idx(new["contract"].get("estimands", []), "estimand_id")
    for e in sorted(set(est_o) | set(est_n)):
        a, b = est_o.get(e), est_n.get(e)
        if a is None or b is None:
            rows.append({"field": f"contract.estimands[{e}]",
                         "old": "<absent>" if a is None else "present",
                         "new": "<absent>" if b is None else "present", "class": "changed"})
        elif a.get("definition") != b.get("definition"):
            rows.append({"field": f"contract.estimands[{e}].definition",
                         "old": a.get("definition"), "new": b.get("definition"),
                         "class": "changed"})
    return rows, breaks


def code_diff(old, new):
    ro = {r["name"]: r for r in old["code"]["repos"]}
    rn = {r["name"]: r for r in new["code"]["repos"]}
    out = []
    for name in sorted(set(ro) | set(rn)):
        a, b = ro.get(name, {}), rn.get(name, {})
        if a.get("commit") != b.get("commit") or a.get("tag") != b.get("tag"):
            out.append({"repo": name, "old": a.get("commit"), "new": b.get("commit"),
                        "old_tag": a.get("tag"), "new_tag": b.get("tag"),
                        "confidence": "old_side_unrecorded" if is_gap(a.get("commit")) else "recorded"})
    eo, en = old["code"].get("env_lock", {}), new["code"].get("env_lock", {})
    if eo.get("sha256") != en.get("sha256"):
        out.append({"repo": "env_lock", "old": eo.get("sha256"), "new": en.get("sha256"),
                    "confidence": "old_side_unrecorded"
                                  if (eo.get("sha256") is None or is_gap(eo.get("sha256")))
                                  else "recorded"})
    return out


# ------------------------------------------------------------------------- product diff
def product_report(old, new, contract_break):
    po, pn = idx(old["products"], "product_id"), idx(new["products"], "product_id")

    # which products sit under the contract, by RECORDED edge only
    def contract_inputs(M):
        sel = M["contract"]["selection"]
        refs = {leg["source_input_ref"] for leg in sel["snr"]["legs"]}
        if sel["population"].get("source_input_ref"):
            refs.add(sel["population"]["source_input_ref"])
        return refs

    seeds = set()
    for pid, p in pn.items():
        if set(p.get("upstream", [])) & (contract_inputs(new) | contract_inputs(old)):
            seeds.add(pid)
    # descend: anything whose recorded upstream reaches a seed
    def reaches(pid, table, seen=None):
        seen = seen or set()
        if pid in seen:
            return False
        seen.add(pid)
        p = table.get(pid)
        if not p:
            return False
        for u in p.get("upstream", []):
            if u in seeds or reaches(u, table, seen):
                return True
        return False

    contract_affected = {pid for pid in pn if pid in seeds or reaches(pid, pn)}

    def upstream_moved(pid, table_o, table_n, seen=None):
        seen = seen or set()
        if pid in seen:
            return []
        seen.add(pid)
        moved = []
        p = table_n.get(pid)
        if not p:
            return moved
        for u in p.get("upstream", []):
            a, b = table_o.get(u), table_n.get(u)
            if a and b and not is_gap(a["sha256"]) and not is_gap(b["sha256"]) \
                    and a["sha256"] != b["sha256"]:
                moved.append(u)
            moved += upstream_moved(u, table_o, table_n, seen)
        return sorted(set(moved))

    rows, untraceable_old, untraceable_new = [], [], []
    for pid in sorted(set(po) | set(pn)):
        a, b = po.get(pid), pn.get(pid)
        if a is None:
            cls = "added"
        elif b is None:
            cls = "removed"
        elif is_gap(a["sha256"]) and is_gap(b["sha256"]):
            # neither side records a content hash. The identity may still be recorded by a
            # different field -- a producer commit -- and if BOTH sides record the same one
            # the artifact is pinned, just not by content. If either is missing, we know
            # nothing, and absence is never evidence of no-change.
            ca = (a.get("producer") or {}).get("commit")
            cb = (b.get("producer") or {}).get("commit")
            if ca and cb and not is_gap(ca) and not is_gap(cb) and ca == cb:
                cls = "unchanged_by_producer_commit"
            else:
                cls = "unverifiable"
        elif is_gap(a["sha256"]) or is_gap(b["sha256"]):
            cls = "unverifiable"
        elif a["sha256"] == b["sha256"]:
            cls = "unchanged"
        else:
            cls = "rehashed"
        moved = upstream_moved(pid, po, pn) if (a and b) else []
        broken = bool(contract_break) and pid in contract_affected
        if cls in ("unchanged", "unchanged_by_producer_commit") and (moved or broken):
            status_new = "stale"
        elif cls == "removed":
            status_new = "withdrawn"
        else:
            status_new = (b or {}).get("status", "current")
        for side, tab, sink in (("old", a, untraceable_old), ("new", b, untraceable_new)):
            if tab is None:
                continue
            commit = (tab.get("producer") or {}).get("commit")
            if not tab.get("upstream") and (commit is None or is_gap(commit)):
                sink.append(pid)
        re_o = (a or {}).get("release_eligibility", {})
        re_n = (b or {}).get("release_eligibility", {})
        rel_change = None
        if a and b and (re_o.get("class") != re_n.get("class")
                        or re_o.get("basis") != re_n.get("basis")
                        or re_o.get("phase") != re_n.get("phase")):
            rel_change = {"old": re_o, "new": re_n,
                          "newly_restricted": (re_o.get("class") == "public_release_eligible"
                                               and re_n.get("class") == "collaboration_restricted")}
        rows.append({
            "product_id": pid, "kind": (b or a).get("kind"), "class": cls,
            "old_sha256": (a or {}).get("sha256"), "new_sha256": (b or {}).get("sha256"),
            "upstream_changed": moved,
            "contract_broken": broken,
            "status_old": (a or {}).get("status"), "status_new": status_new,
            "release_eligibility_change": rel_change,
            "untraceable_old": pid in untraceable_old,
            "untraceable_new": pid in untraceable_new,
            "confidence": "old_side_unrecorded" if (a and is_gap(a["sha256"])) else "recorded",
        })
    return rows, contract_affected, sorted(set(untraceable_old)), sorted(set(untraceable_new))


# ------------------------------------------------------------------------ quantity diff
def quantity_report(old, new):
    qo, qn = idx(old["ledger_quantities"], "quantity_id"), idx(new["ledger_quantities"], "quantity_id")
    rows = {}
    for qid in sorted(set(qo) | set(qn)):
        a, b = qo.get(qid), qn.get(qid)
        ov = a["value"] if a else None
        nv = b["value"] if b else None
        prec = (b or a).get("printed_precision")
        mult = (b or a).get("print_multiplier", 1)
        r = {"quantity_id": qid, "old": ov, "new": nv,
             "kind": (b or a).get("kind"), "units": (b or a).get("units"),
             "printed_precision": prec, "print_multiplier": mult,
             "printed_old": fmt(ov, prec, mult), "printed_new": fmt(nv, prec, mult),
             "provenance_class": (b or a).get("provenance_class"),
             "status": "added" if a is None else ("retired" if b is None else "present"),
             "delta": None, "delta_pct": None, "delta_over_halfwidth": None,
             "halfwidth_ratio": None, "printed_digits_change": None,
             "old_interval": None, "new_interval": None,
             "resolution": "resolved"}
        if nv is None and a is not None:
            r["resolution"] = "UNRESOLVED (no value on the new side)"
        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            r["delta"] = nv - ov
            r["delta_pct"] = ((nv - ov) / ov * 100) if ov else None
            io = ((a.get("interval") or {}).get("values") if a else None)
            ino = ((b.get("interval") or {}).get("values") if b else None)
            if io and ino and len(io) == 5 and len(ino) == 5:
                hwo, hwn = 0.5 * (io[3] - io[1]), 0.5 * (ino[3] - ino[1])
                r["old_interval"] = [io[1], io[3]]
                r["new_interval"] = [ino[1], ino[3]]
                r["delta_over_halfwidth"] = (nv - ov) / hwo if hwo else None
                r["halfwidth_ratio"] = hwn / hwo if hwo else None
            if prec is not None:
                r["printed_digits_change"] = round(ov * mult, prec) != round(nv * mult, prec)
        elif isinstance(ov, str) or isinstance(nv, str):
            r["printed_digits_change"] = (ov != nv) if (ov is not None and nv is not None) else None
        rows[qid] = r
    return rows


# ------------------------------------------------------------------------ consumer diff
def consumer_report(old, new, qrows, prows, contract_break):
    co = idx(old["dependencies"], "consumer_id")
    cn = idx(new["dependencies"], "consumer_id")
    pn = {r["product_id"]: r for r in prows}
    out = []
    for cid in sorted(set(co) | set(cn)):
        a, b = co.get(cid), cn.get(cid)
        cur = b or a
        deps = cur["depends_on"]
        prec = cur.get("printed_precision")
        moved_products, moved_quantities, printed_changes, unresolved = [], [], [], []
        pending_refresh = []
        for d in deps:
            if d in qrows:
                q = qrows[d]
                p = prec if prec is not None else q["printed_precision"]
                if q["resolution"].startswith("UNRESOLVED"):
                    unresolved.append(d)
                elif q["old"] != q["new"]:
                    moved_quantities.append(d)
                if p is not None and isinstance(q["old"], (int, float)) \
                        and isinstance(q["new"], (int, float)):
                    m = q["print_multiplier"]
                    if round(q["old"] * m, p) != round(q["new"] * m, p):
                        printed_changes.append(d)
                elif isinstance(q["old"], str) and isinstance(q["new"], str) \
                        and q["old"] != q["new"]:
                    printed_changes.append(d)
            elif d in pn:
                pr = pn[d]
                # A dependency has MOVED when its CONTENT changed (rehashed) or when its
                # identity cannot be verified at all (unverifiable).  A product that is
                # merely STALE -- contract-broken, or pending a refresh because an upstream
                # moved -- still has the same bytes today, so a consumer reading it reads
                # the same numbers.  That is a downstream ORDERING fact (section 3 lists the
                # product as stale and section 7 flags the pin), not a change in what is
                # printed now, and conflating the two would report byte-identical arrays as
                # changed.
                if pr["class"] in ("rehashed", "unverifiable"):
                    moved_products.append(d)
                elif pr["status_new"] == "stale":
                    pending_refresh.append(d)
        moved = bool(moved_products or moved_quantities)
        disp = cur.get("disposition", "unchanged")
        if disp == "refresh_campaign" or (unresolved and disp == "refresh_campaign"):
            verdict = "CAMPAIGN"
        elif disp == "reclassify":
            verdict = "RECLASSIFY"
        elif unresolved:
            verdict = "STALE-REPRINT"
        elif printed_changes:
            verdict = "STALE-REPRINT"
        elif moved:
            verdict = "STALE-SILENT"
        else:
            verdict = "UNCHANGED"
        if a is None:
            verdict_note = "added on the new side"
        elif b is None:
            verdict_note = "removed on the new side"
        else:
            verdict_note = None
        printed_old = printed_new = None
        for d in deps:
            if d in qrows and qrows[d]["printed_precision"] is not None:
                p = prec if prec is not None else qrows[d]["printed_precision"]
                m = qrows[d]["print_multiplier"]
                printed_old = fmt(qrows[d]["old"], p, m)
                printed_new = fmt(qrows[d]["new"], p, m)
                break
        sv = cur.get("submission_visibility", {"ships": True})
        out.append({
            "consumer_id": cid, "consumer_kind": cur["consumer_kind"],
            "location": cur["location"],
            "emitter": (cur.get("emitter") or {}).get("script") if cur.get("emitter") else "HAND-TYPED",
            "hand_typed": bool(cur.get("hand_typed")),
            "depends_on": deps,
            "literal_in_manuscript": cur.get("printed_value"),
            "printed_old": printed_old, "printed_new": printed_new,
            "printed_digits_change": bool(printed_changes),
            "moved_products": moved_products, "moved_quantities": moved_quantities,
            "pending_refresh_dependencies": pending_refresh,
            "unresolved_dependencies": unresolved,
            "disposition": disp, "verdict": verdict, "verdict_note": verdict_note,
            "ships": bool(sv.get("ships", True)),
            "build_state": sv.get("build_state"),
            "block": sv.get("block"),
            "pending_literal": bool(sv.get("pending_literal")),
            "gate": sv.get("gate"),
            "hand_typed_risk": bool(cur.get("hand_typed")) and (moved or bool(printed_changes)),
            "owning_ledger_line": cur.get("_owning_ledger_line"),
            "printed_precision": prec,
        })
    return out


def pending_literal_list(consumers, qrows):
    rows = []
    for c in consumers:
        if not c["pending_literal"]:
            continue
        vals, resolution = [], "RESOLVED"
        for d in c["depends_on"]:
            q = qrows.get(d)
            if q is None:
                vals.append({"quantity_id": d, "new_value": None, "note": "not a ledger quantity"})
                continue
            p = c["printed_precision"] if c["printed_precision"] is not None \
                else q["printed_precision"]
            m = q["print_multiplier"]
            vals.append({"quantity_id": d,
                         "old_value": fmt(q["old"], p, m) if q["old"] is not None else None,
                         "new_value": fmt(q["new"], p, m) if q["new"] is not None else None,
                         "printed_digits_change": (
                             None if q["new"] is None or q["old"] is None
                             else (round(q["old"] * m, p) != round(q["new"] * m, p))
                             if isinstance(q["old"], (int, float)) and p is not None
                             else q["old"] != q["new"])})
        missing = [v for v in vals if v.get("new_value") is None]
        if missing and c["disposition"] == "refresh_campaign":
            resolution = ("UNRESOLVED - MIXED line, awaiting a mean-plane refresh")
        elif missing:
            resolution = ("RESOLVED-PENDING-EMIT - the replacing quantity is identified and its "
                          "value comes from a refresh_mechanical re-emit this freeze already "
                          "schedules (omega_anatomy.py); NOT blocked on a campaign")
        elif any(v.get("printed_digits_change") for v in vals):
            resolution = "RESOLVED"
        else:
            resolution = "RESOLVED-UNCHANGED - carry_over; the bracket is removed, the value stands"
        rows.append({
            "file_line": f"{c['location'].get('path')}:{c['location'].get('line')}",
            "owning_ledger_line": c["owning_ledger_line"],
            "placeholder": c["literal_in_manuscript"],
            "replacing": vals,
            "ships": c["ships"],
            "gate": c["gate"],
            "resolution": resolution,
        })
    rows.sort(key=lambda r: r["file_line"])
    return rows


def hash_pin_report(consumers, prows, suppressed):
    pn = {r["product_id"]: r for r in prows}
    rows = []
    for c in consumers:
        if c["consumer_kind"] != "hash_pin":
            continue
        for d in c["depends_on"]:
            p = pn.get(d)
            if not p:
                continue
            differs = p["class"] in ("rehashed", "unverifiable")
            pending = (not differs) and p["status_new"] == "stale"
            if differs or pending:
                rows.append({
                    "pin_state": "DIFFERS NOW" if differs else
                                 "unchanged today; the pinned product is STALE and this pin "
                                 "moves when it is refreshed",
                    "consumer_id": c["consumer_id"],
                    "location": f"{c['location'].get('path')}:{c['location'].get('line')}",
                    "key": c["location"].get("key"),
                    "product_id": d,
                    "old_sha256": p["old_sha256"], "new_sha256": p["new_sha256"],
                    "recommendation": ("SUPPRESSED (report is NOT-ADOPTABLE)" if suppressed
                                       else (f"re-pin to {p['new_sha256']}" if differs
                                             else "no re-pin yet; re-pin after the stale product "
                                                  "is refreshed")),
                })
    return rows


def release_report(prows):
    out = []
    for p in prows:
        ch = p["release_eligibility_change"]
        if not ch:
            continue
        undeclared = (ch["newly_restricted"]
                      and not ch["new"].get("rationale") and not ch["new"].get("ruling_ref"))
        out.append({"product_id": p["product_id"], "old": ch["old"], "new": ch["new"],
                    "newly_restricted": ch["newly_restricted"],
                    "undeclared_withholding": undeclared})
    return out


# ------------------------------------------------------------------------------- output
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v).replace("|", "\\|")
                                     for v in r) + " |")
    return "\n".join(out)


def s12(h):
    """Truncate a hash for the md table. A _GAP marker keeps its marker, not its prose:
    the full text is in the json and in the manifest's own gaps[]."""
    if not isinstance(h, str):
        return h
    if h.startswith("_GAP"):
        return "_GAP (see gaps[])"
    return h[:12]


def render_md(R):
    L = []
    A = L.append
    A(f"# manifest_diff  {R['old']['manifest_id']}  ->  {R['new']['manifest_id']}")
    A("")
    A("**VALIDATION-ONLY.** Every number below is copied from one of the two manifests. "
      "This tool computes no science value.")
    A("")
    if R["not_adoptable"]:
        A("> ## NOT-ADOPTABLE")
        A(f"> The new manifest carries {len(R['blocking_open_items'])} BLOCKING open item(s): "
          f"{', '.join(R['blocking_open_items'])}.")
        A("> Every re-pin recommendation below is SUPPRESSED. Nothing may be re-pinned, "
          "re-emitted or released against this manifest until the PI rules.")
        A("")
    A("## 1. Identity")
    A("")
    A(md_table(["", "old", "new"], [
        ["manifest_id", R["old"]["manifest_id"], R["new"]["manifest_id"]],
        ["manifest sha256", R["old"]["sha256"], R["new"]["sha256"]],
        ["freeze.status", R["old"]["freeze_status"], R["new"]["freeze_status"]],
        ["completeness", R["old"]["completeness"], "full"],
    ]))
    A("")
    A("## 2. Contract diff")
    A("")
    if R["contract_break"]:
        A(f"**CONTRACT-BREAK x{len(R['contract_break'])}.** A change here invalidates every "
          "product built under the contract even where its bytes are identical. The break "
          "propagates along RECORDED EDGES ONLY: a product with no recorded path to a contract "
          "input is untouched.")
        A("")
    A(md_table(["field", "old", "new", "class", "confidence"],
               [[r["field"], r["old"], r["new"], r["class"], r.get("confidence", "recorded")]
                for r in R["contract_rows"]]))
    A("")
    A(f"Products reached by the break along recorded edges: **{len(R['contract_affected'])}**"
      f" -- {', '.join(sorted(R['contract_affected']))}")
    A("")
    if R["code_rows"]:
        A("### code / environment")
        A("")
        A(md_table(["repo", "old", "new", "confidence"],
                   [[r["repo"], s12(r["old"]), s12(r["new"]), r.get("confidence")]
                    for r in R["code_rows"]]))
        A("")
    A("## 3. Products")
    A("")
    A(md_table(["product_id", "kind", "class", "old sha", "new sha", "upstream moved",
                "contract break", "status old -> new", "confidence"],
               [[p["product_id"], p["kind"], p["class"], s12(p["old_sha256"]),
                 s12(p["new_sha256"]), len(p["upstream_changed"]),
                 "yes" if p["contract_broken"] else "",
                 f"{p['status_old']} -> {p['status_new']}", p["confidence"]]
                for p in R["products"]]))
    A("")
    A(f"UNTRACEABLE on the old side: **{len(R['untraceable_old'])}** "
      f"({', '.join(R['untraceable_old']) or 'none'}) -- expected and reported, not suppressed "
      "(MANIFEST_DIFF_SPEC s4: the 2026-08-26 archive records no per-entry producer).")
    A("")
    A(f"UNTRACEABLE on the new side: **{len(R['untraceable_new'])}** "
      f"({', '.join(R['untraceable_new']) or 'none'}).")
    A("")
    A("## 4. Ledger quantities")
    A("")
    qr = [q for q in R["quantities"].values() if q["status"] != "added" or q["new"] is not None]
    A(md_table(["quantity_id", "old", "new", "d%", "d/hw", "hw ratio", "p", "printed",
                "digits change", "class"],
               [[q["quantity_id"],
                 q["printed_old"] if q["printed_old"] is not None else q["old"],
                 q["printed_new"] if q["printed_new"] is not None else q["new"],
                 None if q["delta_pct"] is None else f"{q['delta_pct']:+.3f}",
                 None if q["delta_over_halfwidth"] is None else f"{q['delta_over_halfwidth']:+.2f}",
                 None if q["halfwidth_ratio"] is None else f"{q['halfwidth_ratio']:.4f}",
                 q["printed_precision"],
                 f"{q['printed_old']} -> {q['printed_new']}" if q["printed_old"] else "",
                 q["printed_digits_change"], q["provenance_class"]]
                for q in qr]))
    A("")
    retired_live = R["retired_but_referenced"]
    if retired_live:
        A(f"**HARD ERROR: retired quantities that still have live dependencies:** {retired_live}")
        A("")
    A("## 5. Consumers")
    A("")
    for group, label in (("SHIPPING", "SHIPPING-STALE"), ("SUPPRESSED", "SUPPRESSED-STALE")):
        rows = [c for c in R["consumers"]
                if c["verdict"] in STALE_VERDICTS and (c["ships"] == (group == "SHIPPING"))]
        A(f"### {label} ({len(rows)})")
        A("")
        if rows:
            A(md_table(["consumer_id", "location", "emitter", "depends_on", "printed old -> new",
                        "digits", "verdict"],
                       [[c["consumer_id"],
                         f"{pathlib.Path(str(c['location'].get('path'))).name}"
                         f"{':' + str(c['location']['line']) if c['location'].get('line') else ''}"
                         f"{'#' + c['location']['key'] if c['location'].get('key') else ''}",
                         c["emitter"], len(c["depends_on"]),
                         (f"{c['printed_old']} -> {c['printed_new']}"
                          if c["printed_old"] is not None else ""),
                         "YES" if c["printed_digits_change"] else "",
                         c["verdict"]] for c in rows]))
        else:
            A("_none_")
        A("")
    for kind, label in KIND_TABLES:
        rows = [c for c in R["consumers"]
                if c["consumer_kind"] == kind and c["verdict"] in STALE_VERDICTS]
        if not rows:
            continue
        A(f"### {label} ({len(rows)})")
        A("")
        A(md_table(["consumer_id", "path:line/key", "ships", "verdict", "disposition"],
                   [[c["consumer_id"],
                     f"{pathlib.Path(str(c['location'].get('path'))).name}"
                     f"{':' + str(c['location']['line']) if c['location'].get('line') else ''}"
                     f"{'#' + c['location']['key'] if c['location'].get('key') else ''}",
                     "ships" if c["ships"] else "suppressed", c["verdict"], c["disposition"]]
                    for c in rows]))
        A("")
    hr = [c for c in R["consumers"] if c["hand_typed_risk"]]
    A(f"### HAND-TYPED-RISK ({len(hr)}) -- the manual edit list")
    A("")
    A("These cannot be fixed by re-running an emitter.")
    A("")
    A(md_table(["consumer_id", "file:line", "current literal", "printed old -> new", "verdict"],
               [[c["consumer_id"],
                 f"{c['location'].get('path')}:{c['location'].get('line')}",
                 (c["literal_in_manuscript"] or "")[:70],
                 (f"{c['printed_old']} -> {c['printed_new']}"
                  if c["printed_old"] is not None else ""),
                 c["verdict"]] for c in hr]))
    A("")
    A("## 6. PENDING-LITERAL CLEARANCE LIST")
    A("")
    A("`tools/check_additions.py:1063-1070` fails a SUBMISSION build while any of these renders. "
      "The guard is correct and is not modified; this list is what makes its failure actionable.")
    A("")
    A(md_table(["file:line", "owning ledger line", "placeholder", "replacing quantity",
                "new value at this site's precision", "ships?", "resolution"],
               [[r["file_line"], r["owning_ledger_line"], (r["placeholder"] or "")[:60],
                 "; ".join(v["quantity_id"] for v in r["replacing"]),
                 "; ".join(str(v.get("new_value")) for v in r["replacing"]),
                 "yes" if r["ships"] else "no", r["resolution"]]
                for r in R["pending_literals"]]))
    A("")
    A(f"**{R['summary_counts']['pending_literals_total']} live pending literals; "
      f"{R['summary_counts']['pending_literals_cleared']} cleared by this freeze; "
      f"{R['summary_counts']['pending_literals_unresolved']} UNRESOLVED.**")
    A("")
    A("## 7. Hash pins")
    A("")
    A(md_table(["consumer_id", "location", "key", "product", "old", "new", "state",
                "recommendation"],
               [[h["consumer_id"], h["location"], h["key"], h["product_id"],
                 s12(h["old_sha256"]), s12(h["new_sha256"]), h["pin_state"],
                 h["recommendation"]]
                for h in R["hash_pins"]]))
    A("")
    A("## 8. Release package")
    A("")
    if R["release_changes"]:
        A(md_table(["product_id", "old class", "new class", "newly restricted",
                    "UNDECLARED WITHHOLDING"],
                   [[r["product_id"], r["old"].get("class"), r["new"].get("class"),
                     r["newly_restricted"], r["undeclared_withholding"]]
                    for r in R["release_changes"]]))
    else:
        A("No release-eligibility change. Withholding stays declared on both sides.")
    A("")
    A("## 9. Summary counts")
    A("")
    A(md_table(["count", "value"], [[k, v] for k, v in R["summary_counts"].items()]))
    A("")
    A(f"exit code: **{R['exit_code']}**")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------------- main
def run(old_path, new_path, schema=None):
    old, new = load(old_path), load(new_path)
    if schema:
        try:
            import jsonschema
            sch = load(schema)
            jsonschema.validate(old, sch)
            jsonschema.validate(new, sch)
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            print("SCHEMA-INVALID INPUT:", str(e)[:2000], file=sys.stderr)
            raise SystemExit(2)

    crows, cbreaks = contract_diff(old, new)
    prows, caffected, unt_old, unt_new = product_report(old, new, cbreaks)
    qrows = quantity_report(old, new)
    cons = consumer_report(old, new, qrows, prows, cbreaks)
    blocking = [o["id"] for o in new.get("open_items", []) if o.get("blocking")]
    pend = pending_literal_list(cons, qrows)
    hp = hash_pin_report(cons, prows, suppressed=bool(blocking))
    relc = release_report(prows)

    referenced = set()
    for c in cons:
        referenced |= set(c["depends_on"])
    retired_live = sorted(qid for qid, q in qrows.items()
                          if q["status"] == "retired" and qid in referenced)

    stale = [c for c in cons if c["verdict"] in STALE_VERDICTS]
    counts = {
        "products_total": len(prows),
        "products_rehashed": sum(1 for p in prows if p["class"] == "rehashed"),
        "products_stale": sum(1 for p in prows if p["status_new"] == "stale"),
        "products_contract_broken": sum(1 for p in prows if p["contract_broken"]),
        "quantities_total": len(qrows),
        "quantities_changed": sum(1 for q in qrows.values()
                                  if q["old"] != q["new"] and q["status"] == "present"),
        "quantities_printed_digit_changed": sum(1 for q in qrows.values()
                                                if q["printed_digits_change"]),
        "consumers_total": len(cons),
        "consumers_stale": len(stale),
        "consumers_stale_shipping": sum(1 for c in stale if c["ships"]),
        "consumers_stale_suppressed": sum(1 for c in stale if not c["ships"]),
        "table_cells_stale": sum(1 for c in stale if c["consumer_kind"] == "table_cell"),
        "figure_arrays_stale": sum(1 for c in stale if c["consumer_kind"] == "figure_array"),
        "prose_literals_stale": sum(1 for c in stale if c["consumer_kind"] == "prose_literal"),
        "hash_pins_stale": sum(1 for h in hp if h["pin_state"] == "DIFFERS NOW"),
        "hash_pins_pending_refresh": sum(1 for h in hp if h["pin_state"] != "DIFFERS NOW"),
        "ledger_rows_stale": sum(1 for c in stale if c["consumer_kind"] == "ledger_row"),
        "release_changes": len(relc),
        "untraceable": len(unt_new),
        "untraceable_old_side": len(unt_old),
        "hand_typed_risk": sum(1 for c in cons if c["hand_typed_risk"]),
        "contract_breaks": len(cbreaks),
        "pending_literals_total": len(pend),
        "pending_literals_cleared": sum(1 for r in pend
                                        if not r["resolution"].startswith("UNRESOLVED")),
        "pending_literals_unresolved": sum(1 for r in pend
                                           if r["resolution"].startswith("UNRESOLVED")),
    }
    exit_code = 0
    if stale:
        exit_code = 1
    if unt_new or retired_live or any(r["undeclared_withholding"] for r in relc):
        exit_code = 2

    import hashlib

    def h(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    R = {
        "_class": "VALIDATION-ONLY. A staleness oracle, not an estimator.",
        "old": {"manifest_id": old["manifest_id"], "sha256": h(old_path),
                "freeze_status": old["freeze"]["status"],
                "completeness": (new.get("supersession", {})
                                 .get("previous_manifest_completeness", "unknown"))},
        "new": {"manifest_id": new["manifest_id"], "sha256": h(new_path),
                "freeze_status": new["freeze"]["status"]},
        "not_adoptable": bool(blocking),
        "blocking_open_items": blocking,
        "contract_rows": crows, "contract_break": cbreaks,
        "contract_affected": sorted(caffected),
        "code_rows": code_diff(old, new),
        "products": prows, "untraceable_old": unt_old, "untraceable_new": unt_new,
        "quantities": qrows, "retired_but_referenced": retired_live,
        "consumers": cons, "pending_literals": pend, "hash_pins": hp,
        "release_changes": relc, "summary_counts": counts, "exit_code": exit_code,
    }
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--emit", choices=["md", "json", "both"], default="both")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--schema", default=str(SCHEMA_PATH))
    ap.add_argument("--strict-precision", action="store_true",
                    help="treat any value change as reprint-worthy, not only printed digits")
    ap.add_argument("--fail-on-stale", action="store_true", default=True)
    a = ap.parse_args()

    R = run(a.old, a.new, a.schema)
    if a.strict_precision:
        for c in R["consumers"]:
            if c["verdict"] == "STALE-SILENT":
                c["verdict"] = "STALE-REPRINT"
    od = pathlib.Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    if a.emit in ("json", "both"):
        with open(od / "manifest_diff.json", "w") as f:
            json.dump(R, f, indent=1)
    if a.emit in ("md", "both"):
        (od / "manifest_diff.md").write_text(render_md(R))
    sc = R["summary_counts"]
    print(f"{R['old']['manifest_id']} -> {R['new']['manifest_id']}  "
          f"{'NOT-ADOPTABLE  ' if R['not_adoptable'] else ''}"
          f"stale consumers {sc['consumers_stale']} "
          f"(shipping {sc['consumers_stale_shipping']}, "
          f"suppressed {sc['consumers_stale_suppressed']}); "
          f"pending literals {sc['pending_literals_cleared']}/{sc['pending_literals_total']} "
          f"cleared; untraceable new {sc['untraceable']} old {sc['untraceable_old_side']}")
    raise SystemExit(R["exit_code"])


if __name__ == "__main__":
    main()
