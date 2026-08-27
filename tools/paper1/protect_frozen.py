#!/usr/bin/env python
"""Make the frozen Paper-1 artifacts read-only (pre-push hardening 2026-08-26, PI item 4).

Applies `chmod a-w` to every file registered in docs/PAPER1_FROZEN_MANIFEST.json (optionally
under --prefix, e.g. the Turbo archive) and to the frozen directories that hold them, so that
a normal collaborator run cannot overwrite a frozen artifact by accident. `--check` only
reports what is still writable. Files inside git repositories (the notes-repo record) are
skipped: git owns their write bits.

Result of record (2026-08-26): on /scratch all 150 frozen files + the 7 frozen directories are
read-only (a write into a frozen directory is refused). The 18 manifest inputs that live on
/nfs/turbo (DESI catalogues, QSO catalogue, hz dlacats) and the Turbo ARCHIVE copy cannot be
protected this way: the Turbo share applies its own ACL and ignores POSIX mode bits (files stay
rwxrwx--- for owner/group lsa-turbo-cavestru). There the protection is (i) the manifest sha256
verification (`frozen_manifest.py --verify [--prefix]`), which detects any change, and (ii) the
scratch/Turbo duplication. `--check` reports exactly what remains writable.

    python tools/paper1/protect_frozen.py [--prefix DST] [--check] [--manifest PATH]
"""
import argparse, json, os, stat, sys

FROZEN_DIRS = (
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real/logs",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real/ppc_20260826",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/adopted_packs_v2p2_20260821",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/adopted_packs_gfix_v1_20260821",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "docs", "PAPER1_FROZEN_MANIFEST.json"))
    ap.add_argument("--prefix", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    m = json.load(open(a.manifest)); ents = m["entries"] if isinstance(m, dict) else m
    paths = [a.prefix + e["path"] for e in ents]
    paths = [p for p in paths if "/desi_gpy_dla_notes/" not in p]
    dirs = [a.prefix + d for d in FROZEN_DIRS]
    n_fix = n_ok = n_missing = 0
    for p in paths + dirs:
        if not os.path.exists(p):
            n_missing += 1; continue
        st = os.stat(p)
        writable = bool(st.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        if not writable:
            n_ok += 1; continue
        if a.check:
            print("WRITABLE", p); n_fix += 1
        else:
            os.chmod(p, st.st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)); n_fix += 1
    print(f"protect_frozen: {'would fix' if a.check else 'fixed'} {n_fix}, already read-only {n_ok}, "
          f"missing {n_missing} (prefix={a.prefix or '/'}; {len(paths)} files + {len(dirs)} dirs)")
    sys.exit(1 if (a.check and n_fix) else 0)


if __name__ == "__main__":
    main()
