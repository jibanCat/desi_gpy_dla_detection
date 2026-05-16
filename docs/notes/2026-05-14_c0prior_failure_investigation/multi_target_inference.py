"""Run DLA inference on 10 strong 2lpt-loa-124 DLAs with c0prior vs _m models.

Mirrors the loader/holder/Parameters setup from
`examples/dla_recovery_step_c.py` exactly; only the model list, the target
list, and the output JSON change. We do NOT touch any production code.
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path

REPO = Path('/home/mfho/desi_gpy_dla_detection')
NOTES = REPO / 'docs' / 'notes'
sys.path.insert(0, str(REPO))

from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso, PRESETS
from gpy_dla_detection.set_parameters import Parameters
from run_bayes_select import DLAHolder

MODELS = [
    ('c0prior',
     str(NOTES / '2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior' / 'phase2_result.h5')),
    ('m_baseline',
     str(NOTES / '2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m' / 'phase2_result.h5')),
]

OUT_DIR = REPO / 'docs/notes/2026-05-14_c0prior_failure_investigation'
TARGETS_FILE = OUT_DIR / 'sampled_dlas.json'
ZCAT_PATH = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits'
DATA_ROOT = '/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection'

preset = PRESETS['y3']


def build_holder(model_path, k_pca=30):
    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
        dlambda=preset.dlambda, k=k_pca,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=100000, **common)
    params_subdla = Parameters(num_dla_samples=100000, **common)
    holder = DLAHolder(
        learned_file=model_path,
        catalog_name=os.path.join(DATA_ROOT, 'data/dr12q/processed/catalog.mat'),
        los_catalog=os.path.join(DATA_ROOT, 'data/dla_catalogs/dr9q_concordance/processed/los_catalog'),
        dla_catalog=os.path.join(DATA_ROOT, 'data/dla_catalogs/dr9q_concordance/processed/dla_catalog'),
        dla_samples_file=os.path.join(DATA_ROOT, 'data/dr12q/processed/dla_samples_a03_100000.mat'),
        sub_dla_samples_file=os.path.join(DATA_ROOT, 'data/dr12q/processed/subdla_samples_a03_191_200_100000.mat'),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
        max_dlas=4, broadening=True,
        plot_figures=False, max_workers=8, batch_size=12500,
        figure_dir='/tmp',
        single_absorber_model=False,
        filter_low_likelihood=True,
    )
    return holder


def run_one(holder, target):
    wave, flux, nv, mask = load_one_desi_spectrum(target['spec'], target['tid'])
    z_qso = float(target['z_qso'])
    holder.initialize_results(1)
    t0 = time.time()
    try:
        holder.process_qso(idx=0, target_id=str(target['tid']),
                           wavelengths=wave, flux=flux,
                           noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
        dt = time.time() - t0
    except Exception as ex:
        return dict(tid=target['tid'], status='process_qso_failed',
                    error=repr(ex), traceback=traceback.format_exc(),
                    elapsed_s=time.time() - t0)
    res = holder.results
    return dict(
        tid=target['tid'], status='ok',
        truth_nhi=target['nhi'], truth_z_dla=target['z_dla'],
        z_qso=z_qso, snr_truth=target['snr'],
        p_dla=float(res['p_dlas'][0]),
        map_z_dla=float(res['MAP_z_dlas'][0, 0]),
        map_log_nhi=float(res['MAP_log_nhis'][0, 0]),
        model_posteriors=[float(x) for x in res['model_posteriors'][0]],
        elapsed_s=dt,
    )


def main():
    targets = json.load(open(TARGETS_FILE))
    results = {name: [] for name, _ in MODELS}
    for name, mp in MODELS:
        print(f'\n=== {name} ({mp}) ===', flush=True)
        try:
            holder = build_holder(mp)
        except Exception as ex:
            print(f'  failed to build holder: {ex!r}')
            results[name].append(dict(status='holder_failed', error=repr(ex)))
            continue
        for t in targets:
            print(f'  TID={t["tid"]} truth_NHI={t["nhi"]:.3f} ...', end='', flush=True)
            r = run_one(holder, t)
            results[name].append(r)
            if r['status'] == 'ok':
                d = r['map_log_nhi'] - r['truth_nhi']
                print(f' p_DLA={r["p_dla"]:.4f}  ΔNHI={d:+.3f}  ({r["elapsed_s"]:.1f}s)')
            else:
                print(f' FAILED: {r.get("error", "?")[:80]}')

    out = OUT_DIR / 'multi_target_results.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')

    # Summary table
    print('\n=== SUMMARY ===')
    print(f'{"TID":>11}  {"NHI_t":>6}  {"c0p p":>8}  {"c0p ΔN":>7}  {"m p":>8}  {"m ΔN":>7}')
    for i in range(len(targets)):
        t = targets[i]
        rc = results['c0prior'][i]
        rm = results['m_baseline'][i]
        s_c = (f'{rc["p_dla"]:.4f}' if rc['status'] == 'ok' else 'ERR  ')
        d_c = (f'{rc["map_log_nhi"] - rc["truth_nhi"]:+.3f}' if rc['status'] == 'ok' else 'ERR ')
        s_m = (f'{rm["p_dla"]:.4f}' if rm['status'] == 'ok' else 'ERR  ')
        d_m = (f'{rm["map_log_nhi"] - rm["truth_nhi"]:+.3f}' if rm['status'] == 'ok' else 'ERR ')
        print(f'{t["tid"]:>11}  {t["nhi"]:>6.3f}  {s_c:>8}  {d_c:>7}  {s_m:>8}  {d_m:>7}')


if __name__ == '__main__':
    main()
