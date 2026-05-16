"""Kernel-conditioning probe: condition number and singular spectra for
the c0prior and _m trained kernels.

K = c_0² · A_lyα·(MMᵀ + diag(ω²))·A_lyα   (approximately; absorption
attenuates both M and ω during inference, but the bare K-matrix is what
matters at high z_qso when A_lyα ≈ 1 on the red side).

We report:
- cond(MMᵀ + diag(ω²)) at the trained scale (no c_0).
- Singular values of M.
- Effective rank (number of singular values within 1e-3 of the largest).
- log(diag(ω²)) summary statistics.
- The scale factor c_0·A_lyα(z=2.5) for a representative QSO redshift.
"""
import h5py
import numpy as np

MODELS = {
    'c0prior': '/home/mfho/desi_gpy_dla_detection/docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/phase2_result.h5',
    'm_baseline': '/home/mfho/desi_gpy_dla_detection/docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5',
}

def load(p):
    with h5py.File(p, 'r') as f:
        return dict(
            M=f['M'][()], mu=f['mu'][()], log_omega=f['log_omega'][()],
            log_c_0=float(f['log_c_0'][()]),
            log_tau_0=float(f['log_tau_0'][()]),
            log_beta=float(f['log_beta'][()]),
            rest=f['rest_wavelengths'][()],
            norm_min=float(f['normalization_min_lambda'][()]) if 'normalization_min_lambda' in f else float('nan'),
            norm_max=float(f['normalization_max_lambda'][()]) if 'normalization_max_lambda' in f else float('nan'),
        )

for name, p in MODELS.items():
    d = load(p)
    M = d['M']
    omega2 = np.exp(2 * d['log_omega'])
    c_0 = np.exp(d['log_c_0'])
    tau_0 = np.exp(d['log_tau_0'])
    beta = np.exp(d['log_beta'])
    n, k = M.shape
    print(f'\n=== {name} ===')
    print(f'  n_pix={n}, k={k}')
    print(f'  norm band:   [{d["norm_min"]:.1f}, {d["norm_max"]:.1f}] Å rest')
    print(f'  c_0       = {c_0:.6f}  (log_c_0={d["log_c_0"]:.4f})')
    print(f'  τ_0       = {tau_0:.6e} (log_tau_0={d["log_tau_0"]:.4f})')
    print(f'  β         = {beta:.4f}')

    # Singular values of M
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    print(f'  M singular values: max={s[0]:.4g}, min={s[-1]:.4g}, ratio={s[0]/s[-1]:.4g}')
    print(f'  M sv top 5: {s[:5]}')
    print(f'  M sv bottom 5: {s[-5:]}')
    # Frobenius scale
    print(f'  |M|_F^2 = {(M**2).sum():.4g}')

    # diag stats
    print(f'  ω² stats: min={omega2.min():.4g}  median={np.median(omega2):.4g}  max={omega2.max():.4g}')
    print(f'  log_omega stats: min={d["log_omega"].min():.4f}  median={np.median(d["log_omega"]):.4f}  max={d["log_omega"].max():.4f}')

    # K = MMᵀ + diag(ω²)  — full n×n matrix, evaluate cond via low-rank trick
    # Use Woodbury: eigenvalues = diag(ω²) eigenvalues + low-rank contribution
    # Cheap: compute MᵀM eigs and combine with ω² eigs
    # Cleaner: compute SVD of A = [M | diag(ω)·I_n^{1/2}]... but full n×n is 5662×5662, ok.
    print('  computing full K = MMᵀ + diag(ω²) eigendecomp (n=5662) ...', end='', flush=True)
    K = M @ M.T
    K[np.arange(n), np.arange(n)] += omega2
    # symmetric eigs
    eigs = np.linalg.eigvalsh(K)
    eigs.sort()
    print(' done.')
    print(f'  K eigenvalues: min={eigs[0]:.4g}  max={eigs[-1]:.4g}')
    print(f'  cond(K)        = {eigs[-1]/eigs[0]:.4g}')
    # Effective ranks
    rel = eigs / eigs[-1]
    print(f'  K eigs in [1e-3 .. max]: {(rel >= 1e-3).sum()}')
    print(f'  K eigs in [1e-6 .. max]: {(rel >= 1e-6).sum()}')
    print(f'  K eigs in [1e-9 .. max]: {(rel >= 1e-9).sum()}')

    # Reconstructed continuum scale at z_qso=2.5
    # c_0 enters via reconstructed flux = c_0 · A_lyα(z) · (μ + Mη); A_lyα ≈ 1 on red side.
    print(f'  c_0 × μ_mean = {c_0 * d["mu"].mean():.4f}  (data should be ≈ 1.0 after normalization)')
    # The "effective" K is multiplied by c_0² in the GP likelihood
    print(f'  c_0² × eigs[max] = {(c_0**2) * eigs[-1]:.4g}')
    print(f'  c_0² × eigs[min] = {(c_0**2) * eigs[0]:.4g}')
