"""holder.py — LLSHolder: the DLAHolder finder wired to the break-aware LLS GP.

Runs the production finder (`run_bayes_select.DLAHolder` → `process_single_spectrum`) unchanged
EXCEPT that the sub-DLA/LLS channel uses `SubDLAGPMATLymanBreak` (GP+Voigt+912 Å break) instead
of the line-only `SubDLAGPMAT`. This is the single hook that turns the standard sub-DLA search
into the break-aware LLS search.

Usage in an LLS run (mock or real): build the DLAHolder config as usual but
  - set BOTH `params.min_lambda` and `params_subdla.min_lambda` to the extended LLS window
    (e.g. 850.9) so null/DLA/LLS evidences are computed on the SAME pixels AND the 912 Å break
    is in-window (use `gpy_dla_detection.lls.extend_window_to_drop` on each);
  - point `learned_file` at the LLS model (mock-matched relearn, or `model_epoch_920.h5`);
then construct `LLSHolder(...)` instead of `DLAHolder(...)`. A line-only control run is the same
config with plain `DLAHolder`.

FROZEN-CODE DISCIPLINE: `run_bayes_select.py` is NOT modified. `process_qso` below is a faithful
copy of `DLAHolder.process_qso` with the one documented swap — keep in sync with it.
"""
from __future__ import annotations
import time

from run_bayes_select import (
    DLAHolder,
    NullGPMAT,
    DLAGPMAT,
    BayesModelSelect,
    process_single_spectrum,
    log,
)
from .gp import SubDLAGPMATLymanBreak, DLAGPMATLymanBreak


class LLSHolder(DLAHolder):
    """DLAHolder whose absorber model is break-aware. Under the production single-absorber
    config the absorber IS the DLA model, so we swap dla_gp -> DLAGPMATLymanBreak; in the
    (retired) 3-way mode the subdla_gp -> SubDLAGPMATLymanBreak swap also applies."""

    def process_qso(
        self,
        idx,
        target_id,
        wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso,
    ):
        # --- faithful copy of run_bayes_select.DLAHolder.process_qso; ONLY the subdla_gp
        #     class is swapped SubDLAGPMAT -> SubDLAGPMATLymanBreak (marked below). ---
        tic = time.time()

        rest_wavelengths = self.params.emitted_wavelengths(wavelengths, z_qso)

        prev_tau_0_eff = self.prev_tau_0
        if self.enable_tau_eb:
            from gpy_dla_detection.tau_eb import fit_tau_eb
            prev_tau_0_eff, tau_eb_info = fit_tau_eb(
                params=self.params,
                prior=self.prior,
                learned_file=self.learned_file,
                rest_wavelengths=rest_wavelengths,
                flux=flux,
                noise_variance=noise_variance,
                pixel_mask=pixel_mask,
                z_qso=z_qso,
                prev_tau_0_seed=self.prev_tau_0,
                prev_beta=self.prev_beta,
                tau_factors=self.tau_eb_factors,
                apply_hcd_mask=self.tau_eb_apply_hcd_mask,
                mask_threshold_sigma=self.tau_eb_mask_threshold_sigma,
                objective=self.tau_eb_objective,
                dla_samples=self.dla_samples,
            )
            log.info(
                f" ...     τ-EB[{self.tau_eb_objective}, "
                f"hcd_mask={self.tau_eb_apply_hcd_mask}]: "
                f"factor_best={tau_eb_info['tau_factor_best']:.2f}  "
                f"τ_0={prev_tau_0_eff:.5f}  "
                f"n_hcd={tau_eb_info['n_hcd']}"
            )

        null_gp = NullGPMAT(
            self.params,
            self.prior,
            learned_file=self.learned_file,
            prev_tau_0=prev_tau_0_eff,
            prev_beta=self.prev_beta,
        )
        # <<< THE SWAP (single-absorber): break-aware DLA model instead of line-only DLAGPMAT >>>
        dla_gp = DLAGPMATLymanBreak(
            self.params,
            self.prior,
            self.dla_samples,
            min_z_separation=self.min_z_separation,
            learned_file=self.learned_file,
            broadening=self.broadening,
            prev_tau_0=prev_tau_0_eff,
            prev_beta=self.prev_beta,
            early_stop_mode=self.early_stop_mode,
            pair_prior_mode=self.pair_prior_mode,
            pair_prior=self.pair_prior,
        )
        if self.single_absorber_model:
            subdla_gp = None
            bayes = BayesModelSelect([0, self.max_dlas], 1)
        else:
            # <<< THE ONE SWAP: break-aware LLS GP instead of line-only SubDLAGPMAT >>>
            subdla_gp = SubDLAGPMATLymanBreak(
                self.params_subdla,
                self.prior,
                self.subdla_samples,
                min_z_separation=self.min_z_separation,
                learned_file=self.learned_file,
                broadening=self.broadening,
                prev_tau_0=prev_tau_0_eff,
                prev_beta=self.prev_beta,
            )
            bayes = BayesModelSelect([0, 1, self.max_dlas], 2)

        log.info(
            f"Processing spectrum {idx + 1}/{self.num_spectra} (ID: {target_id}) zQSO: {z_qso:.2f}"
        )
        process_single_spectrum(
            idx,
            target_id,
            z_qso,
            wavelengths,
            rest_wavelengths,
            flux,
            noise_variance,
            pixel_mask,
            self.params,
            self.prior,
            self.dla_samples,
            self.subdla_samples,
            bayes,
            self.results,
            self.max_dlas,
            self.broadening,
            null_gp,
            dla_gp,
            subdla_gp,
            self.min_z_separation,
            self.plot_figures,
            self.max_workers,
            self.batch_size,
            self.figure_dir,
            filter_low_likelihood=self.filter_low_likelihood,
            filter_n_initial_floor=self.filter_n_initial_floor,
            filter_empty_mask_fallthrough=self.filter_empty_mask_fallthrough,
            single_absorber_model=self.single_absorber_model,
        )
        if self.single_absorber_model:
            del null_gp, dla_gp
        else:
            del null_gp, dla_gp, subdla_gp

        toc = time.time()
        log.info(
            f"Processed spectrum {idx + 1}/{self.num_spectra} (ID: {target_id}), "
            f"time spent: {(toc - tic) // 60:.0f}m {(toc - tic) % 60:.0f}s"
        )
