% MATLAB driver for Layer 4 parity check.
%
% Loads inputs from a .mat file, calls the DR16Q-public reference
% spectrum_loss.m, writes outputs to another .mat file.
%
% Usage (from MATLAB / Octave):
%     matlab_parity_check('input.mat', 'matlab_outputs.mat')
%
% Usage (from shell):
%     matlab -batch "matlab_parity_check('input.mat', 'matlab_outputs.mat'); exit"

function matlab_parity_check(input_path, output_path)
    % The Layer 4 reference is the DR16Q-public spectrum_loss.m at
    %   /home/mfho/gp_dla_detection_dr16q_public/spectrum_loss.m
    % which already includes the Lyα indicator (older multi_dlas/ does not).
    addpath('/home/mfho/gp_dla_detection_dr16q_public');

    in = load(input_path);

    % Squeeze scalars: MATLAB stores them as 1x1 doubles already, but be
    % defensive against any v7.3 hdf5 quirks.
    [nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta] = ...
        spectrum_loss(in.y, in.lya_1pz, in.noise_variance, in.M, ...
            in.omega2, in.c_0, in.tau_0, in.beta, ...
            double(in.num_forest_lines), in.transition_wavelengths, ...
            in.oscillator_strengths, in.zqso_1pz);

    save(output_path, 'nlog_p', 'dM', 'dlog_omega', 'dlog_c_0', ...
                      'dlog_tau_0', 'dlog_beta', '-v7');
end
