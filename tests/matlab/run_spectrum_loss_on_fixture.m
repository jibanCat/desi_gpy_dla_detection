function run_spectrum_loss_on_fixture()
% Step A.2 MATLAB driver. Loads the frozen 2lpt fixture and calls the
% legacy MATLAB spectrum_loss.m on each TID, saving outputs as
% <TID>_matlab.mat for the Python comparison tests.
%
% Run from anywhere via:
%   matlab -batch "addpath(fullfile(pwd,'tests','matlab')); run_spectrum_loss_on_fixture"
%
% Or, after `cd` into tests/matlab/:
%   matlab -batch "run_spectrum_loss_on_fixture"

  % Resolve fixture dir relative to this .m file
  this_file = mfilename('fullpath');
  this_dir  = fileparts(this_file);
  fixture_dir = fullfile(this_dir, '..', 'fixtures', '2lpt_frozen');
  fprintf('fixture_dir: %s\n', fixture_dir);

  % Add the legacy MATLAB code path so spectrum_loss.m resolves
  matlab_repo = '/home/mfho/MATLAB/gp_dla_detection_dr16q_public';
  if ~exist(fullfile(matlab_repo, 'spectrum_loss.m'), 'file')
    error('spectrum_loss.m not found at %s', matlab_repo);
  end
  addpath(matlab_repo);

  % Load population init
  init = load(fullfile(fixture_dir, 'init_params.mat'));

  % Cast everything to double + column convention as appropriate
  M_full         = double(init.M);                                % (n_pix, k)
  log_omega_full = double(init.log_omega(:));                     % (n_pix, 1)
  c_0            = double(init.c_0(1));                            % scalar
  tau_0          = double(init.tau_0(1));                          % scalar
  beta           = double(init.beta(1));                           % scalar
  num_forest_lines = double(init.num_forest_lines(1));             % scalar
  TW             = double(init.all_transition_wavelengths(:));     % (31, 1)
  OS             = double(init.all_oscillator_strengths(:));       % (31, 1)
  omega2_full    = exp(2 * log_omega_full);                        % (n_pix, 1)

  fprintf('init: n_pix=%d  k=%d  c_0=%.4f  tau_0=%.5f  beta=%.4f\n', ...
    size(M_full, 1), size(M_full, 2), c_0, tau_0, beta);

  tids = [270143607, 250027833, 40000430, 220250636, 180021938, 120046865];

  for ti = 1:length(tids)
    tid = tids(ti);
    spec_path = fullfile(fixture_dir, sprintf('%d.mat', tid));
    spec = load(spec_path);

    flux       = double(spec.flux(:));
    nv         = double(spec.noise_variance(:));
    lya_1pz    = double(spec.lya_1pz(:));
    valid_mask = logical(spec.valid_mask(:));
    zqso_1pz   = double(spec.zqso_1pz(1));

    y_m       = flux(valid_mask);
    lya_1pz_m = lya_1pz(valid_mask);
    nv_m      = nv(valid_mask);
    M_m       = M_full(valid_mask, :);
    omega2_m  = omega2_full(valid_mask);

    [nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta] = ...
        spectrum_loss(y_m, lya_1pz_m, nv_m, M_m, omega2_m, ...
            c_0, tau_0, beta, num_forest_lines, TW, OS, zqso_1pz);

    n_valid = length(y_m);
    fprintf('  TID %10d  z_qso=%.3f  n_valid=%4d  nlog_p=%.6f  dlog_beta=%.6e\n', ...
      tid, zqso_1pz - 1, n_valid, nlog_p, dlog_beta);

    out = struct();
    out.target_id  = tid;
    out.n_valid    = n_valid;
    out.zqso_1pz   = zqso_1pz;
    out.nlog_p     = nlog_p;
    out.dM         = dM;
    out.dlog_omega = dlog_omega;
    out.dlog_c_0   = dlog_c_0;
    out.dlog_tau_0 = dlog_tau_0;
    out.dlog_beta  = dlog_beta;
    out_path = fullfile(fixture_dir, sprintf('%d_matlab.mat', tid));
    save(out_path, '-struct', 'out');
  end

  fprintf('done.\n');
end
