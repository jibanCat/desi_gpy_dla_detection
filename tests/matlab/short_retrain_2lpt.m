function short_retrain_2lpt()
% Step A.3 MATLAB lane. Runs minFunc/L-BFGS for ~50 iterations on the
% same 1300-spectrum 2lpt training set the Python A.3 runner uses,
% via the legacy MATLAB objective.m (which has correct zqso_1pz +
% BOSS DR12Q priors at lines 47, 66-77).
%
% Saves <fixture>/short_retrain/matlab.mat with final M, log_omega,
% log_c_0, log_tau_0, log_beta + loss trajectory captured via a global.
%
% Run via:
%   /sw/pkgs/arc/matlab/R2024b/bin/matlab -batch \
%     "addpath(fullfile(pwd,'tests','matlab')); short_retrain_2lpt"

  this_file = mfilename('fullpath');
  this_dir  = fileparts(this_file);
  fixture_dir = fullfile(this_dir, '..', 'fixtures', '2lpt_frozen');
  out_dir = fullfile(fixture_dir, 'short_retrain');
  if ~exist(out_dir, 'dir'); mkdir(out_dir); end

  % Setup: legacy MATLAB code + minFunc
  matlab_repo = '/home/mfho/MATLAB/gp_dla_detection_dr16q_public';
  addpath(matlab_repo);
  addpath(genpath('/home/mfho/MATLAB/minFunc_2012'));
  if isempty(which('minFunc'))
    error('minFunc not on path');
  end
  if isempty(which('objective'))
    error('legacy objective.m not on path');
  end

  init = load(fullfile(fixture_dir, 'init_params.mat'));
  train = load(fullfile(fixture_dir, 'training_set.mat'));

  M0 = double(init.M);
  log_omega0 = double(init.log_omega(:));
  log_c_0_0 = log(double(init.c_0(1)));
  log_tau_0_0 = log(double(init.tau_0(1)));
  log_beta_0  = log(double(init.beta(1)));

  centered_fluxes = double(train.centered_fluxes);
  noise_variances = double(train.noise_variances);
  z_qsos          = double(train.z_qsos(:));
  rest_wavelengths = double(init.rest_wavelengths(:));

  % Replace stored NaN-fluxes with NaN (already there in fixture); MATLAB
  % objective.m uses ~isnan(...) for masking
  N = size(centered_fluxes, 1);
  n_pix = size(centered_fluxes, 2);
  k = size(M0, 2);

  % lya_1pzs (N × n_pix) — same formula Python uses
  lya_rest = 1215.67;
  lya_1pzs = (1 + z_qsos) * (rest_wavelengths' / lya_rest);

  num_forest_lines = double(init.num_forest_lines(1));
  TW = double(init.all_transition_wavelengths(:));
  OS = double(init.all_oscillator_strengths(:));

  % Build initial x packing required by legacy objective.m:
  %   x = [vec(M); log_omega; log_c_0; log_tau_0; log_beta]
  initial_x = [M0(:); log_omega0; log_c_0_0; log_tau_0_0; log_beta_0];
  expected_len = n_pix * k + n_pix + 3;
  if length(initial_x) ~= expected_len
    error('initial_x length %d != expected %d', length(initial_x), expected_len);
  end
  fprintf('initial_x: %d entries (n_pix=%d, k=%d)\n', length(initial_x), n_pix, k);

  % Track loss trajectory via a persistent global (minFunc doesn't expose
  % a per-iter callback hook directly).
  global SHORT_RETRAIN_HIST;
  SHORT_RETRAIN_HIST = struct( ...
    'loss',     [], ...
    'log_c_0',  [], ...
    'log_tau_0',[], ...
    'log_beta', []);

  fwrap = @(x) wrapped_obj(x, centered_fluxes, lya_1pzs, noise_variances, ...
                           num_forest_lines, TW, OS, z_qsos);

  options = struct( ...
    'MaxIter',     50, ...
    'MaxFunEvals', 100, ...
    'Display',     'iter', ...
    'optTol',      1e-6, ...
    'progTol',     1e-9);

  fprintf('Starting minFunc...\n');
  t0 = tic;
  [x_final, fval_final, exit_flag, output] = minFunc(fwrap, initial_x, options);
  wall = toc(t0);
  fprintf('minFunc done: exit_flag=%d, iter=%d, funcCount=%d, fval=%.4f, wall=%.1fs\n', ...
    exit_flag, output.iterations, output.funcCount, fval_final, wall);

  % Unpack final params
  M_final          = reshape(x_final(1:n_pix*k), [n_pix, k]);
  log_omega_final  = x_final(n_pix*k+1 : n_pix*(k+1));
  log_c_0_final    = x_final(end-2);
  log_tau_0_final  = x_final(end-1);
  log_beta_final   = x_final(end);

  % Save
  out = struct();
  out.lane             = 'matlab';
  out.M_final          = M_final;
  out.mu               = double(init.mu);
  out.log_omega_final  = log_omega_final;
  out.log_c_0_final    = log_c_0_final;
  out.log_tau_0_final  = log_tau_0_final;
  out.log_beta_final   = log_beta_final;
  out.c_0_final        = exp(log_c_0_final);
  out.tau_0_final      = exp(log_tau_0_final);
  out.beta_final       = exp(log_beta_final);
  out.loss_history     = SHORT_RETRAIN_HIST.loss(:);
  out.log_c_0_history  = SHORT_RETRAIN_HIST.log_c_0(:);
  out.log_tau_0_history = SHORT_RETRAIN_HIST.log_tau_0(:);
  out.log_beta_history  = SHORT_RETRAIN_HIST.log_beta(:);
  out.rest_wavelengths = rest_wavelengths;
  out.exit_flag        = exit_flag;
  out.iterations       = output.iterations;
  out.funcCount        = output.funcCount;
  out.wall_s           = wall;

  save_path = fullfile(out_dir, 'matlab.mat');
  save(save_path, '-struct', 'out');
  fprintf('saved: %s\n', save_path);
end


function [f, g] = wrapped_obj(x, cf, lp, nv, nfl, tw, os, zq)
% Wrapper around legacy objective.m that records (loss, log_c_0,
% log_tau_0, log_beta) per call into a global for trajectory plots.
  [f, g] = objective(x, cf, lp, nv, nfl, tw, os, zq);
  global SHORT_RETRAIN_HIST;
  SHORT_RETRAIN_HIST.loss(end+1)      = f;
  SHORT_RETRAIN_HIST.log_c_0(end+1)   = x(end-2);
  SHORT_RETRAIN_HIST.log_tau_0(end+1) = x(end-1);
  SHORT_RETRAIN_HIST.log_beta(end+1)  = x(end);
end
