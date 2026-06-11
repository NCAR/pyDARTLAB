%% generate_reference Produces golden reference data for pyDARTLAB tests.
%
% Run this script from this directory (tests/matlab_reference) in MATLAB.
% It adds DART_LAB's matlab directory to the path, calls each private
% function on fixed inputs, and writes inputs/outputs to CSV files that the
% Python test suite (test_matlab_golden.py) compares against.
%
% The DART_LAB private functions cannot be called from outside their parent
% folder, so this script copies them into a temporary working folder first.
%
% Usage:
%   >> generate_reference            % uses ../../../DART/guide/DART_LAB/matlab
%   or set the environment variable DARTLAB_MATLAB to the matlab directory.

function generate_reference()

dartlab = getenv('DARTLAB_MATLAB');
if isempty(dartlab)
    here = fileparts(mfilename('fullpath'));
    dartlab = fullfile(here, '..', '..', '..', 'DART', 'guide', 'DART_LAB', 'matlab');
end
assert(isfolder(dartlab), 'DART_LAB matlab directory not found: %s', dartlab)

% private/ functions are only callable from their parent, so copy them out
workdir = fullfile(tempdir, 'pydartlab_reference');
if isfolder(workdir), rmdir(workdir, 's'); end
mkdir(workdir);
copyfile(fullfile(dartlab, 'private', '*.m'), workdir);
addpath(workdir);
cleanup = onCleanup(@() rmpath(workdir));

outdir = fileparts(mfilename('fullpath'));

% Fixed test ensembles (kept simple so they paste into Python exactly)
ens5  = [ 0.7,  2.3, -0.4,  1.1,  3.0];
ens10 = [-1.2,  0.3,  1.9,  0.6, -0.8,  2.4,  1.1, -0.1,  0.9,  1.6];
pos10 = [ 0.4,  1.3,  2.6,  0.9,  0.2,  3.1,  1.8,  0.5,  1.1,  2.2];
state10 = [ 2.1,  0.8,  3.0,  1.5,  0.9,  3.6,  2.2,  1.0,  1.7,  2.9];
observation = 0.5;
obs_error_var = 1.0;

%% obs_increment_eakf
[inc, ~] = obs_increment_eakf(ens10, observation, obs_error_var);
writematrix(inc(:)', fullfile(outdir, 'eakf_increments.csv'));

%% obs_increment_rhf, unbounded and bounded
[inc, ~, ~, ~, ~, ~] = obs_increment_rhf(ens10, observation, obs_error_var, false);
writematrix(inc(:)', fullfile(outdir, 'rhf_increments.csv'));

[inc, ~, ~, ~, ~, ~] = obs_increment_rhf(pos10, observation, obs_error_var, true);
writematrix(inc(:)', fullfile(outdir, 'rhf_bounded_increments.csv'));

%% product_of_gaussians
[pm, psd, w] = product_of_gaussians(1.0, 2.0, 0.5, 1.0);
writematrix([pm psd w], fullfile(outdir, 'product_of_gaussians.csv'));

%% comp_cov_factor over a range of distances
dists = 0:0.025:0.5;
cf = arrayfun(@(d) comp_cov_factor(d, 0.2), dists);
writematrix([dists(:) cf(:)], fullfile(outdir, 'comp_cov_factor.csv'));

%% get_state_increments
[obs_inc, ~] = obs_increment_eakf(ens10, observation, obs_error_var);
state_incs = get_state_increments(state10, ens10, obs_inc);
writematrix(state_incs(:)', fullfile(outdir, 'state_increments.csv'));

%% update_inflate, both flavors
% columns: x_p r_var y_o sigma_o_2 ss_base lambda_mean lambda_sd lower upper gamma sd_lower ens_size
cases = [ ...
    0.0 1.0 3.0 1.0 1.0 1.00 0.60 1.0 100.0 1.0 0.10 20; ...
    0.5 2.0 4.0 1.0 1.2 1.20 0.50 1.0 100.0 1.0 0.10 40; ...
    0.0 1.0 0.1 1.0 1.5 1.50 0.60 1.0 100.0 0.5 0.10 20; ...
    1.0 0.5 5.0 2.0 1.0 1.05 0.40 1.0 100.0 0.8 0.05 10];
for flavor = {'Gaussian', 'I-Gamma'}
    out = zeros(size(cases, 1), 2);
    for k = 1:size(cases, 1)
        c = num2cell(cases(k, :));
        [m, s] = update_inflate(c{:}, flavor{1});
        out(k, :) = [m s];
    end
    if strcmp(flavor{1}, 'Gaussian'), name = 'update_inflate_gaussian.csv';
    else, name = 'update_inflate_igamma.csv'; end
    writematrix([cases out], fullfile(outdir, name));
end

%% change_GA_IG
beta = change_GA_IG(1.2, 0.36);
writematrix(beta, fullfile(outdir, 'change_ga_ig.csv'));

%% bnrh_cdf quantiles + inv_bnrh_cdf round trip (bounded below at 0)
% NOTE: bnrh_cdf.m has del_q = 1/(ens_size + 1.8), which looks like a typo
% for 1/(ens_size + 1) (the DART Fortran value). pyDARTLAB implements the
% Fortran value, so tail_amp/tail_mean comparisons use a loose tolerance in
% the Python test; the quantiles themselves are unaffected.
[~, quantiles, ~, ~, ~, ~, ~, ~, ~, ~] = bnrh_cdf(pos10, 10, true, false, 0, -99);
writematrix(quantiles(:)', fullfile(outdir, 'bnrh_quantiles.csv'));

%% ens_quantiles with interior and boundary duplicates
dup_ens = [0.0, 0.0, 1.0, 2.0, 2.0, 3.5];
q = ens_quantiles(dup_ens, 6, true, false, 0, -99);
writematrix(q(:)', fullfile(outdir, 'ens_quantiles_dups.csv'));

%% ppi_update: Normal/Normal and Gamma state
[post_obs_inc, ~] = obs_increment_eakf(ens10, observation, obs_error_var);
post_obs = ens10 + post_obs_inc;
[post_state, ~, ~, ~, ~] = ppi_update(ens10, state10, post_obs, 'Normal', 'Normal');
writematrix(post_state(:)', fullfile(outdir, 'ppi_normal_normal.csv'));
[post_state, ~, ~, ~, ~] = ppi_update(ens10, state10, post_obs, 'Gamma', 'Normal');
writematrix(post_state(:)', fullfile(outdir, 'ppi_gamma_normal.csv'));
[post_state, ~, ~, ~, ~] = ppi_update(ens10, state10, post_obs, 'BNRH', 'RHF');
writematrix(post_state(:)', fullfile(outdir, 'ppi_bnrh_rhf.csv'));

%% inflate_gamma / inflate_bnrh
inf_ens = inflate_gamma(pos10, 10, 2.0);
writematrix(inf_ens(:)', fullfile(outdir, 'inflate_gamma.csv'));
inf_ens = inflate_bnrh(pos10, 10, 2.0, true, false, 0, -99);
writematrix(inf_ens(:)', fullfile(outdir, 'inflate_bnrh.csv'));

%% Model advances
global DELTAT SIGMA R B DELTA_T MODEL_SIZE %#ok<GVMIS>
L63 = lorenz_63_static_init_model();
DELTAT = L63.deltat; SIGMA = L63.sigma; R = L63.r; B = L63.b;
[x_new, ~] = lorenz_63_adv_1step([1.0, 2.0, 3.0], 0);
writematrix(x_new(:)', fullfile(outdir, 'lorenz63_step.csv'));

L96 = lorenz_96_static_init_model();
DELTA_T = L96.delta_t; MODEL_SIZE = L96.model_size;
x0 = 8.0 * ones(1, 40); x0(20) = 8.01;
[x_new, ~] = lorenz_96_adv_1step(x0, 0, 8.0);
writematrix(x_new(:)', fullfile(outdir, 'lorenz96_step.csv'));

%% advance_oned
vals = advance_oned([0.5, -1.0, 2.0], 0.1, 0.25);
writematrix(vals(:)', fullfile(outdir, 'advance_oned.csv'));

%% kurt and get_ens_rank
writematrix(kurt(ens10), fullfile(outdir, 'kurt.csv'));
writematrix([get_ens_rank(ens10, 0.0), get_ens_rank(ens10, 5.0), ...
             get_ens_rank(ens10, -5.0)], fullfile(outdir, 'ens_rank.csv'));

fprintf('Reference CSVs written to %s\n', outdir);
end
