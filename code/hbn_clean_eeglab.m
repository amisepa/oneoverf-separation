function hbn_clean_eeglab(in_dir, out_dir, method, wait)
% HBN_CLEAN_EEGLAB  Clean HBN resting recordings with GEDAI or ICA + ICLabel.
%
% For every <ds>_<sub>.set in IN_DIR (as downloaded by hbn_clean_check.py):
%   1. high-pass 0.4 Hz (pop_eegfiltnew) and resample to 250 Hz, the band
%      edge and rate of hbn_extract_psd.py
%   2. 60 and 120 Hz mains notch (band-stop FIR, passband edges f0 +/- 3 Hz,
%      1 Hz transition, applied twice for ~-106 dB; the fits stop at 55 Hz).
%      HBN mains at 59-61 Hz is a median 8,000x the 50-55 Hz power, and left
%      in, it steers both GEDAI and ICA.
%   3. average reference over all 129 channels, the zero-filled Cz (the
%      recording reference) included, so the reference is recovered without
%      losing rank (Kim et al., 2023) and channels next to Cz are not judged
%      on their small Cz-referenced signal; not repeated after step 4
%   4. bad channels with clean_rawdata's classic channel criteria (flatline
%      5 s, correlation 0.8, line noise 4 SD; ASR and window rejection off),
%      fixed random seed for the RANSAC step
%   5. METHOD
%      'gedai'    GEDAI v1.5, threshold 'auto', 12 cycles per wavelet band,
%                 low cut 0.5 Hz, BEM leadfield interpolated to the HydroCel
%                 positions (E1..E129 are not 10-5 names), no epoch rejection
%      'iclabel'  Picard ICA on a 1 Hz high-passed copy, PCA to the effective
%                 rank (eigenvalues of the covariance > 1e-7, Kim et al., 2023,
%                 against ghost ICs), weights applied to the 0.4 Hz data;
%                 components removed when ICLabel gives eye >= 0.8 or
%                 muscle >= 0.8
%   6. spherical-spline interpolation of the removed channels (after ICA, so
%      the interpolation adds no rank deficiency to it)
% No samples are removed, so event onsets in the BIDS events.tsv still apply.
% Writes <ds>_<sub>_clean.set to OUT_DIR, <ds>_<sub>_diag.mat (posterior-ROI
% spectra of the input and of the removed signal, both average-referenced)
% and one log row per recording to OUT_DIR/clean_log_<pid>.csv.
%
% Parallelise by starting several MATLAB processes on the same folders:
% each claims one recording at a time (OUT_DIR/claims/<name>, an atomic
% mkdir). Do not use a parallel pool: clean_channels calls rng('default')
% before its RANSAC, which on pool workers gives a different stream per
% worker, so the bad channels would depend on which worker cleaned the
% recording (2 of 3 test recordings changed). A separate process gets the
% same stream as a desktop session. With WAIT true a process keeps polling
% for new files until OUT_DIR/STOP exists (used by hbn_clean_all.py).
%
% Usage (MATLAB, one process per core):
%   maxNumCompThreads(1); hbn_clean_eeglab('C:/.../raw', 'C:/.../clean_iclabel', 'iclabel')

if nargin < 4, wait = false; end
assert(any(strcmp(method, {'gedai', 'iclabel'})), 'method must be gedai or iclabel');
addpath('C:/Users/ccann/Documents/MATLAB/eeglab');
eeglab('nogui');
claims = fullfile(out_dir, 'claims');
if ~exist(claims, 'dir'), mkdir(claims); end
logf = fullfile(out_dir, sprintf('clean_log_%d.csv', feature('getpid')));
if ~exist(logf, 'file')
    fid = fopen(logf, 'w');
    fprintf(fid, 'file,status,n_chan,n_bad,bad_labels,score,removed_share,n_ic_removed,n_pc,lambda_min_kept,seconds\n');
    fclose(fid);
end

n_done = 0;
while true
    files = dir(fullfile(in_dir, '*.set'));
    names = {files.name};
    todo = names(cellfun(@(n) ~exist(fullfile(out_dir, strrep(n, '.set', '_clean.set')), 'file') ...
        && ~exist(fullfile(claims, n), 'dir'), names));
    name = '';
    for k = 1:numel(todo)
        [~, ~, id] = mkdir(claims, todo{k});
        if isempty(id), name = todo{k}; break; end   % created here; 'DirectoryExists' means another process has it
    end
    if isempty(name)
        if ~wait || exist(fullfile(out_dir, 'STOP'), 'file'), break; end
        pause(30);
        continue
    end
    row = clean_one(in_dir, out_dir, name, method);
    fid = fopen(logf, 'a');
    fprintf(fid, '%s\n', row);
    fclose(fid);
    if ~contains(row, ',error: ')   % a failed recording keeps its claim, so it is not retried until restart
        rmdir(fullfile(claims, name));
    end
    n_done = n_done + 1;
end
fprintf('%s: process %d cleaned %d recordings\n', method, feature('getpid'), n_done);
end


function row = clean_one(in_dir, out_dir, name, method)
t0 = tic;
roi_labels = {'E62', 'E65', 'E70', 'E75', 'E83', 'E90'};
try
    EEG = pop_loadset('filename', name, 'filepath', in_dir, 'verbose', 'off');
    EEG = pop_eegfiltnew(EEG, 'locutoff', 0.4);
    EEG = pop_resample(EEG, 250);
    for f0 = [60 120]
        for pass = 1:2
            EEG = pop_eegfiltnew(EEG, 'locutoff', f0 - 3, 'hicutoff', f0 + 3, ...
                'revfilt', 1, 'filtorder', 826);
        end
    end
    EEG = pop_reref(EEG, []);
    orig = EEG.chanlocs;

    rng(0);
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'ChannelCriterion', 0.8, ...
        'LineNoiseCriterion', 4, 'Highpass', 'off', 'BurstCriterion', 'off', ...
        'WindowCriterion', 'off', 'BurstRejection', 'off', 'Distance', 'Euclidian');
    bad = setdiff({orig.labels}, {EEG.chanlocs.labels}, 'stable');

    n_ic = NaN; n_pc = NaN; lam = NaN(1, 2);
    switch method
        case 'gedai'
            [EEG, ART, score, ~, ~, share] = ...
                GEDAI(EEG, 'auto', 12, 0.5, 'interpolated', false, false);
            removed = ART.data;
            input = EEG.data + ART.data;
        case 'iclabel'
            input = EEG.data;
            HP = pop_eegfiltnew(EEG, 'locutoff', 1);
            C = cov(double(HP.data'));
            ev = sort(eig((C + C') / 2), 'descend');
            n_pc = sum(ev > 1e-7);
            lam = [ev(n_pc), ev(min(n_pc + 1, end))];   % smallest kept, largest dropped
            HP = pop_runica(HP, 'icatype', 'picard', 'pca', n_pc, 'maxiter', 500);
            EEG.icaweights = HP.icaweights;
            EEG.icasphere = HP.icasphere;
            EEG.icawinv = HP.icawinv;
            EEG.icachansind = HP.icachansind;
            EEG = eeg_checkset(EEG, 'ica');
            EEG = pop_iclabel(EEG, 'default');
            P = EEG.etc.ic_classification.ICLabel.classifications;   % brain muscle eye heart line chan other
            rej = find(P(:, 3) >= 0.8 | P(:, 2) >= 0.8);
            n_ic = numel(rej);
            if ~isempty(rej)
                EEG = pop_subcomp(EEG, rej, 0);
            end
            removed = input - EEG.data;
            score = NaN;
            share = sum(removed(:) .^ 2) / sum(double(input(:)) .^ 2);
    end

    roi = find(ismember({EEG.chanlocs.labels}, roi_labels));
    win = 4 * EEG.srate;
    [p_in, fr] = pwelch(double(input(roi, :))', hann(win, 'periodic'), win/2, win, EEG.srate);
    p_rm = pwelch(double(removed(roi, :))', hann(win, 'periodic'), win/2, win, EEG.srate);
    diag = struct('freqs', fr, 'psd_in', mean(p_in, 2), 'psd_removed', mean(p_rm, 2), ...
        'removed_share', share, 'score', score, 'n_ic_removed', n_ic, 'n_pc', n_pc, ...
        'lambda_kept_dropped', lam, 'bad', {bad});
    save(fullfile(out_dir, strrep(name, '.set', '_diag.mat')), '-struct', 'diag');

    EEG = pop_interp(EEG, orig, 'spherical');
    EEG.setname = strrep(name, '.set', '_clean');
    EEG.data = single(EEG.data);
    EEG.icaweights = []; EEG.icasphere = []; EEG.icawinv = []; EEG.icachansind = [];
    % write under a temporary name, so a killed process leaves no file that looks finished
    tmp = strrep(name, '.set', '_clean_part.set');
    pop_saveset(EEG, 'filename', tmp, 'filepath', out_dir, 'savemode', 'onefile', 'version', '7');
    movefile(fullfile(out_dir, tmp), fullfile(out_dir, strrep(name, '.set', '_clean.set')));
    row = sprintf('%s,ok,%d,%d,%s,%.3f,%.4f,%g,%g,%.3g,%.1f', name, numel(orig), numel(bad), ...
        strjoin(bad, ' '), score, share, n_ic, n_pc, lam(1), toc(t0));
catch err
    row = sprintf('%s,error: %s,,,,,,,,,%.1f', name, strrep(err.message, ',', ';'), toc(t0));
end
fprintf('ROW %s\n', row);
end
