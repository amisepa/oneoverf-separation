function clean_eeglab(job_dir, out_dir, wait)
% CLEAN_EEGLAB  ICA + ICLabel cleaning of the adult resting datasets
% (Dortmund, LEMON, SRM), with the steps of hbn_clean_eeglab.m.
%
% One job per recording session, JOB_DIR/<name>.json (written by
% clean_adult.py), with the fields
%   files   the session's recordings (EDF or BrainVision .vhdr), in
%           JOB_DIR/<name>/; concatenated, so the session gets one
%           decomposition
%   ref     label of the recording reference, absent from the data ('FCz'),
%           added as a zero channel; '' when the data have none
%   mains   mains frequency in Hz (50)
%   eog     labels kept out of the cleaning (LEMON's VEOG) and returned
%           filtered and resampled but otherwise unchanged
%   roi     labels of the diagnostic spectra
% Per session:
%   1. load; the zero reference channel; 10-05 positions by label, channels
%      without one dropped (EOG excepted)
%   2. resample to 250 Hz, then high-pass 0.4 Hz (pop_eegfiltnew), per
%      recording (HBN filters first; at 1000-2500 Hz the 0.4 Hz FIR is 10x
%      longer, and the two commute below 100 Hz)
%   3. mains notch at f0 and 2 f0 (band-stop FIR, passband edges f0 +/- 3 Hz,
%      1 Hz transition, twice for ~-106 dB; outside the 2-45 Hz fits)
%   4. average reference over the EEG channels, the reference included
%      (Kim et al., 2023); not repeated after 5
%   5. bad channels by clean_rawdata's channel criteria (flatline 5 s,
%      correlation 0.8, line noise 4 SD; ASR and window rejection off)
%   6. Picard ICA on a 1 Hz high-passed copy, PCA to the effective rank
%      (eigenvalues of the covariance > 1e-7, Kim et al., 2023), weights
%      applied to the 0.4 Hz data; components removed when ICLabel gives
%      eye >= 0.8 or muscle >= 0.8
%   7. spherical-spline interpolation of the removed channels
%   8. split back into the input recordings, each saved in the input's
%      channel order (the reference last) as OUT_DIR/<input stem>.set
% No samples are removed, so marker times still apply. Also writes
% OUT_DIR/<name>_diag.mat (ROI spectra of the input and of the removed
% signal) and one log row per session to OUT_DIR/clean_log_<pid>.csv.
% Parallelise with separate MATLAB processes, never a pool (see
% hbn_clean_eeglab.m): each claims one job at a time (OUT_DIR/claims/<name>).
% With WAIT true a process keeps polling until OUT_DIR/STOP exists.
%
% Usage (MATLAB, one process per core):
%   maxNumCompThreads(1); clean_eeglab('C:/.../jobs', 'C:/.../clean', true)

if nargin < 3, wait = false; end
addpath('C:/Users/ccann/Documents/MATLAB/eeglab');
eeglab('nogui');
claims = fullfile(out_dir, 'claims');
if ~exist(claims, 'dir'), mkdir(claims); end
logf = fullfile(out_dir, sprintf('clean_log_%d.csv', feature('getpid')));
if ~exist(logf, 'file')
    fid = fopen(logf, 'w');
    fprintf(fid, 'job,status,n_chan,n_bad,bad_labels,n_ic_removed,n_pc,lambda_min_kept,removed_share,seconds\n');
    fclose(fid);
end

n_done = 0;
while true
    jobs = dir(fullfile(job_dir, '*.json'));
    names = cellfun(@(n) n(1:end - 5), {jobs.name}, 'UniformOutput', false);
    todo = names(cellfun(@(n) ~exist(fullfile(out_dir, [n '_diag.mat']), 'file') ...
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
    row = clean_one(job_dir, out_dir, name);
    fid = fopen(logf, 'a');
    fprintf(fid, '%s\n', row);
    fclose(fid);
    if ~contains(row, ',error: ')   % a failed job keeps its claim, so it is not retried until restart
        rmdir(fullfile(claims, name));
    end
    n_done = n_done + 1;
end
fprintf('process %d cleaned %d sessions\n', feature('getpid'), n_done);
end


function row = clean_one(job_dir, out_dir, name)
t0 = tic;
try
    job = jsondecode(fileread(fullfile(job_dir, [name '.json'])));
    files = cellstr(job.files);
    eog = cellstr(job.eog);
    eog = eog(~cellfun('isempty', eog));
    elc = 'C:/Users/ccann/Documents/MATLAB/eeglab/plugins/dipfit/standard_BEM/elec/standard_1005.elc';

    % 1-2: load each recording, add the reference, filter and resample it on its own
    runs = cell(1, numel(files));
    for r = 1:numel(files)
        f = fullfile(job_dir, name, files{r});
        if endsWith(f, '.vhdr', 'IgnoreCase', true)
            [p, n, e] = fileparts(f);
            EEG = pop_loadbv(p, [n e]);
        else
            EEG = pop_biosig(f, 'importevent', 'off', 'importannot', 'off');
        end
        if ~isempty(job.ref)
            EEG.data(end + 1, :) = 0;
            EEG.nbchan = EEG.nbchan + 1;
            EEG.chanlocs(end + 1).labels = job.ref;
        end
        EEG = pop_chanedit(EEG, 'lookup', elc);
        % channels without a 10-05 position are not EEG (biosig adds the EDF
        % 'Status' channel): drop them, the EOG channels excepted
        keep = ~cellfun('isempty', {EEG.chanlocs.X}) | ismember({EEG.chanlocs.labels}, eog);
        if ~all(keep)
            EEG = pop_select(EEG, 'channel', find(keep));
        end
        if any(ismember({EEG.chanlocs.labels}, eog))
            [EEG.chanlocs(ismember({EEG.chanlocs.labels}, eog)).type] = deal('EOG');
        end
        EEG = pop_resample(EEG, 250);
        EEG = pop_eegfiltnew(EEG, 'locutoff', 0.4);
        runs{r} = EEG;
    end
    lens = cellfun(@(E) E.pnts, runs);
    order = {runs{1}.chanlocs.labels};   % the saved files keep this channel order
    EEG = runs{1};
    datas = cellfun(@(E) E.data, runs, 'UniformOutput', false);
    EEG.data = cat(2, datas{:});
    clear datas
    EEG.pnts = size(EEG.data, 2);
    EEG.xmax = (EEG.pnts - 1) / EEG.srate;
    EEG.times = (0:EEG.pnts - 1) / EEG.srate * 1000;
    evs = struct('type', {}, 'latency', {}, 'duration', {});   % runs may carry different event fields
    for r = 1:numel(runs)
        off = sum(lens(1:r - 1));
        if r > 1
            evs(end + 1) = struct('type', 'boundary', 'latency', off + 0.5, 'duration', 0); %#ok<AGROW>
        end
        for e = runs{r}.event(:)'
            d = 0;
            if isfield(e, 'duration') && ~isempty(e.duration), d = e.duration; end
            evs(end + 1) = struct('type', e.type, 'latency', e.latency + off, 'duration', d); %#ok<AGROW>
        end
    end
    EEG.event = evs;
    EEG.urevent = [];
    EEG = eeg_checkset(EEG, 'eventconsistency');

    % 3: notch, per recording across the boundaries
    for f0 = job.mains * [1 2]
        for pass = 1:2
            EEG = pop_eegfiltnew(EEG, 'locutoff', f0 - 3, 'hicutoff', f0 + 3, ...
                'revfilt', 1, 'filtorder', 826);
        end
    end

    % EOG channels leave here and come back unchanged at the end
    is_eog = ismember({EEG.chanlocs.labels}, eog);
    eog_data = EEG.data(is_eog, :);
    eog_locs = EEG.chanlocs(is_eog);
    if any(is_eog)
        EEG = pop_select(EEG, 'nochannel', find(is_eog));
    end

    % 4-5
    EEG = pop_reref(EEG, []);
    orig = EEG.chanlocs;
    rng(0);
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'ChannelCriterion', 0.8, ...
        'LineNoiseCriterion', 4, 'Highpass', 'off', 'BurstCriterion', 'off', ...
        'WindowCriterion', 'off', 'BurstRejection', 'off', 'Distance', 'Euclidian');
    bad = setdiff({orig.labels}, {EEG.chanlocs.labels}, 'stable');

    % 6
    input = EEG.data;
    HP = pop_eegfiltnew(EEG, 'locutoff', 1);
    C = cov(double(HP.data'));   % boundaries: one covariance over the concatenated recordings
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
    if ~isempty(rej)
        EEG = pop_subcomp(EEG, rej, 0);
    end
    removed = input - EEG.data;
    share = sum(removed(:) .^ 2) / sum(double(input(:)) .^ 2);

    roi = find(ismember({EEG.chanlocs.labels}, cellstr(job.roi)));
    win = 4 * EEG.srate;
    [p_in, fr] = pwelch(double(input(roi, :))', hann(win, 'periodic'), win/2, win, EEG.srate);
    p_rm = pwelch(double(removed(roi, :))', hann(win, 'periodic'), win/2, win, EEG.srate);
    diag = struct('freqs', fr, 'psd_in', mean(p_in, 2), 'psd_removed', mean(p_rm, 2), ...
        'removed_share', share, 'n_ic_removed', numel(rej), 'n_pc', n_pc, ...
        'lambda_kept_dropped', lam, 'bad', {bad}, 'files', {files}, ...
        'ic_removed_classes', P(rej, :));   % ICLabel probabilities of the removed components

    % 7-8
    EEG = pop_interp(EEG, orig, 'spherical');
    EEG.icaweights = []; EEG.icasphere = []; EEG.icawinv = []; EEG.icachansind = [];
    EEG.icaact = [];
    if ~isempty(eog_locs)
        EEG.data = [EEG.data; eog_data];
        for k = 1:numel(eog_locs)   % field by field: the interpolated chanlocs may carry other fields
            EEG.chanlocs(end + 1).labels = eog_locs(k).labels;
            EEG.chanlocs(end).type = 'EOG';
        end
        EEG.nbchan = size(EEG.data, 1);
        EEG = eeg_checkset(EEG);
    end
    [~, ord] = ismember(order, {EEG.chanlocs.labels});
    EEG.data = EEG.data(ord, :);
    EEG.chanlocs = EEG.chanlocs(ord);
    EEG.data = single(EEG.data);
    stop = cumsum(lens);
    start = [1, stop(1:end - 1) + 1];
    for r = 1:numel(files)
        R = pop_select(EEG, 'point', [start(r) stop(r)]);
        [~, stem] = fileparts(files{r});
        R.setname = stem;
        tmp = [stem '_part.set'];
        pop_saveset(R, 'filename', tmp, 'filepath', out_dir, 'savemode', 'onefile', 'version', '7');
        movefile(fullfile(out_dir, tmp), fullfile(out_dir, [stem '.set']));
    end
    save(fullfile(out_dir, [name '_diag.mat']), '-struct', 'diag');
    row = sprintf('%s,ok,%d,%d,%s,%d,%d,%.3g,%.4f,%.1f', name, numel(orig), numel(bad), ...
        strjoin(bad, ' '), numel(rej), n_pc, lam(1), share, toc(t0));
catch err
    row = sprintf('%s,error: %s,,,,,,,,%.1f', name, strrep(err.message, ',', ';'), toc(t0));
end
fprintf('ROW %s\n', row);
end
