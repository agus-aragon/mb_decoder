%% Force output in non-GUI mode
if ~usejava('desktop')
    more off;
    format compact;
end

%% Parse arguments or use defaults
if ~exist('subject_arg', 'var')
    error('subject_arg not provided');
end
if ~exist('task_arg', 'var')
    error('task_arg not provided');
end
if ~exist('data_path', 'var')
    error('data_path not provided');
end
if ~exist('eeglab_path', 'var')
    error('eeglab_path not provided');
end
if ~exist('log_path', 'var')
    error('log_path not provided');
end

subject_id = sprintf('sub-%s', subject_arg);
task_name = task_arg;
fprintf('\n========================================\n');
fprintf('EEG-fMRI Artifact Removal Pipeline\n');
fprintf('========================================\n');
fprintf('Subject:     %s\n', subject_id);
fprintf('Task:        %s\n', task_name);
fprintf('Data path:   %s\n', data_path);
fprintf('EEGLAB path: %s\n', eeglab_path);
fprintf('Date:        %s\n', datestr(now));
fprintf('Log directory: %s\n', log_path);
fprintf('========================================\n\n');
drawnow;


%% Initialization
fprintf('EEG Preprocessing Started...\n');
addpath(genpath(eeglab_path));
eeglab nogui;

subject_path = fullfile(data_path, subject_id, 'eeg');
input_filename = sprintf('%s_task-%s_eeg.vhdr', subject_id, task_name);
input_filepath = fullfile(subject_path, input_filename);

%% Log
if ~exist(log_path, 'dir')
    mkdir(log_path);
    fprintf('Created log directory: %s\n', log_path);
end
diary_file = fullfile(log_path, sprintf('log_sub-%s_task-%s_eeglab-fmriartrem.txt', subject_arg, task_arg));
diary(diary_file);
diary on;
fprintf('EEGLAB-fmriartrem (fMRI artifact removal) logging to: %s\n\n', diary_file);
drawnow;

%% Load EEG data
tic;
[EEG, com] = pop_loadbv(subject_path, input_filename);
fprintf('EEG data loaded successfully.\n');
load_time = toc;
fprintf('Loaded successfully in %.2f seconds\n', load_time);
fprintf('      - Channels: %d\n', EEG.nbchan);
fprintf('      - Samples: %d\n', EEG.pnts);
fprintf('      - Duration: %.2f seconds\n', EEG.pnts/EEG.srate);
fprintf('      - Sampling rate: %.1f Hz\n', EEG.srate);

%% Clean Gradient Artifacts using FMRIB's fASTR
fprintf('Removing gradient artifacts using fASTR...\n');
tic;
EEG = pop_fmrib_fastr(EEG, 70, 5, 10, 'Scanner', 0, 0, 0, 0, 0, 0.03, 32:37, 'auto');
EEG = eeg_checkset(EEG);
ga_time = toc;
fprintf('Gradient artifacts removed in %.2f seconds\n', ga_time);

%% Cut data to match fMRI run duration (remove data before first Scanner trigger)
scanner_idx = find(strcmp({EEG.event.type}, 'Scanner'));
first_scanner_latency = EEG.event(scanner_idx(1)).latency;
first_scanner_time = first_scanner_latency / EEG.srate;

fprintf('      - Found %d Scanner triggers\n', length(scanner_idx));
fprintf('      - First Scanner trigger at %.2f seconds (sample %d)\n', ...
    first_scanner_time, round(first_scanner_latency));
fprintf('      - Removing %.2f seconds of pre-scan data\n', first_scanner_time);
drawnow;

% Trim data: keep from first Scanner trigger to end
cut_time = max(0, first_scanner_time - 0.001); % small offset to avoid cutting exactly at event
EEG = pop_select(EEG, 'time', [cut_time EEG.xmax]);
EEG = eeg_checkset(EEG);
fprintf('      - New duration: %.2f seconds\n', EEG.pnts/EEG.srate);
fprintf('      - New samples: %d\n', EEG.pnts);
drawnow;


%% Resample data to 500 Hz
fprintf('Resampling data to 500 Hz..\n');
tic;
EEG = pop_resample(EEG, 500);
EEG = eeg_checkset(EEG);
resample_time = toc;
fprintf('Data resampled in %.2f seconds\n', resample_time);

%% Remove BCG artifacts using CW Regression
fprintf('Removing BCG artifacts using CW Regression...\n');
tic;
EEG = pop_cwregression(EEG,500,4,0.021,1,'hann',33:37,1:31,'taperedhann',0);
bcg_time = toc;
fprintf('BCG artifacts removed in %.2f seconds\n', bcg_time);

%% Save intermediate processed data and log
fprintf('Saving processed EEG data...\n');
EEG = eeg_checkset(EEG);
output_filename = sprintf('%s_task-%s_desc-fmriClean_eeg.set', subject_id, task_name); % Save as .set for easier conversion to .fif

derivatives_path = fullfile(data_path, 'derivatives', 'eeglab_fmriartrem', subject_id, 'eeg');
if ~exist(derivatives_path, 'dir')
    mkdir(derivatives_path);
    fprintf('      - Created directory: %s\n', derivatives_path);
end
output_fullpath = fullfile(derivatives_path, output_filename);
EEG = pop_saveset(EEG, 'filename', output_filename, 'filepath', derivatives_path);
fprintf('Processed data saved as: %s\n', output_fullpath);

diary off;
fprintf('Log saved to: %s\n', diary_file);

%% Summary
total_time = load_time + ga_time + resample_time + bcg_time;
fprintf('\n========================================\n');
fprintf('Processing Complete!\n');
fprintf('========================================\n');
fprintf('Total processing time: %.2f minutes\n', total_time/60);
fprintf('Input:  %s\n', input_filepath);
fprintf('Output: %s\n', fullfile(derivatives_path, output_filename));
fprintf('========================================\n\n');
drawnow;

clear subject_arg task_arg data_path eeglab_path com;
clear load_time ga_time resample_time bcg_time total_time;