%% Print text
fprintf('EEG Preprocessing Started...\n');

%% Load EEG data
EEG = pop_loadset('filename', 'subject1_raw.set', 'filepath', './data/');
fprintf('EEG data loaded successfully.\n');