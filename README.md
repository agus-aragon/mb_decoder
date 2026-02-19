# MBDecoder Project

---

**Authors:** Aragón Daud, Agustina; Boulakis, Paradeisios Alexandros; Simos, Nikolaos-Ioannis; Balla, Marion; Raimondo, Federico; Demertzi, Athena

**Contact:** a.aragondaud@uliege.be

---

## Summary 
The MBDecoder project aim to develop a machine learning (ML) model capable of identifying a mental state known as "Mind Blanking" (MB), in which people feel their mind is empty or have nothing to report about their internal experience [1,2], based on brain activity, particularly on whole-brain dynamic functional connectivity (dFC).

## Data Description

**Target Sample:** N=50 healthy adults  
🚧 Full dataset acquisition ongoing

**Inclusion Criteria:**
- Right-handed (Edinburgh Handedness Inventory)
- Age > 18 years

**Exclusion Criteria:**
- Standard MRI contraindications
- Signs of dementia (AD8 Screening)
- History of neurological/psychiatric disorders

**Acquisition Protocol:**
MRI (3T) with simulatenous EEG
1. **Resting-state:** 10-minute fMRI-EEG 
2. **Experience Sampling Task:** 40-minute fMRI-EEG with intermittent self-report of mental state (Thought, Mind Blanking, Asleep, Sensations) + arousal rating

**Questionnaires:**
- Mind Blanking Questionnaire (MBQ)
- Metacognitions Questionnaire-30 (MCQ-30)
- Attentional Control Scale (ACS)
- Epworth Sleepiness Scale (ESS)
- Mind Blaking Self-report Questionnaire ([OSF link](https://osf.io/k8nfa/overview))
- Amsterdam Resting-state Questionnaire (ARQ)
- Ad-hoc questions

---

## Usage & Reproduction Pipeline

### Psychopy Experiment
- Resting state acquisition: `experiment/run_rest.py`
- Experince Sampling Task: `experiment/run_task.py`
Conda Environment for PsychoPy experiment: `experiment/psychopy_env.yml`

### Prerequisites
- [fMRIprep](https://fmriprep.org/en/stable/) for fMRI preprocessing
- [EEGLAB-Matlab](https://eeglab.org) for initial EEG fMRI-related artifacts cleaning
- - [MNE-Python](https://mne.tools/stable/index.html) for EEG processing
- Python 3.8+ with scientific computing stack (NumPy, SciPy, scikit-learn)
🚧 Conda Environment to be uploaded

### Folders Organization
1. Convert raw data to BIDS format: `00_bids/`
2. Run fMRI and EEG preprocessing: `01_preprocessing`
🚧 Further analysis to be uploaded
---

## Code Execution
**1) Convert RAW to BIDS:** `bash src_to_bids.sh XXX YYYY` (where XXX is subject ID in BIDS format and YYYY is MRI scanner id (e.g, 4540)

**2) Preprocess fMRI:** could be run as `bash src_to_bids.sh XXX` (where XXX is subject ID in BIDS format, line 49 uncommented) or `bash src_to_bids.sh` (if line 48 is uncommented and 49 is commented)
**3) Preprocess EEG:

   (3.1) Clean MRI-related artifacts:`bash run_eeg.sh --subject XXX XXX --task ES rest` (it can run as many subjects as specificied)
   
   (3.2) Convert to fif: `python 1_convert_to_fif.py`
   
   (3.3) Pick bad channels (manual selection): `bash 2_pick_bad_chs.sh /data/project/mb_decoder/data/bids/mb_decoder` (manually select bad channels in pop-up window)
   
   (3.4) Cut in epochs: `python 3_cut_epochs.py`
   
   (3.5) Pick bad epochs (manual selection): `bash 4_pick_bad_epochs.sh /data/project/mb_decoder/data/bids/mb_decoder` (manually selected bad epochs in pop-up window)
   
   (3.6) Run ICA: `python 5_run_ica.py`
   
   (3.7) Create ICA report (manual selection): `bash 6_create_ica_report.sh` (open report and choose components to remove, then save the information in the .json)
   
   (3.8) Apply ICA: `python 8_apply_ica.py`
   

---

## Resources & Links

- **Open Science Framework (OSF):**  https://osf.io/kxj6z/overview  🚧 To be updated
  - *Contains:* Pilot data, study protocol, and preliminary materials
  - *Status:* 🚧 Regularly updated as the project progresses
  
- **Data Repository:** https://gitlab.uliege.be/poc/datasets/mb_decoder# 🚧 To be updated
  - *Contains:* Full dataset in BIDS format
  - *Status:* 🚧 Under active development

**Data Access Policy:**
- Pilot data is currently available on OSF
- Full dataset will be uploaded to the ULiège GitLab repository
- All data will be made publicly available upon publication of associated papers


---

*This project is actively under development. Please contact the authors for the most current status.*


[1] Andrillon, T., Lutz, A., Windt, J., & Demertzi, A. (2025). Where is my mind? A neurocognitive investigation of mind blanking. Trends in cognitive sciences, 29(7), 600-613. https://doi.org/10.1016/j.tics.2025.02.002

[2] Boulakis, P. A., & Demertzi, A. (2025). Relating mind-blanking to the content and dynamics of spontaneous thinking. Current Opinion in Behavioral Sciences, 61, 101481. https://doi.org/10.1016/j.cobeha.2024.101481
