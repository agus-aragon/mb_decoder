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

## 

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
