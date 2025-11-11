# MBDecoder Project

---

**Authors:** Aragón Daud, Agustina; Boulakis, Paradeisios Alexandros; Raimondo, Federico; Demertzi, Athena
**Contact:** a.aragondaud@uliege.be

---
## Summary 
The MBDecoder projects aim to develop a machine learning (ML) model capable of identifying a mental state known as "Mind Blanking" (MB), in which people feel their mind is empty or have nothing to report about their internal experience, based on brain activity, particularly on whole-brain dynamic functional connectivity (dFC).

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
- [MNE-Python](https://mne.tools/stable/index.html) for EEG processing
- [BrainVision Analyzer](https://www.brainproducts.com/products/) for initial EEG artifact handling
- Python 3.8+ with scientific computing stack (NumPy, SciPy, scikit-learn)
🚧 Conda Environment to be uploaded

### Data Preprocessing
1. Convert raw data to BIDS format in `00_bids/`
2. Run fMRI preprocessing: `01_preprocessing/fMRI/run_fmriprep.sh`
🚧 EEG preprocessing pipeline to be uploaded

🚧 Further analysis to be uploaded
---

## Preliminary Analysis Plan
We are currently acquiring data and setting the analysis pipeline. 

---

## Resources & Links

- **Open Science Framework (OSF):**  https://osf.io/kxj6z/overview  🚧 To be updated
- **Data Repository:** https://gitlab.uliege.be/poc/datasets/mb_decoder# 🚧 To be updated


---

*This project is actively under development. Please contact the authors for the most current status.*


