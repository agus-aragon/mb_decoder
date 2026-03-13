# Acquisitions Log

---

**Which information does this log contain?**
This is a log file of the acquisitions, speficially of the EEG-fMRI session. It contains information about the EEG and MRI (errors, status) which might be relavant for downstream analysis. 

It also cointains the date of the acquisition, the interal ID (CRC participant's log book), and the temperatures reached by the first column (located "more" on the inside of the MR) of amplifiers (2 BrainAmps MR + 1 PowerPack). 


---
## sub-008
Date: 13/03/2026

CRC ID: 1602

### EEG
Ok

#### Amplifiers Temperature:
Order from top to bottom: Amp 6023 -> 47 C (channels #1) / PowerPack 41 C / Amp 9224 -> 46 C (channels #2)

### fMRI
T1 was repeated because participant move significantly during the first run. Thus, run 1 of the T1 should be used for preprocessing/analysis (t1_mpr_sag_p2_iso_2_)

---
## sub-007
Date: 11/03/2026

CRC ID: 1598

### EEG
EEG disconnected at trial 49 out of 50 of task-ES ("connection between Brainamp and USB2 Adapter / PCI is broken").

#### Amplifiers Temperature:
Order from top to bottom: Amp 6023 -> 50 C (channels #1) / PowerPack 45 C / Amp 9224 -> 50 C (channels #2)

### fMRI
Ok

---
## sub-006
Date: 10/03/2026

CRC ID: 4595

### EEG
EEG disconnected at trial 44 out of 50 of task-ES ("connection between Brainamp and USB2 Adapter / PCI is broken").

#### Amplifiers Temperature:
Order from top to bottom: Amp 9224 -> 58 C (channels #1) / PowerPack 40 C / Amp 6023 -> ~60 C (channels #2)

### fMRI
Ok


---
## sub-005
Date: 06/03/2026

CRC ID: 4588

### EEG
Ok during acquisition but .vmrk only has events for 15 trials (can and should be retrieved from Psychopy).

#### Amplifiers Temperature:
Order from top to bottom: Amp 9224 -> 52 C (channels #2) / PowerPack 44 C / Amp 6023 -> 50 C (channels #1)

### fMRI
Ok


---
## sub-004
Date: 03/03/2026

CRC ID: 4578

### EEG
Ok

#### Amplifiers Temperature:
Order from top to bottom: Amp 6023 -> 49 C / PowerPack 44 C / Amp 9224 -> 50 C

### fMRI
Ok

----------
## sub-003
Date: 20/02/2026

CRC ID: 4561

### EEG
BrainVision Recorder Template Error: Fc1 channel was recorder with "Low Cutoff [s] = DC" instead of 10

Impedance values are not real, they are consistent for each amplifier but during EEG set-up impedances where indeed lower than or around 10kOhm. 

EEG disconnected ar trial 43 out of 50 of task-ES ("connection between Brainamp and USB2 Adapter / PCI is broken").

The .vmrk has missing events: S 1, S 32, S 2, S 4.

#### Amplifiers Temperature:
Order from top to bottom: Amp 6023 -> 49 C / PowerPack 44 C / Amp 9224 -> 54 C

### fMRI
Gre Field Map was acquired after entering the MRI room to check for the EEG devices, given the error stated above. Therefore the room of the MRI was accessed, what causes minor air currents and static field inhomogeneities in the air/field. Nevertheless, this does not affect our current preprocessing pipelne as Distorsion Correction is performed with PEB/PEPOLAR sequences acquire before. 

----------
## sub-002
Date: 11/02/2026

CRC ID: 4540

### EEG
BrainVision Recorder Template Error: Fc1 channel was recorder with "Low Cutoff [s] = DC" instead of 10

#### Amplifiers Temperature:
Amplifiers: 41-45 C
PowerPack: 38 C

### fMRI
Ok

----------
## sub-001
Date: 10/02/2026

CRC ID: 4539

### EEG
BrainVision Recorder Template Error: Fc1 channel was recorder with "Low Cutoff [s] = DC" instead of 10

#### Temperature: 
Amplifiers: 45 C
PowerPack: 38 C

### fMRI
Ok