## ###################### forms_to_bids ########################## ##
## This script formats the participants.tsv (with demographic and  ##
# questionnaires information) and the participants.json (with      ##
# explanation of participants.tsv).                                ##
#####################################################################
# %%

import os
import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata
import json

pd.set_option("future.no_silent_downcasting", True)
data_path = Path("/data/project/mb_decoder/data/subj_raw")
bids_path = data_path.parent / "bids" / "mb_decoder"


# Function for unifying format (remove french accents)
def remove_accents(text):
    if pd.isna(text):
        return text
    text = str(text).strip().replace("?", "")  # Clean first
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# %% Load & clean forms data
forms = pd.read_csv(
    data_path / "questionnaires_subjects(joined).csv",
    encoding="windows-1252",
    sep=";",
)
forms = forms.iloc[:50, :152]
string_cols = forms.select_dtypes(include=["object", "string"]).columns
forms[string_cols] = forms[string_cols].apply(
    lambda col: col.map(remove_accents)
)


# %% Score questionnaires

## EHI
EHI_mapping = {
    "Right": 1,
    "No preference": 0,
    "Left": -1,
    "Droite": 1,
    "Pas de preference": 0,
    "Gauche": -1,
}
EHI_cols = [col for col in forms.columns if col.startswith("EHI_")]
forms[EHI_cols] = forms[EHI_cols].replace(EHI_mapping).astype("Int64")
forms["EHI_score"] = forms[EHI_cols].sum(axis=1) / 10 * 100

## SRMBQ
SRMBQ_mapping = {
    "Almost never": 1,
    "Infrequently": 2,
    "Sometimes": 3,
    "Frequently": 4,
    "Almost always": 5,
    "Presque jamais": 1,
    "Rarement": 2,
    "Parfois": 3,
    "Souvent": 4,
    "Presque toujours": 5,
}
SRMBQ_cols = [col for col in forms.columns if col.startswith("SRMBQ_")]
forms[SRMBQ_cols] = forms[SRMBQ_cols].replace(SRMBQ_mapping).astype("Int64")
forms["SRMBQ_score"] = forms[SRMBQ_cols].sum(axis=1)

## MBQ
MBQ_mapping = {
    "Almost never": 1,
    "Very infrequently": 2,
    "Somewhat infrequently": 3,
    "Somewhat frequently": 4,
    "Very frequently": 5,
    "Almost always": 6,
    "Presque jamais": 1,
    "Tres rarement": 2,
    "Assez rarement": 3,
    "Assez frequemment": 4,
    "Tres frequemment": 5,
    "Presque toujours": 6,
}
MBQ_cols = [col for col in forms.columns if col.startswith("MBQ_")]
forms[MBQ_cols] = forms[MBQ_cols].replace(MBQ_mapping).astype("Int64")
forms["MBQ_score"] = forms[MBQ_cols].sum(axis=1)

## MCQ30
# Rename columns MCQ30_* to MCQ30_*
forms = forms.rename(
    columns={
        col: col.replace("MQC30_", "MCQ30_")
        for col in forms.columns
        if col.startswith("MQC30_")
    }
)

MCQ30_mapping = {
    "Do not agree": 1,
    "Agree slightly": 2,
    "Agree moderately": 3,
    "Agree very much": 4,
    "Pas d’accord": 1,
    "Legerement d’accord": 2,
    "Assez d’accord": 3,
    "Tout a fait d’accord": 4,
}
MCQ30_cols = [col for col in forms.columns if col.startswith("MCQ30_")]
forms[MCQ30_cols] = forms[MCQ30_cols].replace(MCQ30_mapping).astype("Int64")
forms["MCQ30_score"] = forms[MCQ30_cols].sum(axis=1)

# Indexing for subescales done with scale complete name and not with column index
# to avoid indexing issues given Python-based indexing (e.g, item 8 is column 7)
subscale_items_mcq30 = {
    "MCQ30_subscale_lack_of_cognitive_confidence": [
        "MCQ30_8",
        "MCQ30_14",
        "MCQ30_17",
        "MCQ30_24",
        "MCQ30_26",
        "MCQ30_29",
    ],
    "MCQ30_subscale_positive_beliefs_about_worry": [
        "MCQ30_1",
        "MCQ30_7",
        "MCQ30_10",
        "MCQ30_19",
        "MCQ30_23",
        "MCQ30_28",
    ],
    "MCQ30_subscale_cognitive_selfconsciousness": [
        "MCQ30_3",
        "MCQ30_5",
        "MCQ30_12",
        "MCQ30_16",
        "MCQ30_18",
        "MCQ30_30",
    ],
    "MCQ30_subscale_negative_beliefs_about_uncontrollability_and_danger": [
        "MCQ30_2",
        "MCQ30_4",
        "MCQ30_9",
        "MCQ30_11",
        "MCQ30_15",
        "MCQ30_21",
    ],
    "MCQ30_subscale_need_to_control_thoughts": [
        "MCQ30_6",
        "MCQ30_13",
        "MCQ30_20",
        "MCQ30_22",
        "MCQ30_25",
        "MCQ30_27",
    ],
}

for subscale_name, cols in subscale_items_mcq30.items():
    forms[subscale_name] = forms[cols].sum(axis=1)

## ACS
ACS_mapping = {
    "Almost never": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4,
    "Presque jamais": 1,
    "Parfois": 2,
    "Souvent": 3,
    "Toujours": 4,
}
ACS_cols = [col for col in forms.columns if col.startswith("ACS_")]
forms[ACS_cols] = forms[ACS_cols].replace(ACS_mapping).astype("Int64")

# REVERSE specific items (1-4 → 5-value)
reverse_acs_items = [
    "ACS_1",
    "ACS_2",
    "ACS_3",
    "ACS_6",
    "ACS_7",
    "ACS_8",
    "ACS_11",
    "ACS_12",
    "ACS_15",
    "ACS_16",
    "ACS_20",
]
for item in reverse_acs_items:
    forms[item] = 5 - forms[item]

forms["ACS_score"] = forms[ACS_cols].sum(axis=1)

subscale_items_acs = {
    "ACS_subscale_focus": [
        "ACS_1",
        "ACS_2",
        "ACS_3",
        "ACS_4",
        "ACS_5",
        "ACS_6",
        "ACS_7",
        "ACS_8",
        "ACS_9",
    ],
    "ACS_subscale_shift": [
        "ACS_10",
        "ACS_11",
        "ACS_12",
        "ACS_13",
        "ACS_14",
        "ACS_15",
        "ACS_16",
        "ACS_17",
        "ACS_18",
        "ACS_19",
        "ACS_20",
    ],
}

for subscale_name, cols in subscale_items_acs.items():
    forms[subscale_name] = forms[cols].sum(axis=1)

## ESS
ESS_cols = [col for col in forms.columns if col.startswith("ESS_")]

for col in ESS_cols:
    forms[col] = (
        forms[col]
        .astype(str)
        .str.strip()
        .str.replace("change", "chance", regex=False)
        .str.replace("'", "", regex=True)
        .str.replace("’", "", regex=True)  # Curly apostrophe
        .str.replace("`", "", regex=True)
        .str.replace("´", "", regex=True)
    )
ESS_mapping = {
    "Would never nod off": 0,
    "Slight chance of nodding off": 1,
    "Moderate chance of nodding off": 2,
    "High chance of nodding off": 3,
    "Aucune chance de somnoler ou de sendormir": 0,
    "Aucunechance desomnoleroudesendormir": 0,
    "Faible chance de sendormir": 1,
    "Faiblechance desendormir": 1,
    "Chance moyenne de sendormir": 2,
    "Chancemoyennedesendormir": 2,
    "Forte chance de sendormir": 3,
    "Forte chance desendormir": 3,
}

forms[ESS_cols] = forms[ESS_cols].replace(ESS_mapping).astype("Int64")
forms["ESS_score"] = forms[ESS_cols].sum(axis=1)


## Confidence
confidence_cols = [
    col for col in forms.columns if col.startswith("confidence_")
]

for col in confidence_cols:
    forms[col] = (
        forms[col]
        .astype(str)
        .str.strip()
        .str.replace("'", "", regex=True)
        .str.replace("’", "", regex=True)  # Curly apostrophe
        .str.replace("`", "", regex=True)
        .str.replace("´", "", regex=True)
    )

confidence_mapping = {
    "Very confident": 4,
    "Tres confiant": 4,
    "Confident": 3,
    "Confiant": 3,
    "Neutral": 2,
    "Neutre": 2,
    "Slightly confident": 1,
    "Legerement confiant": 1,
    "Not confident at all": 0,
    "Pas du tout confiant": 0,
    "Je ne lai pas declare": "n/a",
    "I did not report it":  "n/a"
}

forms[confidence_cols] = (
    forms[confidence_cols].replace(confidence_mapping)
)

## ARSQ
ARSQ_cols = [col for col in forms.columns if col.startswith("ARSQ_")]
for col in ARSQ_cols:
    forms[col] = (
        forms[col]
        .astype(str)
        .str.strip()
        .str.replace("'", "", regex=True)
        .str.replace("’", "", regex=True)  # Curly apostrophe
        .str.replace("`", "", regex=True)
        .str.replace("´", "", regex=True)
        .str.lower()
    )

ARSQ_mapping = {
    "completely disagree": 1,
    "pas du tout daccord": 1,
    "disagree": 2,
    "en desaccord": 2,
    "neither agree nor disagree": 3,  # Encoding error: This category was not included in the questionnaire
    "ni daccord ni en desaccord": 3,  # Encoding error: This category was not included in the questionnaire
    "agree": 4,
    "daccord": 4,
    "completely agree": 5,
    "tout a fait daccord": 5,
}
forms[ARSQ_cols] = forms[ARSQ_cols].replace(ARSQ_mapping).astype("Int64")
reverse_arsq_cols = ["ARSQ_8", "ARSQ_10", "ARSQ_23", "ARSQ_44"]

for item in reverse_arsq_cols:
    forms[item] = 5 - forms[item]

forms["ARSQ_score"] = forms[ARSQ_cols].sum(axis=1)

subscale_items_arsq = {
    "ARSQ_subscale_discontinuity_of_mind": [
        "ARSQ_2",
        "ARSQ_18",
        "ARSQ_23",
        "ARSQ_27",
        "ARSQ_34",
    ],
    "ARSQ_subscale_theory_of_mind": ["ARSQ_20", "ARSQ_35", "ARSQ_45"],
    "ARSQ_subscale_self": ["ARSQ_1", "ARSQ_16", "ARSQ_21"],
    "ARSQ_subscale_planning": [
        "ARSQ_15",
        "ARSQ_24",
        "ARSQ_29",
        "ARSQ_31",
        "ARSQ_32",
        "ARSQ_38",
    ],
    "ARSQ_subscale_sleepiness": ["ARSQ_3", "ARSQ_4", "ARSQ_26"],
    "ARSQ_subscale_comfort": ["ARSQ_5", "ARSQ_6", "ARSQ_7"],
    "ARSQ_subscale_somatic_awareness": [
        "ARSQ_14",
        "ARSQ_39",
        "ARSQ_42",
        "ARSQ_43",
    ],
}

for subscale_name, cols in subscale_items_arsq.items():
    forms[subscale_name] = forms[cols].sum(axis=1)

## Stress levels
stress_cols = [col for col in forms.columns if col.startswith("stress_")]

stress_mapping = {
    "Not at all stressed": 0,
    "Pas du tout stresse": 0,
    "Legerement stresse": 1,
    "Slightly stressed": 1,
    "Moderately stressed": 2,
    "Moderement stresse": 2,
    "Tres stresse": 3,
    "Very stressed": 3,
    "Extremely stressed": 4,
    "Extremement stresse": 4,
}

forms[stress_cols] = forms[stress_cols].replace(stress_mapping).astype("Int64")


forms.to_csv(bids_path / "participants.tsv", sep="\t", index=False)

# %% Participants.json with metadata for each column in participants.tsv

participants_json = {
    "participant_id": {
        "LongName": "",
        "Description": "unique identifier for each participant",
    },
    "sex": {
        "LongName": "",
        "Description": "sex of the participant as reported by the participant",
        "Levels": {"M": "male", "F": "female"},
    },
    "age": {
        "LongName": "",
        "Description": "age of the participant",
        "Units": "years",
    },
    "EHI_1": {
        "LongName": "Which hand do you prefer to use when writing?",
        "Description": "Edinburgh Handedness Inventory item 1",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_2": {
        "LongName": "Which hand do you prefer to use when drawing?",
        "Description": "Edinburgh Handedness Inventory item 2",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_3": {
        "LongName": "Which hand do you prefer to use when throwing?",
        "Description": "Edinburgh Handedness Inventory item 3",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_4": {
        "LongName": "Which hand do you prefer to use when using scissors?",
        "Description": "Edinburgh Handedness Inventory item 4",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_5": {
        "LongName": "Which hand do you prefer to use when brushing your teeth?",
        "Description": "Edinburgh Handedness Inventory item 5",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_6": {
        "LongName": "Which hand do you prefer to use when using a knife (without a fork)?",
        "Description": "Edinburgh Handedness Inventory item 6",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_7": {
        "LongName": "Which hand do you prefer to use when using a spoon?",
        "Description": "Edinburgh Handedness Inventory item 7",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_8": {
        "LongName": "Which hand do you prefer to use when using a broom (upper hand)?",
        "Description": "Edinburgh Handedness Inventory item 8",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_9": {
        "LongName": "Which hand do you prefer to use when striking a match?",
        "Description": "Edinburgh Handedness Inventory item 9",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "EHI_10": {
        "LongName": "Which hand do you prefer to use when opening a box (holding the lid)?",
        "Description": "Edinburgh Handedness Inventory item 10",
        "Levels": {
            1: "Right",
            0: "No preference",
            -1: "Left",
        },
    },
    "SRMBQ_1": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 1",
        "Description": "There are moments when I pay attention to nothing at all",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_2": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 2",
        "Description": "When I am tired of paying attention to someone speaking, my mind empties out",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_3": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 3",
        "Description": "I lose track of my thoughts and I can't remember what I was thinking",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_4": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 4",
        "Description": "When reading a book, I must reread pages because I ended at the end of the page I do not remember how I got there",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_5": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 5",
        "Description": "There are moments when I am sure that I had a thought, but I am not sure what",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_6": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 6",
        "Description": "There are moments when I know I was thinking of something, but I cannot recover it",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_7": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 7",
        "Description": "During the day, I notice that I am thinking of nothing",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_8": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 8",
        "Description": "When I am calm, it feels like my mind is empty",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_9": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 9",
        "Description": "My mind blanks when I am under pressure",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_10": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 10",
        "Description": "When something bad happens and I need to think of solutions, my mind goes blank",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_11": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 11",
        "Description": "When I am bored, I zone out without thinking of anything",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_12": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 12",
        "Description": "When I am sleepy, I easily forget what I am thinking about",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_13": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 13",
        "Description": "There are moments when it feels like I am not thinking about anything in particular",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_14": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 14",
        "Description": "When people ask me what I am thinking, I respond that I am thinking about nothing",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_15": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 15",
        "Description": "Periods of time can pass when I am not thinking of anything",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_16": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 16",
        "Description": "I notice myself staring at nothing without realizing how long it’s been",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_17": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 17",
        "Description": "I catch myself halfway through an action without knowing how I started it",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "SRMBQ_18": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 18",
        "Description": "During the day, I feel like I had brief time-skips, as if I missed the last few seconds",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always",
        },
    },
    "MBQ_1": {
        "LongName": "Mind Blanking Questionnaire item 1",
        "Description": "There are moments that I can't remember what I was just thinking about",
        "Levels": {
            1: "Almost never",
            2: "Very infrequently",
            3: "Somewhat infrequently",
            4: "Somewhat frequently",
            5: "Very frequently",
            6: "Almost always",
        },
    },
    "MBQ_2": {
        "LongName": "Mind Blanking Questionnaire item 2",
        "Description": "There are times when my mind goes completely blank",
        "Levels": {
            1: "Almost never",
            2: "Very infrequently",
            3: "Somewhat infrequently",
            4: "Somewhat frequently",
            5: "Very frequently",
            6: "Almost always",
        },
    },
    "MBQ_3": {
        "LongName": "Mind Blanking Questionnaire item 3",
        "Description": "I have times where I just space out without thinking about anything",
        "Levels": {
            1: "Almost never",
            2: "Very infrequently",
            3: "Somewhat infrequently",
            4: "Somewhat frequently",
            5: "Very frequently",
            6: "Almost always",
        },
    },
    "MBQ_4": {
        "LongName": "Mind Blanking Questionnaire item 4",
        "Description": "I find myself not knowing what I was doing even though I wasn't thinking about anything else",
        "Levels": {
            1: "Almost never",
            2: "Very infrequently",
            3: "Somewhat infrequently",
            4: "Somewhat frequently",
            5: "Very frequently",
            6: "Almost always",
        },
    },
    "MBQ_5": {
        "LongName": "Mind Blanking Questionnaire item 5",
        "Description": "There are moments when my mind empties out",
        "Levels": {
            1: "Almost never",
            2: "Very infrequently",
            3: "Somewhat infrequently",
            4: "Somewhat frequently",
            5: "Very frequently",
            6: "Almost always",
        },
    },
    "MCQ30_1": {
        "LongName": "Metacognitions Questionnaire 30 item 1",
        "Description": "Worry helps me to avoid problems in the future",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_2": {
        "LongName": "Metacognitions Questionnaire 30 item 2",
        "Description": "My worrying is dangerous for me",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_3": {
        "LongName": "Metacognitions Questionnaire 30 item 3",
        "Description": "I think a lot about my thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_4": {
        "LongName": "Metacognitions Questionnaire 30 item 4",
        "Description": "I could make myself sick with worrying",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_5": {
        "LongName": "Metacognitions Questionnaire 30 item 5",
        "Description": "I am aware of the way my mind works when I am thinking through a problem",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_6": {
        "LongName": "Metacognitions Questionnaire 30 item 6",
        "Description": "If I did not control a worrying thought, and then it happened, it would be my fault",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_7": {
        "LongName": "Metacognitions Questionnaire 30 item 7",
        "Description": "I need to worry in order to remain organized",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_8": {
        "LongName": "Metacognitions Questionnaire 30 item 8",
        "Description": "I have little confidence in my memory for words and names",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_9": {
        "LongName": "Metacognitions Questionnaire 30 item 9",
        "Description": "My worrying thoughts persists, no matter how I try to stop them",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_10": {
        "LongName": "Metacognitions Questionnaire 30 item 10",
        "Description": "Worrying helps me to get things sorted out in my mind",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_11": {
        "LongName": "Metacognitions Questionnaire 30 item 11",
        "Description": "I cannot ignore my worrying thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_12": {
        "LongName": "Metacognitions Questionnaire 30 item 12",
        "Description": "I monitory my thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_13": {
        "LongName": "Metacognitions Questionnaire 30 item 13",
        "Description": "I should be in control of my thoughts all the time",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_14": {
        "LongName": "Metacognitions Questionnaire 30 item 14",
        "Description": "My memory can mislead me at times",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_15": {
        "LongName": "Metacognitions Questionnaire 30 item 15",
        "Description": "My worrying could make me go mad",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_16": {
        "LongName": "Metacognitions Questionnaire 30 item 16",
        "Description": "I am constantly aware of my thinking",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_17": {
        "LongName": "Metacognitions Questionnaire 30 item 17",
        "Description": "I have a poor memory",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_18": {
        "LongName": "Metacognitions Questionnaire 30 item 18",
        "Description": "I pay close attention to thw way my mind works",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_19": {
        "LongName": "Metacognitions Questionnaire 30 item 19",
        "Description": "Worrying helps me cope",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_20": {
        "LongName": "Metacognitions Questionnaire 30 item 20",
        "Description": "Not being able to control my thoughts is a sign of weakness",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_21": {
        "LongName": "Metacognitions Questionnaire 30 item 21",
        "Description": "When I start worrying, I cannot stop",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_22": {
        "LongName": "Metacognitions Questionnaire 30 item 22",
        "Description": " will be punished for not controlling certain thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_23": {
        "LongName": "Metacognitions Questionnaire 30 item 23",
        "Description": "Worrying helps me to solve problems",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_24": {
        "LongName": "Metacognitions Questionnaire 30 item 24",
        "Description": "I have little confidence in my memory for places",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_25": {
        "LongName": "Metacognitions Questionnaire 30 item 25",
        "Description": "It is bad to think certain thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_26": {
        "LongName": "Metacognitions Questionnaire 30 item 26",
        "Description": "I do not trust my memory",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_27": {
        "LongName": "Metacognitions Questionnaire 30 item 27",
        "Description": "If I could not control my thoughts, I would not be able to function",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_28": {
        "LongName": "Metacognitions Questionnaire 30 item 28",
        "Description": "I need to worry, in order to work well",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_29": {
        "LongName": "Metacognitions Questionnaire 30 item 29",
        "Description": "I have little confidence in my memory for my actions",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "MCQ30_30": {
        "LongName": "Metacognitions Questionnaire 30 item 30",
        "Description": "I constantly examine my thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much",
        },
    },
    "ACS_1": {
        "LongName": "Attentional Control Scale item 1",
        "Description": "It’s very hard for me to concentrate on a difficult task when there are noises around.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_2": {
        "LongName": "Attentional Control Scale item 2",
        "Description": "When I need to concentrate and solve a problem, I have trouble focusing my attention.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_3": {
        "LongName": "Attentional Control Scale item 3",
        "Description": "When I am working hard on something, I still get distracted by events around me.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_4": {
        "LongName": "Attentional Control Scale item 4",
        "Description": "My concentration is good even if there is music in the room around me.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_5": {
        "LongName": "Attentional Control Scale item 5",
        "Description": "When concentrating, I can focus my attention so that I become unaware of what’s going on in the room around me.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_6": {
        "LongName": "Attentional Control Scale item 6",
        "Description": "When I am reading or studying, I am easily distracted if there are people talking in the same room.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_7": {
        "LongName": "Attentional Control Scale item 7",
        "Description": "When trying to focus my attention on something, I have difficulty blocking out distracting thoughts.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_8": {
        "LongName": "Attentional Control Scale item 8",
        "Description": "I have a hard time concentrating when I’m excited about something.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_9": {
        "LongName": "Attentional Control Scale item 9",
        "Description": "When concentrating I ignore feelings of hunger or thirst.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_10": {
        "LongName": "Attentional Control Scale item 10",
        "Description": "I can quickly switch from one task to another.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_11": {
        "LongName": "Attentional Control Scale item 11",
        "Description": "It takes me a while to get really involved in a new task.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_12": {
        "LongName": "Attentional Control Scale item 12",
        "Description": "It is difficult for me to coordinate my attention between listening and writing required when taking notes during lectures.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_13": {
        "LongName": "Attentional Control Scale item 13",
        "Description": "I can become interested in a new topic very quickly when I need to.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_14": {
        "LongName": "Attentional Control Scale item 14",
        "Description": "It is easy for me to read or write while I’m also talking on the phone.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_15": {
        "LongName": "Attentional Control Scale item 15",
        "Description": "I have trouble carrying on two conversations at once.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_16": {
        "LongName": "Attentional Control Scale item 16",
        "Description": "I have a hard time coming up with new ideas quickly",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ACS_17": {
        "LongName": "Attentional Control Scale item 17",
        "Description": "After being interrupted or distracted, I can easily shift my attention back to what I was doing before.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_18": {
        "LongName": "Attentional Control Scale item 18",
        "Description": "When a distracting thought comes to mind, it is easy for me to shift my attention away from it.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_19": {
        "LongName": "Attentional Control Scale item 19",
        "Description": "It is easy for me to alternate between two different tasks.",
        "Levels": {1: "Almost never", 2: "Sometimes", 3: "Often", 4: "Always"},
    },
    "ACS_20": {
        "LongName": "Attentional Control Scale item 20",
        "Description": "It is hard for me to break from one way of thinking about something and look at it from another point of view.",
        "Levels": {4: "Almost never", 3: "Sometimes", 2: "Often", 1: "Always"},
    },
    "ESS_1": {
        "LongName": "Epworth Sleepiness Scale item 1",
        "Description": "Sitting and reading",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "ESS_2": {
        "LongName": "Epworth Sleepiness Scale item 2",
        "Description": "Watching TV",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "ESS_3": {
        "LongName": "Epworth Sleepiness Scale item 3",
        "Description": "Sitting inactive in a public place (e.g., in a meeting, theater, or dinner event)",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "ESS_4": {
        "LongName": "Epworth Sleepiness Scale item 4",
        "Description": "As a passenger in a car for an hour or more without stopping for a break",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "ESS_5": {
        "LongName": "Epworth Sleepiness Scale item 5",
        "Description": "Lying down to rest when circumstances permit",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "ESS_6": {
        "LongName": "Epworth Sleepiness Scale item 6",
        "Description": "Sitting and talking to someone",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "ESS_7": {
        "LongName": "Epworth Sleepiness Scale item 7",
        "Description": "Sitting quietly after lunch without alcohol",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "ESS_8": {
        "LongName": "Epworth Sleepiness Scale item 8",
        "Description": "In a car, while stopped for a few minutes in traffic or at a light",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off",
        },
    },
    "hour_acquisition": {
        "Description": "Hour of the date that the MRI-EEG acquisition started (24 hr format)"
    },
    "confidence_thought": {
        "LongName": "How confident were you when you reported... Thought",
        "Levels": {
            0: "Not confident at all",
            1: "Slightly confident",
            2: "Neutral",
            3: "Confident",
            4: "Very confident",
            "NR": "I did not report it",
        },
    },
    "confidence_blank": {
        "LongName": "How confident were you when you reported... Blank",
        "Levels": {
            0: "Not confident at all",
            1: "Slightly confident",
            2: "Neutral",
            3: "Confident",
            4: "Very confident",
            "NR": "I did not report it",
        },
    },
    "confidence_sleep": {
        "LongName": "How confident were you when you reported... Asleep",
        "Levels": {
            0: "Not confident at all",
            1: "Slightly confident",
            2: "Neutral",
            3: "Confident",
            4: "Very confident",
            "NR": "I did not report it",
        },
    },
    "confidence_sensations": {
        "LongName": "How confident were you when you reported... Sensations",
        "Levels": {
            0: "Not confident at all",
            1: "Slightly confident",
            2: "Neutral",
            3: "Confident",
            4: "Very confident",
            "NR": "I did not report it",
        },
    },
    "ARSQ_1": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 1",
        "Description": "I thought about my feelings",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_2": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 2",
        "Description": "I felt restless",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_3": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 3",
        "Description": "I felt tired",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_4": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 4",
        "Description": "I felt sleepy",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_5": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 5",
        "Description": "I felt comfortable",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_6": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 6",
        "Description": "I felt relaxed",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_7": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 7",
        "Description": "I felt happy",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_8": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 8",
        "Description": "I felt ill",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_9": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 9",
        "Description": "I enjoyed the session",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_10": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 10",
        "Description": "I had negative feelings",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_11": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 11",
        "Description": "I felt bored",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_12": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 12",
        "Description": "I felt nothing",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_13": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 13",
        "Description": "I felt the same throughout the session",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_14": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 14",
        "Description": "I thought about my health",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_15": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 15",
        "Description": "I thought about my work/study",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_16": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 16",
        "Description": "I thought about my behavior",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_17": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 17",
        "Description": "I had thoughts that I would not readily share with others",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_18": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 18",
        "Description": "I had busy thoughts",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_19": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 19",
        "Description": "I had similar thoughts throughout the session",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_20": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 20",
        "Description": "I thought about others",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_21": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 21",
        "Description": "I thought about myself",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_22": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 22",
        "Description": "I thought about pleasant things",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_23": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 23",
        "Description": "I had my thoughts under control",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_24": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 24",
        "Description": "I thought about solving problems",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_25": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 25",
        "Description": "I thought about the aim of the experiment",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_26": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 26",
        "Description": "I had difficulty staying awake",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_27": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 27",
        "Description": "I had rapidly switching thoughts",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_28": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 28",
        "Description": "I had superficial thoughts",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_29": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 29",
        "Description": "I thought about the past",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_30": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 30",
        "Description": "I thought about the present",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_31": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 31",
        "Description": "I thought about the future",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_32": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 32",
        "Description": "I had deep thoughts",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_33": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 33",
        "Description": "I thought about nothing",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_34": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 34",
        "Description": "I had difficulty holding on to my thoughts",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_35": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 35",
        "Description": "I thought about people I like",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_36": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 36",
        "Description": "I thought in images",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_37": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 37",
        "Description": "I thought in words",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_38": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 38",
        "Description": "I thought about things I need to do",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_39": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 39",
        "Description": "I was conscious of my body",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_40": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 40",
        "Description": "I thought about the sounds around me",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_41": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 41",
        "Description": "I thought about the odors around me",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_42": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 42",
        "Description": "I thought about my heathbeat",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_43": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 43",
        "Description": "I thought about my breathing",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_44": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 44",
        "Description": "I felt pain",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_45": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 45",
        "Description": "I placed myself in other peoples' shoes",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_46": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 46",
        "Description": "I felt motivated to participate",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_47": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 47",
        "Description": "I have difficulty remembering my thoughts",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_48": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 48",
        "Description": "I have difficulty remembering my feelings",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_49": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 49",
        "Description": "I had my eyes closed",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "ARSQ_50": {
        "LongName": "Amsterdam Resting-State Questionnaire Item 50",
        "Description": "I was able to rate the statements",
        "Levels": {
            1: "Completely disagree",
            2: "Disagree",
            3: "MISSING given encoding error - (should be: Neither agree nor disagree)",
            4: "Agree",
            5: "Completely agree",
        },
    },
    "hours_sleep_night_before": {
        "LongName": "How many hours of sleep did you approximately have last night?",
        "Description": "Self-reported hours of sleep during the previous night",
        "Units": "hours",
    },
    "stress_week_before": {
        "LongName": "How much stress have you felt during the last week?",
        "Description": "Self-reported level of stress the week before the acquisition",
        "Levels": {
            0: "Not at all stressed",
            1: "Slightly stressed",
            2: "Moderately stressed",
            3: "Very stressed",
            4: "Extremely stressed",
        },
    },
    "stress_just_before_acquisition": {
        "LongName": "How much stress were you feeling just before starting the MRI scanning?",
        "Description": "Self-reported level of stress just before the acquisition",
        "Levels": {
            0: "Not at all stressed",
            1: "Slightly stressed",
            2: "Moderately stressed",
            3: "Very stressed",
            4: "Extremely stressed",
        },
    },
    "EHI_score": {
        "LongName": "Edinburgh Handedness Inventory (EHI) total score",
        "Description": "total score of the EHI. Right-handed > 40; Ambidextrous: 40 to -40; Left-handed < -40",
        "Units": "total score (addition)",
    },
    "SRMBQ_score": {
        "LongName": "Self-reported Mind Blanking Questionnaire total score",
        "Description": "total score of the SRMBQ",
        "Units": "total score (addition)",
    },
    "MBQ_score": {
        "LongName": "Mind Blanking Questionnaire total score",
        "Description": "total score of the MBQ",
        "Units": "total score (addition)",
    },
    "MCQ30_score": {
        "LongName": "Metacognition Questionnaire 30 total score",
        "Description": "total score of the MCQ30",
        "Units": "total score (addition)",
    },
    "MCQ30_subscale_lack_of_cognitive_confidence": {
        "LongName": "subscale of the Metacognition Questionnaire 30, Lack of Cognitive Confidence",
        "Description": "Measures lack of trust in one's cognition",
        "Units": "score of items that load for this subscale (addition)",
    },
    "MCQ30_subscale_positive_beliefs_about_worry": {
        "LongName": "subscale of the Metacognition Questionnaire 30, Positive Beliefs about Worry",
        "Description": "Measures the extend to which a person believes that worrying is beneficial for coping or problem-solving",
        "Units": "score of items that load for this subscale (addition)",
    },
    "MCQ30_subscale_cognitive_selfconsciousness": {
        "LongName": "subscale of the Metacognition Questionnaire 30, Cognitive Self-Consciousness",
        "Description": "Measures the tendency to monitor and focus one's attention intward on one's own thoughts",
        "Units": "score of items that load for this subscale (addition)",
    },
    "MCQ30_subscale_negative_beliefs_about_uncontrollability_and_danger": {
        "LongName": "subscale of the Metacognition Questionnaire 30, Negative Beliefs about Uncontrollability",
        "Description": "Measures the extend to which a person believes that worrying is out of control and cause physical or mental harm",
        "Units": "score of items that load for this subscale (addition)",
    },
    "MCQ30_subscale_need_to_control_thoughts": {
        "LongName": "subscale of the Metacognition Questionnaire 30, Need to control thoughts",
        "Description": "Measures the belief that certain types of thoughts are dangerous and must be controlled or stopped",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ACS_score": {
        "LongName": "Attentional Control Scale (ACS) total score",
        "Description": "total score of the ACS",
        "Units": "total score (addition)",
    },
    "ACS_subscale_focus": {
        "LongName": "subscale of the Attentional Control Scale, Attentional Focusing",
        "Description": "Measures the autorreported ability to intentionally mantain the attention on a task and resist distractions",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ACS_subscale_shift": {
        "LongName": "subscale of the Attentional Control Scale, Attentional Shifting",
        "Description": "Measures the autorreported ability to intentionally switch focus between different tasks",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ESS_score": {
        "LongName": "Epworth Sleepiness Scale (ESS) total score",
        "Description": "total score of the ESS",
        "Units": "total score (addition)",
    },
    "ARSQ_score": {
        "LongName": "Amsterdam Resting-State Scale (ARSQ) total score",
        "Description": "total score of the ARSQ",
        "Units": "total score (addition)",
    },
    "ARSQ_subscale_discontinuity_of_mind": {
        "LongName": "subscale of the Amsterdam Resting-State Scale (ARSQ), Discontinuity of Mind",
        "Description": "Measures rapidly switching thoughts, lack of focus and restlesness during the acquisition",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ARSQ_subscale_theory_of_mind": {
        "LongName": "subscale of the Amsterdam Resting-State Scale (ARSQ), Theory of Mind",
        "Description": "Measures thoughts directed towards other people during the acquisition",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ARSQ_subscale_self": {
        "LongName": "subscale of the Amsterdam Resting-State Scale (ARSQ), Self",
        "Description": "Measures thoughts about one's own feelings and introspection during the acquisition",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ARSQ_subscale_planning": {
        "LongName": "subscale of the Amsterdam Resting-State Scale (ARSQ), Planning",
        "Description": "Measures thoughts focused on the future and goals during the acquisition",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ARSQ_subscale_sleepiness": {
        "LongName": "subscale of the Amsterdam Resting-State Scale (ARSQ), Sleepiness",
        "Description": "Measures feelings of drowsiness during the acquisition",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ARSQ_subscale_comfort": {
        "LongName": "subscale of the Amsterdam Resting-State Scale (ARSQ), Comfort",
        "Description": "Measures feelings of relaxation and physical ease during the acquisition",
        "Units": "score of items that load for this subscale (addition)",
    },
    "ARSQ_subscale_somatic_awareness": {
        "LongName": "subscale of the Amsterdam Resting-State Scale (ARSQ), Somatic Awareness",
        "Description": "Measures the attention focused on bodily sensations during the acquisition",
        "Units": "score of items that load for this subscale (addition)",
    },
}

# Save .json
with open(bids_path / "participants.json", "w") as f:
    json.dump(participants_json, f, indent=4)

# %%
