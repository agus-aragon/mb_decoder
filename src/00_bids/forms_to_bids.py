# %%
import os

import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata
import json

pd.set_option("future.no_silent_downcasting", True)
datapath = Path("/Users/agusaragon/Downloads/") #/data/project/mb_decoder/data/subj_raw
bids_path = datapath.parent() / "bids" / "mb_decoder"

# Function for unifying format (remove french accents)
def remove_accents(text):
    if pd.isna(text):
        return text
    text = str(text).strip()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# %% Load & clean forms data
forms = pd.read_csv(
    datapath / "questionnaires_subjects(joined).csv",
    encoding="windows-1252",
    sep=";",
)

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

## AD8
AD8_mapping = {
    "Oui, un changement": 1,
    "Non, Pas de changement": 0,
    "NA, Ne sais pas": pd.NA,
    "YES, a change": 1,
    "NO, no change": 0,
    "N/A, Don't know": pd.NA
}
AD8_cols = [col for col in forms.columns if col.startswith("AD8_")]
forms[AD8_cols] = forms[AD8_cols].replace(AD8_mapping).astype("Int64")
forms["AD8_score"] = forms[AD8_cols].sum(axis=1)
forms.loc[forms["age"] < 60, "AD8_score"] = pd.NA



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
MCQ30_cols = [col for col in forms.columns if col.startswith("MQC30_")]
forms[MCQ30_cols] = forms[MCQ30_cols].replace(MCQ30_mapping).astype("Int64")
forms["MCQ30_score"] = forms[MCQ30_cols].sum(axis=1)

subscale_items_mcq30 = {
    "MCQ30_lack_of_cognitive_confidence": [8, 14, 17, 24, 26, 29],
    "MCQ30_positive_beliefs_about_worry": [1, 7, 10, 19, 23, 28],
    "MCQ30_cognitive_selfconsciousness": [3, 5, 12, 16, 18, 30],
    "MCQ30_negative_beliefs_about_uncontrollability_and_danger": [
        2,
        4,
        9,
        11,
        15,
        21,
    ],
    "MCQ30_need_to_control_thoughts": [6, 13, 20, 22, 25, 27],
}

forms["MCQ30_subscore_lack_of_cognitive_confidence"] = forms[
    [
        f"MQC30_{i}"
        for i in subscale_items_mcq30["MCQ30_lack_of_cognitive_confidence"]
    ]
].sum(axis=1)
forms["MCQ30_subscore_positive_beliefs_about_worry"] = forms[
    [
        f"MQC30_{i}"
        for i in subscale_items_mcq30["MCQ30_positive_beliefs_about_worry"]
    ]
].sum(axis=1)
forms["MCQ30_subscore_cognitive_selfconsciousness"] = forms[
    [f"MQC30_{i}" for i in subscale_items_mcq30["MCQ30_cognitive_selfconsciousness"]]
].sum(axis=1)
forms["MCQ30_subscore_negative_beliefs_about_uncontrollability_and_danger"] = (
    forms[
        [
            f"MQC30_{i}"
            for i in subscale_items_mcq30[
                "MCQ30_negative_beliefs_about_uncontrollability_and_danger"
            ]
        ]
    ].sum(axis=1)
)
forms["MCQ30_subscore_need_to_control_thoughts"] = forms[
    [f"MQC30_{i}" for i in subscale_items_mcq30["MCQ30_need_to_control_thoughts"]]
].sum(axis=1)

## ACS
ACS_mapping = {
    'Almost never': 1,
    'Sometimes': 2,
    'Often': 3,
    'Always': 4,
    'Presque jamais': 1,
    'Parfois': 2,
    'Souvent': 3,
    'Toujours': 4
}
ACS_cols = [col for col in forms.columns if col.startswith("ACS_")]
forms[ACS_cols] = forms[ACS_cols].replace(ACS_mapping).astype("Int64")

# REVERSE specific items (1-4 → 5-value)
reverse_acs_items = [1, 2, 3, 6, 7, 8, 11, 12, 15, 16, 20]
for item in reverse_acs_items:
    col_name = f"ACS_{item}"
    forms[col_name] = 5 - forms[col_name]  # 1→4, 2→3, 3→2, 4→1
forms["ACS_score"] = forms[ACS_cols].sum(axis=1)

subscale_items_acs = {
    'ACS_subscore_focus': [1,2,3,4,5,6,7,8,9],
    'ACS_subscore_shift': [10,11,12,13,14,15,16,17,18,19,20]
}

forms["ACS_subscore_focus"] = forms[
    [
        f"ACS_{i}"
        for i in subscale_items_acs["ACS_subscore_focus"]
    ]
].sum(axis=1)
forms["ACS_subscore_shift"] = forms[
    [
        f"ACS_{i}"
        for i in subscale_items_acs["ACS_subscore_shift"]
    ]
].sum(axis=1)


## ESS
ESS_mapping = {
    "Would never nod off": 0,
    "Slight chance of nodding off": 1,
    "Moderate chance of nodding off": 2,
    "High chance of nodding off": 3,
    "Aucune chance de somnoler ou de s’endormir": 0,
    "Faible chance de s’endormir": 1,
    "Chance moyenne de s’endormir": 2,
    "Forte chance de s’endormir": 3
}
ESS_cols = [col for col in forms.columns if col.startswith("ESS_")]
forms[ESS_cols] = forms[ESS_cols].replace(ESS_mapping).astype("Int64")
forms["ESS_score"] = forms[ESS_cols].sum(axis=1)

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
        "age": {
            "LongName": "",
            "Description": "age of the participant",
            "Units": "years",
        },
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
    "AD8_1":{
        "LongName": "Dementia Screening Interview item 1",
        "Description": "Problems with judgment (e.g., problems making decisions, bad financial decisions, problems with thinking)",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "AD8_2":{
        "LongName": "Dementia Screening Interview item 2",
        "Description": "Less interest in hobbies/activities",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "AD8_3":{
        "LongName": "Dementia Screening Interview item 3",
        "Description": "Repeats questions, stories, or statements",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "AD8_4":{
        "LongName": "Dementia Screening Interview item 4",
        "Description": "Trouble learning how to use a tool, appliance, or gadget (e.g., VCR, computer, microwave)",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "AD8_5":{
        "LongName": "Dementia Screening Interview item 5",
        "Description": "Forgets correct month or year",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "AD8_6":{
        "LongName": "Dementia Screening Interview item 6",
        "Description": "Trouble handling complicated financial affairs (e.g., balancing checkbook, income taxes, paying bills)",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "AD8_7":{
        "LongName": "Dementia Screening Interview item 7",
        "Description": "Trouble remembering appointments",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "AD8_8":{
        "LongName": "Dementia Screening Interview item 8",
        "Description": "Daily problems with thinking and/or memory",
        "Levels": {
            1: "Yes, a change",
            0: "No, no change",
            pd.NA: "N/A, Don't know"
        }
    },
    "SRMBQ_1": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 1",
        "Description": "There are moments when I pay attention to nothing at all",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_2": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 2",
        "Description": "When I am tired of paying attention to someone speaking, my mind empties out",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_3": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 3",
        "Description": "I lose track of my thoughts and can’t remember what I was thinking",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_4": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 4",
        "Description": "When reading a book, I must reread pages because I ended at the end of the page but I do not remember how I got there",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_5": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 5",
        "Description": "There are moments when I am sure that I had a thought, but I am not sure what",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_6": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 6",
        "Description": "There are moments when I know I was thinking of something, but I cannot recover it",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_7": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 7",
        "Description": "During the day, I notice that I am thinking of nothing",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_8": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 8",
        "Description": "When I am calm, it feels like my mind is empty",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_9": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 9",
        "Description": "My mind blanks when I am under pressure",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_10": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 10",
        "Description": "When something bad happens and I need to think of solutions, my mind goes blank",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_11": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 11",
        "Description": "When I am bored, I zone out without thinking of anything",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_12": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 12",
        "Description": "When I am sleepy, I easily forget what I am thinking about",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_13": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 13",
        "Description": "There are moments when it feels like I am not thinking about anything in particular",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_14": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 14",
        "Description": "When people ask me what I am thinking, I respond that I am thinking about nothing",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_15": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 15",
        "Description": "Periods of time can pass when I am not thinking of anything",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_16": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 16",
        "Description": "I notice myself staring at nothing without realizing how long it’s been",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_17": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 17",
        "Description": "I catch myself halfway through an action without knowing how I started it",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
    },
    "SRMBQ_18": {
        "LongName": "Self-reported Mind Blanking Questionnaire item 18",
        "Description": "During the day, I feel like I had brief time-skips, as if I missed the last few seconds",
        "Levels": {
            1: "Almost never",
            2: "Infrequently",
            3: "Sometimes",
            4: "Frequently",
            5: "Almost always"
        }
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
            6: "Almost always"
        }
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
            6: "Almost always"
        }
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
            6: "Almost always"
        }
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
            6: "Almost always"
        }
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
            6: "Almost always"
        }
    },
    "MCQ30_1": {
        "LongName": "Metacognitions Questionnaire 30 item 1",
        "Description": "Worry helps me to avoid problems in the future",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_2": {
        "LongName": "Metacognitions Questionnaire 30 item 2",
        "Description": "My worrying is dangerous for me",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_3": {
        "LongName": "Metacognitions Questionnaire 30 item 3",
        "Description": "I think a lot about my thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_4": {
        "LongName": "Metacognitions Questionnaire 30 item 4",
        "Description": "I could make myself sick with worrying",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_5": {
        "LongName": "Metacognitions Questionnaire 30 item 5",
        "Description": "I am aware of the way my mind works when I am thinking through a problem",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_6": {
        "LongName": "Metacognitions Questionnaire 30 item 6",
        "Description": "If I did not control a worrying thought, and then it happened, it would be my fault",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_7": {
        "LongName": "Metacognitions Questionnaire 30 item 7",
        "Description": "I need to worry in order to remain organized",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_8": {
        "LongName": "Metacognitions Questionnaire 30 item 8",
        "Description": "I have little confidence in my memory for words and names",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_9": {
        "LongName": "Metacognitions Questionnaire 30 item 9",
        "Description": "My worrying thoughts persists, no matter how I try to stop them",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_10": {
        "LongName": "Metacognitions Questionnaire 30 item 10",
        "Description": "Worrying helps me to get things sorted out in my mind",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_11": {
        "LongName": "Metacognitions Questionnaire 30 item 11",
        "Description": "I cannot ignore my worrying thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_12": {
        "LongName": "Metacognitions Questionnaire 30 item 12",
        "Description": "I monitory my thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_13": {
        "LongName": "Metacognitions Questionnaire 30 item 13",
        "Description": "I should be in control of my thoughts all the time",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_14": {
        "LongName": "Metacognitions Questionnaire 30 item 14",
        "Description": "My memory can mislead me at times",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_15": {
        "LongName": "Metacognitions Questionnaire 30 item 15",
        "Description": "My worrying could make me go mad",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_16": {
        "LongName": "Metacognitions Questionnaire 30 item 16",
        "Description": "I am constantly aware of my thinking",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_17": {
        "LongName": "Metacognitions Questionnaire 30 item 17",
        "Description": "I have a poor memory",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_18": {
        "LongName": "Metacognitions Questionnaire 30 item 18",
        "Description": "I pay close attention to thw way my mind works",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_19": {
        "LongName": "Metacognitions Questionnaire 30 item 19",
        "Description": "Worrying helps me cope",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_20": {
        "LongName": "Metacognitions Questionnaire 30 item 20",
        "Description": "Not being able to control my thoughts is a sign of weakness",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_21": {
        "LongName": "Metacognitions Questionnaire 30 item 21",
        "Description": "When I start worrying, I cannot stop",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_22": {
        "LongName": "Metacognitions Questionnaire 30 item 22",
        "Description": " will be punished for not controlling certain thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_23": {
        "LongName": "Metacognitions Questionnaire 30 item 23",
        "Description": "Worrying helps me to solve problems",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_24": {
        "LongName": "Metacognitions Questionnaire 30 item 24",
        "Description": "I have little confidence in my memory for places",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_25": {
        "LongName": "Metacognitions Questionnaire 30 item 25",
        "Description": "It is bad to think certain thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_26": {
        "LongName": "Metacognitions Questionnaire 30 item 26",
        "Description": "I do not trust my memory",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_27": {
        "LongName": "Metacognitions Questionnaire 30 item 27",
        "Description": "If I could not control my thoughts, I would not be able to function",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_28": {
        "LongName": "Metacognitions Questionnaire 30 item 28",
        "Description": "I need to worry, in order to work well",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_29": {
        "LongName": "Metacognitions Questionnaire 30 item 29",
        "Description": "I have little confidence in my memory for my actions",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "MCQ30_30": {
        "LongName": "Metacognitions Questionnaire 30 item 30",
        "Description": "I constantly examine my thoughts",
        "Levels": {
            1: "Do not agree",
            2: "Agree slightly",
            3: "Agree moderately",
            4: "Agree very much"
        }
    },
    "ACS_1": {
        "LongName": "Attentional Control Scale item 1",
        "Description": "It’s very hard for me to concentrate on a difficult task when there are noises around.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_2": {
        "LongName": "Attentional Control Scale item 2",
        "Description": "When I need to concentrate and solve a problem, I have trouble focusing my attention.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_3": {
        "LongName": "Attentional Control Scale item 3",
        "Description": "When I am working hard on something, I still get distracted by events around me.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        },
    },
    "ACS_4": {
        "LongName": "Attentional Control Scale item 4",
        "Description": "My concentration is good even if there is music in the room around me.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_5": {
        "LongName": "Attentional Control Scale item 5",
        "Description": "When concentrating, I can focus my attention so that I become unaware of what’s going on in the room around me.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_6": {
        "LongName": "Attentional Control Scale item 6",
        "Description": "When I am reading or studying, I am easily distracted if there are people talking in the same room.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_7": {
        "LongName": "Attentional Control Scale item 7",
        "Description": "When trying to focus my attention on something, I have difficulty blocking out distracting thoughts.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_8": {
        "LongName": "Attentional Control Scale item 8",
        "Description": "I have a hard time concentrating when I’m excited about something.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_9": {
        "LongName": "Attentional Control Scale item 9",
        "Description": "When concentrating I ignore feelings of hunger or thirst.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_10": {
        "LongName": "Attentional Control Scale item 10",
        "Description": "I can quickly switch from one task to another.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_11": {
        "LongName": "Attentional Control Scale item 11",
        "Description": "It takes me a while to get really involved in a new task.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_12": {
        "LongName": "Attentional Control Scale item 12",
        "Description": "It is difficult for me to coordinate my attention between listening and writing required when taking notes during lectures.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_13": {
        "LongName": "Attentional Control Scale item 13",
        "Description": "I can become interested in a new topic very quickly when I need to.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_14": {
        "LongName": "Attentional Control Scale item 14",
        "Description": "It is easy for me to read or write while I’m also talking on the phone.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_15": {
        "LongName": "Attentional Control Scale item 15",
        "Description": "I have trouble carrying on two conversations at once.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_16": {
        "LongName": "Attentional Control Scale item 16",
        "Description": "I have a hard time coming up with new ideas quickly",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ACS_17": {
        "LongName": "Attentional Control Scale item 17",
        "Description": "After being interrupted or distracted, I can easily shift my attention back to what I was doing before.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_18": {
        "LongName": "Attentional Control Scale item 18",
        "Description": "When a distracting thought comes to mind, it is easy for me to shift my attention away from it.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_19": {
        "LongName": "Attentional Control Scale item 19",
        "Description": "It is easy for me to alternate between two different tasks.",
        "Levels": {
            1: "Almost never",
            2: "Sometimes",
            3: "Often",
            4: "Always"
        }
    },
    "ACS_20": {
        "LongName": "Attentional Control Scale item 20",
        "Description": "It is hard for me to break from one way of thinking about something and look at it from another point of view.",
        "Levels": {
            4: "Almost never",
            3: "Sometimes",
            2: "Often",
            1: "Always"
        }
    },
    "ESS_1": {
        "LongName": "Epworth Sleepiness Scale item 1",
        "Description": "Sitting and reading",
        "Levels": {
            0: "Would never nod off,
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    },
    "ESS_2": {
        "LongName": "Epworth Sleepiness Scale item 2",
        "Description": "Watching TV",
        "Levels": {
            0: "Would never nod off,
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    },
    "ESS_3": {
        "LongName": "Epworth Sleepiness Scale item 3",
        "Description": "Sitting inactive in a public place (e.g., in a meeting, theater, or dinner event)",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    },
    "ESS_4": {
        "LongName": "Epworth Sleepiness Scale item 4",
        "Description": "As a passenger in a car for an hour or more without stopping for a break",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    },
    "ESS_5": {
        "LongName": "Epworth Sleepiness Scale item 5",
        "Description": "Lying down to rest when circumstances permit",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    },
    "ESS_6": {
        "LongName": "Epworth Sleepiness Scale item 6",
        "Description": "Sitting and talking to someone",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    },
    "ESS_7": {
        "LongName": "Epworth Sleepiness Scale item 7",
        "Description": "Sitting quietly after lunch without alcohol",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    },
    "ESS_8": {
        "LongName": "Epworth Sleepiness Scale item 8",
        "Description": "In a car, while stopped for a few minutes in traffic or at a light",
        "Levels": {
            0: "Would never nod off",
            1: "Slight chance of nodding off",
            2: "Moderate chance of nodding off",
            3: "High chance of nodding off"
        }
    }
}

# Save .json
with open(bids_path / "participants.json", "w") as f:
    json.dump(participants_json, f, indent=4)
