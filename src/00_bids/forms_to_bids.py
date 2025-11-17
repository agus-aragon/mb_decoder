# %%
import numpy as np
import pandas as pd
from pathlib import Path

datapath = Path("/Users/agusaragon/Downloads/")

# %%
forms = pd.read_excel(datapath / "responses_form.xlsx")

#TODO: EHI missing 1 line
# %%

new_columns_names = [
    "id",
    "start_response",
    "submit_response",
    "email_default",
    "name_default",
    "language_default",
    "language_chosen",
    "consent",
    "consent_incidential_finding",
    "name_chosen",
    "email_chosen",
    "gender",
    "mri_contraindication",
    "psychiatric_disorder",
    "ehi1_writing",
    "ehi2_drawing",
    "ehi3_throwing",
    "ehi4_usingascissors",
    "ehi5_usingatoothbrush",
    "ehi6_usingaknife",
    "ehi7_usingaspoon",
    "ehi8_usingabroom",
    "ehi9_strikingmatch",
    "ehi10_openingabox",
    "age_criteria",
    "ad1_problemswithjudgment",
    "ad2_lessinterestinhobbies",
    "ad3_repeatsthesamethings",
    "ad4_troublelearning",
    "ad5_forgetscorrectmonth",
    "ad6_troublehandlingfinancial",
    "ad7_dailyproblemswiththinking",
    "age",
    "mbq1_icannotrememberwhatiwasthinking",
    "mbq2_mymindgoescompletelyblank",
    "mbq3_ijustspaceoutwithoutthinking",
    "mbq4_ifindmyselfnotknowingwhatiwasdoing",
    "mbq5_momentswhenmymindemptiesout",
    "mcq1_worryhelpsmeavoidproblems",
    "mcq2_myworryingisdangerousforme",
    "mcq3_ithinkalotaboutmythoughts",
    "mcq4_icouldmakemyselfsickwithworrying",
    "mcq5_iamawareofthewaymymindworks",
    "mcq6_ifididnotcontrolaworryingthought",
    "mcq7_ineedtoworryinordertoremainorganized",
    "mcq8_ihavelittleconfidenceinmymemory",
    "mcq9_myworryingthoughtspersists",
    "mcq10_worryinghelpsme",
    "mcq11_icannotignoremyworryingthoughts",
    "mcq12_imonitormythoughts",
    "mcq13_ishouldbeincontrolofmythoughts",
    "mcq14_mymemorycanmisleadme",
    "mcq15_myworryingcoulmakemegomad",
    "mcq16_iamconstantlyawareofmythinking",
    "mcq17_ihaveapoormemory",
    "mcq18_ipaycloseattentiontothewaymymindworks",
    "mcq19_worryinghelpsmecope",
    "mcq20_notbeingabletocontrolmythoughtsisasignofweakness",
    "mcq21_whenistartworryingicannotstop",
    "mcq22_iwillbepunishedfornotcontrollingcertainthoughts",
    "mcq23_worryinghelpsmetosolveproblems",
    "mcq24_ihavelittleconfidenceinmymemoryforplaces",
    "mcq25_itisbadtothinkcertainthoughts",
    "mcq26_idonottrustmymemory",
    "mcq27_ificouldnotcontrolmythoughtsiwouldnotbeabletofunction",
    "mcq28_ineedtoworryinordertoworkwell",
    "mcq29_ihavelittleconfidenceinmymemoryforactions",
    "mcq30_iconstantly examine my thoughts",
    "acs1_itisveryhardformetoconcentrateonadifficulttask",
    "acs2_whenineedtoconcentrateandsolveaproblemihavetroublefocusing",
    "acs3_wheniamworkinghardonsomethingistillgetdistracted",
    "acs4_myconcentrationisgoodevenifthereismusic",
    "acs5_whenconcentratingicanfocusmyattention",
    "acs6_wheniamreadingorstudyingiameasilydistracted",
    "acs7_whentryingtofocusmyattentiononsomethingihavedifficulty",
    "acs8_ihaveahardtimeconcentratingwheniamexcited",
    "acs9_whenconcentratingiignorefeelingsofhunger",
    "acs10_icanquicklyswitchfromonetasktoanother",
    "acs11_ittakesmeawhiletogetreallyinvolvedinanewtask",
    "acs12_itisdifficultformetocoordinatemyattention",
    "acs13_icanbecomeinterestedinanewtopicveryquick",
    "acs14_itiseasyformetoreadorwritewhileIamalsotalking",
    "acs15_ihavetroublecarryingontwoconversationsatonce",
    "acs16_ihaveahardtimecomingupwithnewideasquick",
    "acs17_afterbeinginterruptedordistractedicaneasilyshift",
    "acs18_whenadistractingthoughtcomestominditiseasyformetoshift",
    "acs19_itiseasyformetoalternatebetweentwodiffer",
    "acs20_itishardformetobreakfromonewayofthinking",
    "mri_availability",
    "mri_availability_details",
    "comment",
    ### FRENCH ###
    "consent",
    "consent_incidential_finding",
    "consent2",
    "consent_incidential_finding2",
    "name_chosen",
    "email_chosen",
    "gender",
    "mri_contraindication",
    "psychiatric_disorder",
    "ehi1_writing",
    "ehi2_drawing",
    "ehi3_throwing",
    "ehi4_usingascissors",
    "ehi5_usingatoothbrush",
    "ehi6_usingaknife",
    "ehi7_usingaspoon",
    "ehi8_usingabroom",
    "ehi9_strikingmatch",
    "ehi10_openingabox",
    "age_criteria",
    "ad1_problemswithjudgment",
    "ad2_lessinterestinhobbies",
    "ad3_repeatsthesamethings",
    "ad4_troublelearning",
    "ad5_forgetscorrectmonth",
    "ad6_troublehandlingfinancial",
    "ad7_dailyproblemswiththinking",
    "age",
    "mbq1_icannotrememberwhatiwasthinking",
    "mbq2_mymindgoescompletelyblank",
    "mbq3_ijustspaceoutwithoutthinking",
    "mbq4_ifindmyselfnotknowingwhatiwasdoing",
    "mbq5_momentswhenmymindemptiesout",
    "mcq1_worryhelpsmeavoidproblems",
    "mcq2_myworryingisdangerousforme",
    "mcq3_ithinkalotaboutmythoughts",
    "mcq4_icouldmakemyselfsickwithworrying",
    "mcq5_iamawareofthewaymymindworks",
    "mcq6_ifididnotcontrolaworryingthought",
    "mcq7_ineedtoworryinordertoremainorganized",
    "mcq8_ihavelittleconfidenceinmymemory",
    "mcq9_myworryingthoughtspersists",
    "mcq10_worryinghelpsme",
    "mcq11_icannotignoremyworryingthoughts",
    "mcq12_imonitormythoughts",
    "mcq13_ishouldbeincontrolofmythoughts",
    "mcq14_mymemorycanmisleadme",
    "mcq15_myworryingcoulmakemegomad",
    "mcq16_iamconstantlyawareofmythinking",
    "mcq17_ihaveapoormemory",
    "mcq18_ipaycloseattentiontothewaymymindworks",
    "mcq19_worryinghelpsmecope",
    "mcq20_notbeingabletocontrolmythoughtsisasignofweakness",
    "mcq21_whenistartworryingicannotstop",
    "mcq22_iwillbepunishedfornotcontrollingcertainthoughts",
    "mcq23_worryinghelpsmetosolveproblems",
    "mcq24_ihavelittleconfidenceinmymemoryforplaces",
    "mcq25_itisbadtothinkcertainthoughts",
    "mcq26_idonottrustmymemory",
    "mcq27_ificouldnotcontrolmythoughtsiwouldnotbeabletofunction",
    "mcq28_ineedtoworryinordertoworkwell",
    "mcq29_ihavelittleconfidenceinmymemoryforactions",
    "mcq30_iconstantly examine my thoughts",
    "acs1_itisveryhardformetoconcentrateonadifficulttask",
    "acs2_whenineedtoconcentrateandsolveaproblemihavetroublefocusing",
    "acs3_wheniamworkinghardonsomethingistillgetdistracted",
    "acs4_myconcentrationisgoodevenifthereismusic",
    "acs5_whenconcentratingicanfocusmyattention",
    "acs6_wheniamreadingorstudyingiameasilydistracted",
    "acs7_whentryingtofocusmyattentiononsomethingihavedifficulty",
    "acs8_ihaveahardtimeconcentratingwheniamexcited",
    "acs9_whenconcentratingiignorefeelingsofhunger",
    "acs10_icanquicklyswitchfromonetasktoanother",
    "acs11_ittakesmeawhiletogetreallyinvolvedinanewtask",
    "acs12_itisdifficultformetocoordinatemyattention",
    "acs13_icanbecomeinterestedinanewtopicveryquick",
    "acs14_itiseasyformetoreadorwritewhileIamalsotalking",
    "acs15_ihavetroublecarryingontwoconversationsatonce",
    "acs16_ihaveahardtimecomingupwithnewideasquick",
    "acs17_afterbeinginterruptedordistractedicaneasilyshift",
    "acs18_whenadistractingthoughtcomestominditiseasyformetoshift",
    "mri_availability",  # When converting forms to excel this questions flip order (they were administered correctly)
    "mri_availability_details",
    "acs19_itiseasyformetoalternatebetweentwodiffer",
    "acs20_itishardformetobreakfromonewayofthinking",
    "comment",
]
column_mapping = dict(
    zip(
        forms.columns,  # current names
        new_columns_names,  # your new names
    )
)

forms = forms.rename(columns=column_mapping)

# %%
# Preview the mapping
print("Column mapping preview:")
for old, new in list(column_mapping.items())[:5]:
    print(f"  '{old}' → '{new}'")
print(f"  ... ({len(column_mapping)} total)")

# Show any that might be problematic
if len(forms.columns) != len(new_columns_names):
    print("\n⚠️ Length mismatch - NOT renaming!")
else:
    forms = forms.rename(columns=column_mapping)
    print("✓ Columns renamed successfully!")

    # %%%
# Check if lengths match first
if len(forms.columns) != len(new_columns_names):
    print(f"WARNING: Column count mismatch!")
    print(f"Excel has {len(forms.columns)} columns")
    print(f"New names has {len(new_columns_names)} names")
    print(f"Difference: {len(forms.columns) - len(new_columns_names)}")
else:
    print("✓ Column counts match!")

# Create the mapping
column_mapping = dict(zip(forms.columns, new_columns_names))

# Rename
forms = forms.rename(columns=column_mapping)

# Verify it worked
print(f"\nFirst few columns: {forms.columns[:5].tolist()}")

# TODO: bids format
