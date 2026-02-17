# Section 1: Define keys
# For each sequence, define a key variables (e.g., t1w, dwi etc) and template using the create_key function:
# key = create_key(output_directory_path_and_name).


def create_key(template, outtype=("nii.gz",), annotation_classes=None):
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return (template, outtype, annotation_classes)


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where"""

    # Anatomical
    t1w = create_key("sub-{subject}/anat/sub-{subject}_T1w")

    # Functional - Rest (10min)
    rest = create_key("sub-{subject}/func/sub-{subject}_task-rest_bold")

    # Functional - Task ES (70min)
    task_ES = create_key("sub-{subject}/func/sub-{subject}_task-ES_bold")

    # Field maps
    fmap_magnitude = create_key("sub-{subject}/fmap/sub-{subject}_magnitude")
    fmap_phasediff = create_key("sub-{subject}/fmap/sub-{subject}_phasediff")

    # RevPE
    revpe_pa = create_key("sub-{subject}/fmap/sub-{subject}_dir-PA_epi")
    revpe_ap = create_key("sub-{subject}/fmap/sub-{subject}_dir-AP_epi")

    info = {t1w: [], rest: [], task_ES: [], fmap_magnitude: [], fmap_phasediff: [], revpe_pa: [], revpe_ap: []}

    for s in seqinfo:
        # T1w
        if "t1_mpr_sag_p2_iso" in s.protocol_name:
            info[t1w].append(s.series_id)

        # Rest (10min) 
        elif "StdPE-10min" in s.protocol_name and "_Pha_" not in s.dcm_dir_name:
            info[rest].append(s.series_id)

        # Task ES (70min) 
        elif "StdPE-70min" in s.protocol_name and "_Pha_" not in s.dcm_dir_name:
            info[task_ES].append(s.series_id)

        # GRE Field maps 
        elif "gre_field_mapping" in s.protocol_name:
            if s.dim4 == 2:
                info[fmap_magnitude].append(s.series_id)
            elif s.dim4 == 1:
                info[fmap_phasediff].append(s.series_id)
                
        # # InvPE 
        elif "InvPE" in s.protocol_name and "_Pha_" not in s.dcm_dir_name:
            info[revpe_pa].append(s.series_id)
        elif "StdPE" in s.protocol_name and "_Pha_" not in s.dcm_dir_name and s.dim4 < 10:
            info[revpe_ap].append(s.series_id)


    return info
