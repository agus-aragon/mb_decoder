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

    info = {t1w: [], rest: [], task_ES: [], fmap_magnitude: [], fmap_phasediff: []}

    for s in seqinfo:
        # T1w
        if s.dcm_dir_name == "t1_mpr_sag_p2_iso_2_MR":
            info[t1w].append(s.series_id)

        # Rest (10min) - series 9, 10, 12
        elif (
            s.dcm_dir_name == "Agustina_StdPE-10min_cmrr_mb2ep2d_1GE_TR1p5s_3p0mm_12_MR"
        ):
            info[rest].append(s.series_id)

        # Task ES (70min) - series 13, 14, 16
        elif (
            s.dcm_dir_name == "Agustina_StdPE-70min_cmrr_mb2ep2d_1GE_TR1p5s_3p0mm_29_MR"
        ):
            info[task_ES].append(s.series_id)

        # Field maps - Series 17 has magnitudes
        elif s.dcm_dir_name == "gre_field_mapping_35_MR":
            if s.image_type[2] == "M":
                if len(info[fmap_magnitude]) == 0:
                    info[fmap_magnitude].append(s.series_id)

        # Field maps - Series 18 has the phasediff
        elif s.dcm_dir_name == "gre_field_mapping_36_MR":
            if s.image_type[2] == "P":
                info[fmap_phasediff].append(s.series_id)

    return info
