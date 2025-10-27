# Section 1: Define keys
# For each sequence, define a key variables (e.g., t1w, dwi etc) and template using the create_key function:
# key = create_key(output_directory_path_and_name).
def create_key(template, outtype=("nii.gz",), annotation_classes=None):
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return (template, outtype, annotation_classes)


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    task: task id
    """
    t1w = create_key("sub-{subject}/anat/sub-{subject}_T1w")
    rest = create_key("sub-{subject}/func/sub-{subject}_task-rest_bold")
    task_ES = create_key("sub-{subject}/func/sub-{subject}_task-ES_bold")
    fmap_magnitude = create_key("sub-{subject}/fmap/sub-{subject}_magnitude")
    fmap_phasediff = create_key("sub-{subject}/fmap/sub-{subject}_phasediff")

    info = {
        t1w: [],
        rest: [],
        task_ES: [],
        fmap_magnitude: [],
        fmap_phasediff: [],
    }
    
    for s in seqinfo:
        # Skip phase and physio files
        if "Pha_" in s.dcm_dir_name or "PhysioLog" in s.dcm_dir_name:
            continue
            
        # T1w
        if "t1_mpr" in s.dcm_dir_name:
            info[t1w].append(s.series_id)
        
        # Rest (series 31)
        elif s.dcm_dir_name == "1dyn_cmrr_mb2p3_3GEs_2mm_200mm_TR2p5s_31_MR":
            info[rest].append(s.series_id)
        
        # ES task (series 35)
        elif s.dcm_dir_name == "1dyn_cmrr_mb2p3_3GEs_2mm_200mm_TR2p5s_35_MR":
            info[task_ES].append(s.series_id)
        
        # Field maps
        elif "gre_field_mapping" in s.dcm_dir_name:
            if s.image_type[2] == "M":
                info[fmap_magnitude].append(s.series_id)
            elif s.image_type[2] == "P":
                info[fmap_phasediff].append(s.series_id)

    return info
