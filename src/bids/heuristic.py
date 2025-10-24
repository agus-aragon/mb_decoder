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
    fmap_rev = create_key("sub-{subject}/fmap/sub-{subject}_dir-AP_epi")
    fmap_map = create_key("sub-{subject}/fmap/sub-{subject}_magnitude")
    fmap_phase = create_key("sub-{subject}/fmap/sub-{subject}_phasediff")

    info = {t1w: [], rest: [], task_ES: [], fmap_rev: [], fmap_map: [], fmap_phase: []}
   
    ###############################################
    # PILOT
    dummmy_bold_3 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummy3_bold")
    dummmy_bold_6 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummyTR186-2p5mm_bold")
    dummmy_bold_9 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummyTR175-3p0mm_bold")
    dummmy_bold_12 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummyTR110-3p0mm_bold")
    dummmy_bold_15 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummyTR160-3p0mm-2790Hz_bold")
    dummmy_bold_18 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummyTR160-3p0mm-2442Hz_bold")
    dummmy_bold_21 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummy-ege-enc_bold")
    dummmy_bold_24 =  create_key("sub-{subject}/func/sub-{subject}_acq-dummy-TR25-2p0mm_bold")

    ###############################################

    for s in seqinfo:
        if s.protocol_name == 't1_mpr_sag_p2_iso_2_MR':
            info[t1w] = [s.series_id]
        if s.protocol_name == 'Idyn_cmrr_mb2p3_3GEs_2mm_200mm_TR2p5s_31_MR':
            info[rest] = [s.series_id]
        if s.protocol_name == '1dyn_cmrr_mb2p3_3GEs_2mm_200mm_TR2p5s_35_MR':
            info[rest] = [s.series_id]
        if s.protocol_name == 'gre_field_mapping_29_MR':
            info[fmap***] = [s.series_id]
   # gre_field_mapping_ 28_MR    gre_field_mapping_29_MR 
   # gre_field_mapping_ 30_MR
# TODO: physiology
    ###############################################
    # PILOT
    
        if s.protocol_name == 'Agustina_cmrr_mb2ep2d_bold_3_MR':
            info[dummmy_bold_3] == [s.series_id]
        if s.protocol_name == 'Agustina_cmrr_mb2ep2d_1GE_TR1p86s_2p5mm_6_MR':
            info[dummmy_bold_6] == [s.series_id]
        if s.protocol_name == 'Agustina_cmrr_mbZep2d_1GE_TRIp75s_3p0mm_9_MR':
            info[dummmy_bold_9] == [s.series_id]
        if s.protocol_name == 'Agustina_cmrr_mb3ep2d_1GE_TR1p10s_3p0mm_12_MR':
            info[dummmy_bold_12] == [s.series_id]
        if s.protocol_name == 'Agustina_cmrr_mb2p2ep2d_2GEs_TRIp60s_3p0mm_2790Hz_15_MR':
            info[dummmy_bold_15] == [s.series_id]  
        if s.protocol_name == 'Agustina_cmrr_mb2p2ep2d_2GEs_TR1p60s_3p0mm_2442Hz_18_MR':
            info[dummmy_bold_18] == [s.series_id]  
        if s.protocol_name == 'cmrr_mbep2d_bold_mb2_3ge_enc_SBRef_21_MR':
            info[dummmy_bold_21] == [s.series_id]  
        if s.protocol_name == '1dyn_cmrr_mb2p3_3GEs_2mm_200mm_TR2p5s_24_MR':
            info[dummmy_bold_24] == [s.series_id]  


    ###############################################
    