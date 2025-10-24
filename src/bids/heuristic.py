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

    for s in seqinfo: