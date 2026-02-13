from copy import deepcopy

import mne
import numpy as np


def _bp_filter(raw, params=None, n_jobs=1):
    if params is None:
        params = {}
    lpass = params.get("lpass", 45.0)
    hpass = params.get("hpass", 0.5)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, ecg=False, exclude=[])
    _filter_params = dict(method="fir")
    filter_params = [
        dict(l_freq=hpass, h_freq=None),
        dict(l_freq=None, h_freq=lpass),
    ]

    for fp in filter_params:
        _filter_params2 = deepcopy(_filter_params)
        if fp.get("method") == "fft":
            _filter_params2.pop("iir_params")
        if isinstance(fp, dict):
            _filter_params2.update(fp)
        raw.filter(picks=picks, n_jobs=n_jobs, **_filter_params2)

    notches = [50, 100, 150, 200]
    print("Notch filters at {}".format(notches))
    raw.notch_filter(notches, method="fft", n_jobs=n_jobs)

