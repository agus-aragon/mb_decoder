"""Provide implementation for read-and-transform function."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import numpy as np
import pandas as pd
import structlog

from junifer.typing import StorageLike
from junifer.utils import raise_error, warn_with_log

__all__ = ["read_transform"]

_log = structlog.get_logger("junifer")
logger = _log.bind(pkg="onthefly")


def read_transform(
    storage: StorageLike,
    transform: str,
    feature_name: str | None = None,
    feature_md5: str | None = None,
    nan_policy: str | None = "bypass",
    transform_args: tuple | None = None,
    transform_kw_args: dict | None = None,
    preprocess_kw_args: dict | None = None,
) -> pd.DataFrame:
    """Read stored feature and transform to specific statistical output.

    Parameters
    ----------
    storage : storage-like
        The storage class, for example, SQLiteFeatureStorage.
    transform : str
        The kind of transform formatted as ``<package>_<function>``,
        for example, ``bctpy_degrees_und``.
    feature_name : str, optional
        Name of the feature to read (default None).
    feature_md5 : str, optional
        MD5 hash of the feature to read (default None).
    nan_policy : str, optional
        The policy to handle NaN values (default "ignore").
        Options are:

        * "bypass": Do nothing and pass NaN values to the transform function.
        * "drop_element": Drop (skip) elements with NaN values.
        * "drop_rows": Drop (skip) rows with NaN values.
        * "drop_columns": Drop (skip) columns with NaN values.
        * "drop_symmetric": Drop (skip) symmetric pairs with NaN values.

    transform_args : tuple, optional
        The positional arguments for the callable of ``transform``
        (default None).
    transform_kw_args : dict, optional
        The keyword arguments for the callable of ``transform``
        (default None). ``bct.rich_club_wu`` requires ``klevel`` here.
    preprocess_kw_args : dict, optional
        The keyword arguments for the preprocessing step (default None).
        Only ``p`` (proportion of strongest edges kept, default 0.2) is
        supported, and only used by thresholded transforms.

    Returns
    -------
    pandas.DataFrame
        The transformed feature as a dataframe. One row per element (and
        timepoint for ``timeseries_2d``), columns depend on the function.

    Raises
    ------
    ValueError
        If ``nan_policy`` is invalid,
        if *package* is invalid,
        if ``preprocess_kw_args`` has unknown keys, or
        if no output was computed (all elements skipped).
    RuntimeError
        If *package* is ``bctpy`` and stored data kind is not ``matrix`` or
        ``timeseries_2d``.
    ImportError
        If ``bctpy`` cannot be imported.
    AttributeError
        If *function* to be invoked in invalid.

    Notes
    -----
    This function has been only tested for:

    * ``bct.degrees_und``
    * ``bct.strengths_und``
    * ``bct.clustering_coef_wu``
    * ``bct.eigenvector_centrality_und``

    And partially tested for:

    * ``bct.strengths_und_sign``
    * ``bct.community_louvain``
    * ``bct.efficiency_wei``
    * ``bct.betweenness_wei``
    * ``bct.edge_betweenness_wei``
    * ``bct.distance_wei``
    * ``bct.assortativity_wei``
    * ``bct.transitivity_wu``
    * ``bct.rich_club_wu``

    Using other functions may fail and require tweaking.

    """
    # Set default values for args and kwargs
    transform_args = transform_args or ()
    transform_kw_args = transform_kw_args or {}
    preprocess_kw_args = preprocess_kw_args or {}

    # Only 'p' is supported for preprocess_kw_args
    unknown_keys = set(preprocess_kw_args) - {"p"}
    if unknown_keys:
        raise_error(
            f"Unknown preprocess_kw_args: {sorted(unknown_keys)}. "
            "Only 'p' is supported.",
            klass=ValueError,
        )

    # Define valid nan_policy options
    if nan_policy not in [
        "bypass",
        "drop_element",
        "drop_rows",
        "drop_columns",
        "drop_symmetric",
    ]:
        raise_error(
            f"Unknown nan_policy: {nan_policy}",
            klass=ValueError,
        )

    # Read storage
    stored_data = storage.read(
        feature_name=feature_name, feature_md5=feature_md5
    )  # type: ignore
    # Retrieve package and function
    package, func_str = transform.split("_", 1)
    # Condition for package
    if package == "bctpy":
        # Check that "matrix" or "timeseries_2d" is the feature data kind
        if stored_data["kind"] not in (
            "matrix",
            "timeseries_2d",
        ):  #### My edits
            raise_error(
                msg=(
                    f"'{stored_data['kind']}' is not valid data kind for "
                    f"'{package}'"
                ),
                klass=RuntimeError,
            )

        # The dropping policies that change the matrix size would break
        # the output columns (N headers) and the graph functions
        if stored_data["kind"] == "timeseries_2d" and nan_policy in (
            "drop_rows",
            "drop_columns",
            "drop_symmetric",
        ):
            raise_error(
                f"nan_policy '{nan_policy}' is not supported for timeseries_2d; "
                "use 'bypass' or 'drop_element'.",
                klass=ValueError,
            )

        # Check bctpy import
        try:
            import bct
        except ImportError as err:  # pragma: no cover
            raise_error(msg=str(err), klass=ImportError)

        # Warning about function usage
        if func_str not in [
            "strengths_und_sign",
            "community_louvain",
            "degrees_und",
            "efficiency_wei",
            "strengths_und",
            "clustering_coef_wu",
            "eigenvector_centrality_und",
            "betweenness_wei",
            "edge_betweenness_wei",
            "distance_wei",
            "assortativity_wei",
            "transitivity_wu",
            "rich_club_wu",
        ]:
            warn_with_log(
                f"You are about to use '{package}.{func_str}' which has not "
                "been tested to run. In case it fails, you will need to tweak"
                " the code yourself."
            )
        # Defaults for community_louvain, user values win (**kwargs last)
        if func_str == "community_louvain":
            transform_kw_args = {
                "B": "negative_sym",
                "seed": 0,
                **transform_kw_args,
            }
            logger.info(
                f"'{package}.{func_str}' uses B={transform_kw_args['B']!r} "
                f"and seed={transform_kw_args['seed']!r}. Override them via "
                "transform_kw_args."
            )
        # rich_club_wu returns a vector of length klevel. Without a
        # fixed klevel, its length (max degree) changes per timepoint and
        # rows cannot be stacked. For "matrix" klevel is optional.
        if (
            stored_data["kind"] == "timeseries_2d"
            and func_str == "rich_club_wu"
            and "klevel" not in transform_kw_args
        ):
            raise_error(
                "'rich_club_wu' needs a fixed 'klevel' in transform_kw_args "
                "so that every row has the same number of columns.",
                klass=ValueError,
            )

        # Preprocessing functions for specific transforms
        def _clean(W, p=None):
            """Zero the diagonal (IPC diagonal is 1)."""
            W = np.array(W, dtype=float)
            np.fill_diagonal(W, 0.0)
            return W

        def _abs(W, p=None):
            """Return absolute value of |W| (weighted, non-negative)."""
            return np.abs(_clean(W))

        def _thres(W, p=0.2):
            """Keep the strongest proportion p of |edges| (weighted, non-negative)."""
            return bct.threshold_proportional(_abs(W), p)

        def _lengths(W, p=0.2):
            """|W| -> proportional threshold -> connection lengths (1 / w)."""
            return bct.weight_conversion(_thres(W, p), "lengths")

        preprocessing = {
            # --- signed ---
            "strengths_und_sign": _clean,
            "community_louvain": _clean,
            # --- weighted, non-negative: abs ---
            "strengths_und": _abs,
            "clustering_coef_wu": _abs,
            "eigenvector_centrality_und": _abs,
            "assortativity_wei": _abs,  # flag=0 (default) is undirected
            # --- thresholded ---
            "degrees_und": _thres,
            "efficiency_wei": _thres,
            "rich_club_wu": _thres,  # thresholded to avoid degenerate rich club coefficient
            "transitivity_wu": _thres,  # thresholded to avoid mean FC
            # --- length-based: not weights ---
            "betweenness_wei": _lengths,
            "edge_betweenness_wei": _lengths,
            "distance_wei": _lengths,
        }

        # Check if the transform function is thresholded and get the proportion p
        thresholded = {
            k for k, f in preprocessing.items() if f in (_thres, _lengths)
        }
        p = None
        if func_str in thresholded:
            p = preprocess_kw_args.get("p", 0.2)
            if not 0 < p <= 1:
                raise_error(f"p must be in (0, 1], got {p}", klass=ValueError)
            logger.info(f"'{func_str}' uses proportional threshold p={p}")
        elif "p" in preprocess_kw_args:
            warn_with_log(f"'{func_str}' does not threshold; 'p' is ignored")

        # Postprocessing for reshaping the output and defining column names
        headers = list(stored_data["row_headers"])
        N = len(headers)
        # every post-processing function receives the raw output `o` of
        # the bct function for one matrix and must return a 1D array.
        postprocessing = {
            "strengths_und_sign": lambda o: np.concatenate(
                [o[0], o[1]]
            ),  # Spos, Sneg
            "community_louvain": lambda o: np.array(
                [o[1], len(np.unique(o[0]))]
            ),  # community labels are arbitrary and not comparable across t
            "edge_betweenness_wei": lambda o: (
                o[1] / ((N - 1) * (N - 2))
            ),  # nodal BC
            "betweenness_wei": lambda o: o / ((N - 1) * (N - 2)),
            "distance_wei": lambda o: np.array(
                [o[0][~np.eye(N, dtype=bool) & np.isfinite(o[0])].mean()]
            ),
            "efficiency_wei": np.atleast_1d,
            "assortativity_wei": np.atleast_1d,
            "transitivity_wu": np.atleast_1d,
        }
        columns = {
            "strengths_und_sign": [f"pos_{h}" for h in headers]
            + [f"neg_{h}" for h in headers],
            "community_louvain": ["Q", "n_modules"],
            "distance_wei": ["mean_path_length"],
            "efficiency_wei": ["efficiency_wei"],
            "assortativity_wei": ["assortativity_wei"],
            "transitivity_wu": ["transitivity_wu"],
        }
        if func_str == "rich_club_wu" and "klevel" in transform_kw_args:
            columns["rich_club_wu"] = [
                f"k={k}" for k in range(1, transform_kw_args["klevel"] + 1)
            ]
        try:
            func = getattr(bct, func_str)
        except AttributeError as err:
            raise_error(msg=str(err), klass=AttributeError)

        # Apply function and store element-wise
        output_list = []
        element_list = []
        logger.debug(
            f"Computing '{package}.{func_str}' for feature "
            f"{feature_name or feature_md5} ..."
        )
        for i_element, element in enumerate(
            stored_data["element"]
        ):  #### My edits (from here)
            if stored_data["kind"] == "matrix":
                slices = [(None, stored_data["data"][:, :, i_element])]
            elif stored_data["kind"] == "timeseries_2d":
                elem_data = stored_data["data"][i_element]
                slices = list(enumerate(elem_data))

            for t, t_data in slices:
                label = (
                    element if t is None else {**element, "timepoint": t}
                )  #### My edits (until here)
                has_nan = np.isnan(np.min(t_data))
                if nan_policy == "drop_element" and has_nan:
                    logger.debug(
                        f"Skipping element {element} due to NaN values ..."
                    )
                    continue
                # Maintain original junifer code for matrix (for timeseries
                # dimensions need to be the same across all rows)
                if stored_data["kind"] == "matrix" and has_nan:
                    if nan_policy == "drop_rows" and has_nan:
                        logger.debug(
                            f"Skipping rows with NaN values in element {element} ..."
                        )
                        t_data = t_data[~np.isnan(t_data).any(axis=1)]
                    elif nan_policy == "drop_columns" and has_nan:
                        logger.debug(
                            f"Skipping columns with NaN values in element {element} "
                            "..."
                        )
                        t_data = t_data[:, ~np.isnan(t_data).any(axis=0)]
                    elif nan_policy == "drop_symmetric":
                        logger.debug(
                            f"Skipping pairs of rows/columns with NaN values in "
                            f"element {element}..."
                        )
                        good_rows = ~np.isnan(t_data).any(axis=1)
                        good_columns = ~np.isnan(t_data).any(axis=0)
                        good_idx = np.logical_and(good_rows, good_columns)
                        t_data = t_data[good_idx][:, good_idx]

                prep = preprocessing.get(
                    func_str, lambda W, p=None: W
                )  #### My edits
                t_data = prep(t_data, **preprocess_kw_args)
                output = func(
                    t_data,
                    *transform_args,
                    **transform_kw_args,
                )
                output = postprocessing.get(func_str, lambda o: o)(output)
                output_list.append(output)
                element_list.append(label)  #### My edits

        # Check if any output was computed
        if not output_list:
            raise_error(
                "Nothing was computed (all elements skipped?).",
                klass=ValueError,
            )

        # rich_club_wu for "matrix"
        if func_str == "rich_club_wu" and "klevel" not in transform_kw_args:
            max_k = max(len(o) for o in output_list)
            output_list = [
                np.pad(
                    np.asarray(o, dtype=float),
                    (0, max_k - len(o)),
                    constant_values=np.nan,
                )
                for o in output_list
            ]
            columns["rich_club_wu"] = [f"k={k}" for k in range(1, max_k + 1)]


        # Create dataframe for index
        idx_df = pd.DataFrame(data=element_list)
        # Create multiindex from dataframe
        logger.debug(
            "Generating pandas.MultiIndex for feature "
            f"{feature_name or feature_md5} ..."
        )
        data_idx = pd.MultiIndex.from_frame(df=idx_df)

        # Create dataframe
        df = pd.DataFrame(
            data=output_list,
            index=data_idx,
            columns=columns.get(func_str, headers),
        )
        return df
    else:
        raise_error(f"Unknown package: {package}")
