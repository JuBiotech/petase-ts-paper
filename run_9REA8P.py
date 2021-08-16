import numpy
from numpy.lib.function_base import _parse_input_dimensions
import pandas
import pathlib
import typing

import retl
import robotools


import cutisplit


DP_RUN = pathlib.Path(r"\\IBT705\DATA\CM\Pahpshmir_BL-sGFP-Cutinase-BSI\9REA8P")


def get_df_calibration():
    df_calibration = cutisplit.read_nitrophenol_calibration(DP_RUN)
    return df_calibration


def get_df_inputs():
    """ Construct a DataFrame of cutinase concentration factors in the "standards"
    """
    # the standards were located in columns [1-3] of the "samples" DWP
    # they were copied into [4-6] within the "samples" DWP
    # and again copied into [7-12] of the assay plate
    plan = robotools.DilutionPlan(
        xmin=0.01, xmax=1,
        R=8, C=3,
        stock=1,
        mode="log",
        vmax=950,
        min_transfer=30
    )

    df_inputs = pandas.DataFrame(
        columns=["input_well", "type", "concentration_factor"]
    ).set_index("input_well")
    for ir, r in enumerate("ABCDEFGH"):
        for ic, c in enumerate([1,2,3,4,5,6]):
            df_inputs.loc[f"{r}{c:02d}", "type"] = f"reference"
            df_inputs.loc[f"{r}{c:02d}", "concentration_factor"] = plan.x[ir, ic % 3]
    return df_inputs


def get_df_cutinase():
    df_cutinase = cutisplit.read_cutinase(DP_RUN)
    return df_cutinase


def get_df_sgfp():
    df_sgfp, t0_delta = cutisplit.read_sgfp(DP_RUN)
    print("!! Overriding t0_delta with 0.25 hours !!")
    t0_delta = 0.25
    return df_sgfp, t0_delta