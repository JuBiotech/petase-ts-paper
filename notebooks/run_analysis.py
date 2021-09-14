import numpy
from numpy.lib.function_base import _parse_input_dimensions
import pandas
import pathlib
import typing

import retl
import robotools


import cutisplit


class CutisplitAnalysis():
    """
    Convenience class to execute the general functions for cutisplit analysis.
    
    Attributes
    ----------
    dcs_experiment : str
        User-defined name of the DCS experiment (see IBT705)
    
    run_id : str
        ID of the DCS run to be analysed
        
    """
    def __init__(self, dcs_experiment, run_id):
        
        self.run_id = run_id
        self.dcs_experiment = dcs_experiment
        self.DP_RUN = pathlib.Path(rf"\\IBT705\DATA\CM\{self.dcs_experiment}\{self.run_id}")
        pass

    def get_df_calibration(self):
        df_calibration = cutisplit.read_nitrophenol_calibration(self.DP_RUN)
        return df_calibration


    def get_df_inputs(self, standards=False):
        """ Construct a DataFrame of cutinase concentration factors in the "standards"
        """
        # the standards were located in columns [1-3] of the "samples" DWP
        # they were copied into [4-6] within the "samples" DWP
        # and again copied into [7-12] of the assay plate
        if standards:
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
                    df_inputs.loc[f"{r}{c:02d}", "type"] = "reference"
                    df_inputs.loc[f"{r}{c:02d}", "concentration_factor"] = plan.x[ir, ic % 3]
        else:
            mtp_wells = numpy.array([
                f"{l}{n:02d}" 
                for l in "ABCDEFGH" 
                for n in range(1,7)
            ]).reshape(8,6)
            strains = pandas.read_excel(
                f"{self.DP_RUN}\Wells_Assay.xlsx", 
                sheet_name="MTP", 
                index_col=0
            ).values[:,:6]
            df_inputs = pandas.DataFrame(
                        columns=["input_well", "type", "concentration_factor"]
                    ).set_index("input_well")

            for well, strain in zip(mtp_wells.flatten(), strains.flatten()):
                if strain != "water":
                    df_inputs.loc[well, "type"] = strain
                    df_inputs.loc[well, "concentration_factor"] = 1
        return df_inputs


    def get_df_cutinase(self):
        df_cutinase = cutisplit.read_cutinase(self.DP_RUN)
        return df_cutinase


    def get_df_sgfp(self):
        df_sgfp, t0_delta = cutisplit.read_sgfp(self.DP_RUN)
        print("!! Overriding t0_delta with 0.25 hours !!")
        t0_delta = 0.25
        return df_sgfp, t0_delta