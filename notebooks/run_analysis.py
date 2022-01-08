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
        self.DP_RUN = pathlib.Path(rf"C:\Users\helleckes\TiGr_PETases\{self.dcs_experiment}\{self.run_id}")
        pass

    def get_df_calibration(self, repetition=None):
        df_calibration = cutisplit.read_nitrophenol_calibration(self.DP_RUN, repetition=repetition)
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
                columns=["input_well", "strain", "concentration_factor"]
            ).set_index("input_well")
            for ir, r in enumerate("ABCDEFGH"):
                for ic, c in enumerate([1,2,3,4,5,6]):
                    df_inputs.loc[f"{r}{c:02d}", "strain"] = "reference"
                    df_inputs.loc[f"{r}{c:02d}", "concentration_factor"] = plan.x[ir, ic % 3]
        else:
            fp_wells = numpy.array([
                f"{l}{n:02d}" 
                for l in "ABCDEF" 
                for n in range(1,9)
            ]).reshape(6,8)
            strains = pandas.read_excel(
                f"{self.DP_RUN}\Wells_Assay.xlsx", 
                sheet_name="FP", 
                index_col=0
            ).values
            df_inputs = pandas.DataFrame(
                        columns=["input_well", "strain_rep", "concentration_factor"]
                    ).set_index("input_well")

            for well, strain in zip(fp_wells.flatten(), strains.flatten()):
                if strain != "water" and not "PC" in strain:
                    assert "_" in strain, f"Strain name was '{strain}' in Wells_Assay.xlsx, but should contain '_1' etc. to indicate replicates"
                    df_inputs.loc[well, "strain_rep"] = strain
                    df_inputs.loc[well, "concentration_factor"] = 1
        return df_inputs

    def get_assay_wells_for_strain(self, strain_rep):
        df_assay = pandas.read_excel(
            f"{self.DP_RUN}\Wells_Assay.xlsx", 
            sheet_name="MTP", 
            index_col=0
        )
        df_wells = df_assay.where(df_assay == strain_rep).dropna(how="all").dropna(axis=1)
        row = df_wells.index.values[0]
        assay_wells = [
            f"{row}{number}"
            for number in df_wells
        ]
        return assay_wells


    def get_df_cutinase(self, repetition=None):
        df_cutinase = cutisplit.read_cutinase(self.DP_RUN, repetition=repetition)
        return df_cutinase


    def get_df_sgfp(self):
        df_sgfp, t0_delta = cutisplit.read_sgfp(self.DP_RUN)
        print("!! Overriding t0_delta with 0.25 hours !!")
        t0_delta = 0.25
        return df_sgfp, t0_delta


def read_repetition(run_id, repetition, *, dcs_experiment="Pahpshmir_MTP-Screening-PETase"):
    analyser = CutisplitAnalysis(dcs_experiment=dcs_experiment, run_id=run_id)
    df_inputs = analyser.get_df_inputs().dropna().drop(columns=["concentration_factor"])
    df_inputs.reset_index(inplace=True)
    df_inputs.index = pandas.Index([f"{run_id}_{fp_well}" for fp_well in df_inputs.input_well], name="culture_id")
    df_inputs.rename(columns={"input_well": "fp_well"}, inplace=True)
    df_cutinase = analyser.get_df_cutinase(repetition=repetition)
    df_kinetics = pandas.DataFrame(columns=["culture_id", "assay_well", "assay_column", "time", "value", "concentration_factor"])
    shortened_strains = []
    for row in df_inputs.itertuples():
        culture_ID = row.Index
        strain_rep = row.strain_rep
        shortened_strains.append(strain_rep.split("_")[0])
        assay_wells = analyser.get_assay_wells_for_strain(strain_rep)
        for assay_well in assay_wells:
            if run_id == "C3C1XZ":
                df_kinetics.loc[f"{run_id}_{repetition}_{assay_well}"] = (
                    str(culture_ID), 
                    str(assay_well),
                    int(assay_well[-2:]),
                    df_cutinase.loc[assay_well, "time"].to_numpy(dtype=float),
                    df_cutinase.loc[assay_well, "value"].to_numpy(dtype=float),
                    0.5 
                )
            else:
                df_kinetics.loc[f"{run_id}_{repetition}_{assay_well}"] = (
                    str(culture_ID), 
                    str(assay_well),
                    int(assay_well[-2:]),
                    df_cutinase.loc[assay_well, "time"].to_numpy(dtype=float),
                    df_cutinase.loc[assay_well, "value"].to_numpy(dtype=float),
                    1 
                )
    df_inputs["strain"] = shortened_strains
    df_inputs = df_inputs.drop(columns=["strain_rep"])
    return df_inputs, df_kinetics


def read_rounds(run_ids, dcs_experiment="TiGr_PETase_Screening"):
    dfs_kinetics = []
    dfs_inputs = []
    for run_id in run_ids:
        for i in range(1,50):
            try:
                dfi, dfk = read_repetition(run_id, i, dcs_experiment=dcs_experiment)
                dfi = [dfi]
                dfk = [dfk]
                dfs_kinetics += dfk
            except FileNotFoundError:
                pass
            if i == 1:
                dfs_inputs += dfi
    df_kinetics = pandas.concat(dfs_kinetics)
    df_inputs = pandas.concat(dfs_inputs)
    return df_inputs, df_kinetics