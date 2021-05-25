import logging
import typing

import calibr8
import numpy
import pandas
import pymc3
import theano.tensor as tt

from .utils import replicate_wells_from

_log = logging.getLogger(__file__)


class NitrophenolAbsorbanceModel(calibr8.BasePolynomialModelT):
    def __init__(self):
        super().__init__(
            independent_key="4NP_concentration",
            dependent_key="absorbance",
            mu_degree=1,
            scale_degree=1,
        )


def _validate_df_inputs(df_inputs) -> pandas.DataFrame:
    if df_inputs.index.name != "input_well":
        _log.warning(
            "The inputs index should be named 'input_well', with entries corresponding to "
            f"well IDs in the deep well plate. Index name was '{df_inputs.index.name}'."
        )
        df_inputs.index.name = "input_well"
    missing_cols = {"type", "concentration_factor"} - set(df_inputs.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")
    if len(df_inputs) > 48:
        raise ValueError(
            f"Too many input wells. Expected max 48, got {len(df_inputs)}."
        )
    if not numpy.array_equal(df_inputs.index, df_inputs.sort_index().index):
        raise ValueError("The input must be sorted by the 'input_well' index.")
    return df_inputs


def _validate_df_cutinase(df_cutinase) -> pandas.DataFrame:
    actual_levels = [l.name for l in df_cutinase.index.levels]
    expected_levels = ["well", "cycle"]
    if actual_levels != expected_levels:
        raise ValueError(
            f"The cutinase data should be indexed by {expected_levels}, not {actual_levels}."
        )
    actual_columns = set(df_cutinase.columns)
    expected_columns = {"time", "value"}
    if not actual_columns.issubset(expected_columns):
        raise ValueError(
            f"Columns for the cutinase should be {expected_columns}, not {actual_columns}."
        )
    return df_cutinase


def _validate_df_sgfp(df_sgfp: pandas.DataFrame) -> pandas.DataFrame:
    if df_sgfp.index.name != "time_hours":
        raise ValueError(
            f"Expected index to be named 'time_hours' not '{df_sgfp.index.name}'"
        )
    return df_sgfp


def _reshape_cutinase(df_cutinase: pandas.DataFrame, input_wells: typing.List[str]):
    """Creates 2D matrices of time and value from a long-format DataFrame.

    Parameters
    ----------
    df_cutinase : pandas.DataFrame
        Cutinase observations
    input_wells : list of str
        IDs of the input wells to include in the output

    Returns
    -------
    t_obs : numpy.ndarray
        Times of measurements, shaped (n_assay_wells, n_timepoints)
    y_obs : numpy.ndarray
        Values of measurements, shaped (n_assay_wells, n_timepoints)
    """
    assay_wells = replicate_wells_from(input_wells)
    n_assay_wells = len(assay_wells)
    n_time = df_cutinase.reset_index()["cycle"].max()

    # reshape data into two 3D arrays:
    t_obs = numpy.empty((n_assay_wells, n_time))
    y_obs = numpy.empty((n_assay_wells, n_time))
    for w, awell in enumerate(assay_wells):
        t_obs[w, :] = df_cutinase.loc[awell, "time"].values
        y_obs[w, :] = df_cutinase.loc[awell, "value"].values
    return t_obs, y_obs


def _add_or_assert_coords(
    coords: typing.Dict[str, typing.Sequence], pmodel: pymc3.Model
):
    """Ensures that the coords are available in the model."""
    for cname, cvalues in coords.items():
        if cname in pmodel.coords:
            numpy.testing.assert_array_equal(pmodel.coords[cname], cvalues)
        else:
            pmodel.coords[cname] = cvalues


class CombinedModel:
    def __init__(
        self,
        df_inputs: pandas.DataFrame,
        *,
        df_cutinase: typing.Optional[pandas.DataFrame],
        cm_nitrophenol: typing.Optional[calibr8.CalibrationModel],
        df_sgfp: typing.Optional[pandas.DataFrame],
        t0_delta: typing.Optional[float] = 0.25,
        sd_cutinase: float = 0.3,
        sd_sgfp: float = 0.1,
    ):
        """Create a model for the (optionally) combined analysis of split-GFP and cutinase assay.

        Parameters
        ----------
        df_inputs : pandas.DataFrame
            A table that describes the layout of the samples that went into the experiment.
            Index: well ID, corresponding to the well IDs in the DWP
            Columns:
                "type" (str) is the kind of sample, for example "reference", "strain5", etc.
                "concentration_factor" (float) is the concentration in the inpu well. 1 ≙ undiluted
        df_cutinase : pandas.DataFrame
            Observations of the cutinase assay.
            Index: [well, cycle]
            Columns: [time, value]
        cm_nitrophenol : calibr8.CalibrationModel
            The calibration model for 4-nitrophenol absorbance/concentration
        df_sgfp : pandas.DataFrame
            Observations of the split-GFP assay
            Index: time_hours
            Columns: assay well IDs
        t0_delta : float
            Time delay (in hours) between pipetting and first measurement of the split-GFP assay
        sd_cutinase : float
            Approximate pipetting error from input well to cutinase assay well.
            0.1 ≙ 10 % relative error
        sd_sgfp : float
            Approximate pipetting error from input well to split-GFP assay well.
            0.1 ≙ 10 % relative error
        """
        self.df_inputs = _validate_df_inputs(df_inputs.copy())

        if df_cutinase is not None:
            df_cutinase = _validate_df_cutinase(df_cutinase)
        self.df_cutinase = df_cutinase
        self.sd_cutinase = sd_cutinase
        self.cm_nitrophenol = cm_nitrophenol

        if df_sgfp is not None:
            df_sgfp = _validate_df_sgfp(df_sgfp.copy())
        self.df_sgfp = df_sgfp
        self.sd_sgfp = sd_sgfp
        self.t0_delta = t0_delta

        with pymc3.Model() as self.pmodel:
            self._type_indices = self._model_concentrations()
            if df_cutinase is not None:
                self._model_cutinase()
            if df_sgfp is not None:
                self._model_sgfp()
        super().__init__()

    def _model_concentrations(self) -> typing.List:
        df_inputs = self.df_inputs

        # first check that the input dataframe corresponds to the standard format:

        # input checks successful! Can now start building the model...
        # First get a handle on the PyMC3 model
        pmodel = pymc3.modelcontext(None)

        # Now create "coords" that describe the relevant dimensions
        _add_or_assert_coords(
            {
                "type": list(sorted(set(df_inputs.type))),
                "input_well": df_inputs.index.values,
                "assay_well": numpy.array(replicate_wells_from(df_inputs.index)),
            },
            pmodel,
        )

        cf_input = pymc3.Data(
            "cf_input", df_inputs.concentration_factor.values, dims=("input_well",)
        )
        cf_cutinase_assay = pymc3.Lognormal(
            "cf_cutinase_assay",
            mu=tt.log(tt.concatenate([cf_input] * 2)),
            sd=self.sd_cutinase,
            dims=("assay_well",),
        )
        cf_sgfp_assay = pymc3.Lognormal(
            "cf_sgfp_assay",
            mu=tt.log(tt.concatenate([cf_input] * 2)),
            sd=self.sd_cutinase,
            dims=("assay_well",),
        )
        type_indices = 2 * list(map(pmodel.coords["type"].index, df_inputs.type))
        return type_indices

    def _model_cutinase(self):
        t_obs, y_obs = _reshape_cutinase(self.df_cutinase, self.df_inputs.index)

        pmodel = pymc3.modelcontext(None)
        _add_or_assert_coords(
            {
                "cutinase_cycle": numpy.arange(t_obs.shape[-1]),
            },
            pmodel,
        )

        cf_input = pmodel["cf_input"]
        cf_cutinase_assay = pmodel["cf_cutinase_assay"]

        # For the cutinase assay, we introduce the specific enzyme activity in the supernatant [µmol/mL/min].
        # Again, we assume one undiluted activity and calculate the activities in the sample/replicate wells
        # deterministically from the relative concentrations above.
        vmax = pymc3.HalfFlat("vmax", dims=("type",))
        # The inputs are diluted 500x into the assay.
        dilution_factor = pymc3.Data("dilution_factor", 500)
        vmax_assay = pymc3.Deterministic(
            "vmax_assay",
            cf_cutinase_assay * vmax[self._type_indices] / dilution_factor,
            dims=("assay_well",),
        )

        # prediction
        cutinase_time = pymc3.Data(
            "cutinase_time", t_obs, dims=("assay_well", "cutinase_cycle")
        )
        P0 = pymc3.Uniform("P0", -0.5, 1, dims=("assay_well",))
        product_concentration = pymc3.Deterministic(
            "product_concentration",
            P0[:, None] + vmax_assay[:, None] * cutinase_time,
            dims=("assay_well", "cutinase_cycle"),
        )

        # The reaction product (4-nitrophenol) is measured in [mmol/L] = [µmol/mL] via the error model.
        # Our prediction is also in µmol/mL.
        absorbance = pymc3.Data(
            "cutinase_absorbance", y_obs, dims=("assay_well", "cutinase_cycle")
        )
        L_cut = self.cm_nitrophenol.loglikelihood(
            y=absorbance,
            x=product_concentration,
            replicate_id="cutinase_all",
            dependent_key=self.cm_nitrophenol.dependent_key,
        )
        return

    def _model_sgfp(self):
        pmodel = pymc3.modelcontext(None)
        # list of columns in the assay
        assay_column = list(sorted({w[1:] for w in pmodel.coords["assay_well"]}))
        _add_or_assert_coords(
            {
                "sgfp_cycle": numpy.arange(len(self.df_sgfp)),
                "assay_column": numpy.array(assay_column),
            },
            pmodel,
        )

        cf_input = pmodel["cf_input"]
        cf_sgfp_assay = pmodel["cf_sgfp_assay"]

        sgfp_cycle_time = pymc3.Data(
            "sgfp_cycle_time", self.df_sgfp.index.values, dims=("sgfp_cycle",)
        )
        # time lags are modeled by column, because all wells in a column are pipetted simultaneously
        sgfp_time_lag_cols = pymc3.Lognormal(
            "sgfp_time_lag_cols", numpy.log(self.t0_delta), sd=1, dims=("assay_column",)
        )
        sgfp_time_lag_wells = pymc3.Deterministic(
            "sgfp_time_lag_wells",
            tt.stack(
                [
                    sgfp_time_lag_cols[assay_column.index(w[1:])]
                    for w in pmodel.coords["assay_well"]
                ]
            ),
        )

        sgfp_time = pymc3.Deterministic(
            "sgfp_time",
            sgfp_time_lag_wells[:, None] + sgfp_cycle_time[None, :],
            dims=("assay_well", "sgfp_cycle"),
        )

        # fluorescence limits =^= cutinase amount
        # fmax = fluorescence capacity in undiluted supernatant
        fmax = pymc3.Lognormal("fmax", numpy.log(700), sd=1, dims=("type",))
        # fmax_assay = fluorescence capacity in diluted assay well
        fmax_assay = pymc3.Deterministic(
            "fmax_assay", cf_sgfp_assay * fmax[self._type_indices], dims=("assay_well",)
        )

        # saturation curve
        sgfp_ht_assembly = pymc3.HalfNormal("sgfp_ht_assembly", 3)
        tau = sgfp_ht_assembly / numpy.log(2)
        sgfp_assembled_fraction = pymc3.Deterministic(
            "sgfp_assembled_fraction",
            (1 - tt.exp(-sgfp_time / tau)),
            dims=("assay_well", "sgfp_cycle"),
        )

        # exponential decay:
        sgfp_ht_decay = pymc3.HalfNormal("sgfp_ht_decay", 20)
        sgfp_ht_decay_assay = pymc3.Lognormal(
            "sgfp_ht_decay_assay", tt.log(sgfp_ht_decay), sd=0.5, dims=("assay_well",)
        )
        tau_decay = sgfp_ht_decay_assay / numpy.log(2)
        sgfp_decay = pymc3.Deterministic(
            "sgfp_decay",
            (tt.exp(-sgfp_time / tau_decay[:, None])),
            dims=("assay_well", "sgfp_cycle"),
        )

        # the final fluorescence depends on the well-wise fluorescence capacity, aggregration and decay
        sgfp_fluorescence_pred = pymc3.Deterministic(
            "sgfp_fluorescence_pred",
            fmax_assay[:, None] * sgfp_assembled_fraction * sgfp_decay,
            dims=("assay_well", "sgfp_cycle"),
        )

        sgfp_fluorescence_obs = pymc3.Data(
            "sgfp_fluorescence_obs",
            self.df_sgfp[pmodel.coords["assay_well"]].values.T,
            dims=("assay_well", "sgfp_cycle"),
        )
        L_sgfp = pymc3.Normal(
            "L_sgfp",
            mu=sgfp_fluorescence_pred,
            sd=10,
            observed=sgfp_fluorescence_obs,
            dims=("assay_well", "sgfp_cycle"),
        )
        return

    def summary(self):
        for cname, cvals in self.pmodel.coords.items():
            shape = numpy.shape(cvals)
            vals = numpy.asarray(cvals).flatten()
            if numpy.prod(shape) < 5:
                examples = ", ".join([str(v) for v in vals])
            else:
                examples = ", ".join([str(v) for v in vals][:3])
                examples += f", …, {cvals[-1]}"
            print(f"{cname: <20}{shape}\t{examples}")
