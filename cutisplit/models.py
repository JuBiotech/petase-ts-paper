import logging
import typing

import calibr8
import numpy
import pandas
try:
    import pymc3
except:
    import pymc as pymc3
import aesara.tensor as at


_log = logging.getLogger(__file__)


class NitrophenolAbsorbanceModel(calibr8.BasePolynomialModelT):
    def __init__(self):
        super().__init__(
            independent_key="4NP_concentration",
            dependent_key="absorbance",
            mu_degree=1,
            scale_degree=1,
        )


def _add_or_assert_coords(
    coords: typing.Dict[str, typing.Sequence], pmodel: pymc3.Model
):
    """Ensures that the coords are available in the model."""
    for cname, cvalues in coords.items():
        if cname in pmodel.coords:
            numpy.testing.assert_array_equal(pmodel.coords[cname], cvalues)
        else:
            pmodel.add_coord(name=cname, values=cvalues)

class LongFormModel:
    def __init__(
        self,
        df_inputs: pandas.DataFrame,
        *,
        df_kinetics: typing.Optional[pandas.DataFrame],
        cm_nitrophenol: typing.Optional[calibr8.CalibrationModel],
        sd_cutinase: float = 0.3,
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
        sd_cutinase : float
            Approximate pipetting error from input well to cutinase assay well.
            0.1 ≙ 10 % relative error
        """
        self.df_inputs = df_inputs
        self.df_kinetics = df_kinetics
        self.sd_cutinase = sd_cutinase
        self.cm_nitrophenol = cm_nitrophenol
        self.idx_CIDtoS = None
        self.idx_KIDtoCID = None

        with pymc3.Model() as self.pmodel:
            self._model_concentrations()
            self._model_cutinase()
        super().__init__()

    def _model_concentrations(self):
        df_inputs = self.df_inputs

        # First get a handle on the PyMC3 model
        pmodel = pymc3.modelcontext(None)

        # Now define the most important coords
        strains = list(sorted(set(df_inputs.strain)))
        # The culture ID corresponds to biological replicates
        culture_ids = list(sorted(set(df_inputs.index.to_numpy(dtype=str))))
        # The kinetic ID corresponds to technical replicates
        kinetic_ids = list(sorted(set(self.df_kinetics.index.to_numpy(dtype=str))))
        # The column IDs are all the columns used in the assay microtiter plate
        column_ids = list(sorted(set(self.df_kinetics.assay_column.to_numpy(dtype=int))))

        S = "strain"
        CID = "culture_id"
        KID = "kinetic_id"
        COID = "column_id"

        # The model describes the data in the "long form". It needs some lists for indexing to map
        # between variables of different dimensionality.
        # For example the `idx_CIDtoC` maps each culture id (CID) to the 0-based index of the corresponding condition (C).
        self.idx_CIDtoS = [
            strains.index(row.strain)
            for row in df_inputs.loc[culture_ids].itertuples()
        ]
        assert numpy.shape(self.idx_CIDtoS) == (len(culture_ids),)
        assert max(self.idx_CIDtoS) == len(strains) - 1

        self.idx_KIDtoCID = [
            culture_ids.index(row.culture_id)
            for row in self.df_kinetics.loc[kinetic_ids].itertuples()
        ]
        assert numpy.shape(self.idx_KIDtoCID) == (len(kinetic_ids),), f"{numpy.shape(self.idx_KIDtoCID)}, {len(kinetic_ids)}"
        assert max(self.idx_KIDtoCID) == len(culture_ids) - 1

        self.idx_KIDtoCOID = [
            column_ids.index(row.assay_column)
            for row in self.df_kinetics.loc[kinetic_ids].itertuples()
        ]
        assert numpy.shape(self.idx_KIDtoCOID) == (len(kinetic_ids),), f"{numpy.shape(self.idx_KIDtoCOID)}, {len(kinetic_ids)}"
        assert max(self.idx_KIDtoCOID) == len(column_ids) - 1, f"{max(self.idx_KIDtoCOID)}, {len(column_ids)}"

        # Now create "coords" that describe the relevant dimensions
        _add_or_assert_coords(
            {
                "strain": numpy.array(strains, dtype=str),
                "culture_id": numpy.array(culture_ids, dtype=str),
                "kinetic_id": numpy.array(kinetic_ids, dtype=str),
                "column_id": numpy.array(column_ids, dtype=int)
            },
            pmodel,
        )

        cf_input = pymc3.Data(
            "cf_input", self.df_kinetics.concentration_factor.to_numpy(dtype=float), dims=("kinetic_id",)
        )
        cf_cutinase_assay = pymc3.Lognormal(
            "cf_cutinase_assay",
            mu=at.log(cf_input),
            sd=self.sd_cutinase,
            dims=("kinetic_id",),
        )

        return

    def _model_cutinase(self):
        t_obs = numpy.vstack(self.df_kinetics.sort_index().time)
        y_obs = numpy.vstack(self.df_kinetics.sort_index().value)

        pmodel = pymc3.modelcontext(None)
        _add_or_assert_coords(
            {
                "cutinase_cycle": numpy.arange(t_obs.shape[1]),
            },
            pmodel,
        )

        cf_input = pmodel["cf_input"]
        cf_cutinase_assay = pmodel["cf_cutinase_assay"]

        # For the cutinase assay, we introduce the specific enzyme activity in the supernatant [µmol/mL/min].
        # Again, we assume one undiluted activity and calculate the activities in the sample/replicate wells
        # deterministically from the relative concentrations above.
        k = pymc3.HalfNormal("k", sd=3, dims=("strain",), )
        batch_effect = pymc3.Lognormal(
            "batch_effect",
            mu=0,
            sd=0.3,
            dims=("culture_id")
        )
        k_batch = pymc3.Deterministic(
            "k_batch",
            k[self.idx_CIDtoS] * batch_effect,
            dims=("culture_id")
        )
        # The inputs are diluted 72x into the assay.
        dilution_factor = pymc3.Data("dilution_factor", 72)
        assay_effect = pymc3.Lognormal(
            "assay_effect",
            mu=0,
            sd=0.1,
            dims=("column_id")
        )
        k_assay = pymc3.Deterministic(
            "k_assay",
            cf_cutinase_assay * k_batch[self.idx_KIDtoCID] * assay_effect[self.idx_KIDtoCOID] / dilution_factor,
            dims=("kinetic_id",),
        )

        # prediction
        cutinase_time = pymc3.Data(
            "cutinase_time", t_obs, dims=("kinetic_id", "cutinase_cycle")
        )
        S0 = pymc3.Uniform("S0", 0.5, 0.7) #truth should be 0.662 mM

        product_concentration = pymc3.Deterministic(
            "product_concentration",
            S0 * (1 - at.exp(-k_assay[:, None] * cutinase_time )),
            dims=("kinetic_id", "cutinase_cycle"),
        )
        absorbance_intercept = pymc3.Normal("absorbance_intercept", mu=self.cm_nitrophenol.theta_fitted[0], sd=0.1, dims=("kinetic_id"))

        # The reaction product (4-nitrophenol) is measured in [mmol/L] = [µmol/mL] via the error model.
        # Our prediction is also in µmol/mL.
        absorbance = pymc3.Data(
            "cutinase_absorbance", y_obs , dims=("kinetic_id", "cutinase_cycle")
        )
        L_cut = self.cm_nitrophenol.loglikelihood(
            y=absorbance,
            x=product_concentration,
            name="cutinase_all",
            dims=("kinetic_id", "cutinase_cycle"),
            dependent_key=self.cm_nitrophenol.dependent_key,
            # For unknown reasons the cutinase assay exhibits well-wise shifts in the intercept.
            # Replacing the intercept parameter with a vector-tensor essentially gives calibration
            # models for each well, such that they can have independent intercepts.
            theta=[absorbance_intercept[:, None], *self.cm_nitrophenol.theta_fitted[1:]]
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
