import arviz
import ipywidgets
import numpy
try:
    import pymc3
except:
    import pymc as pymc3
from matplotlib import pyplot

from .utils import replicate_wells_from


def plot_sgfp_fit(
    idata: arviz.InferenceData,
    sample_type: str,
    input_well: str,
):
    """Creates a comprehensive plot of the split-GFP assay kinetics from one input well.

    Parameters
    ----------
    idata : arviz.InferenceData
        Contains the model result and data.
    sample_type : str
        Kind of sample.
    input_well : str
        The ID of the input well in the DWP

    Returns
    -------
    fig : matplotlib.Figure
    axs : array of matplotlib.Axes
    """
    # derived from inputs:
    assay_wells = replicate_wells_from([input_well])

    # local vars to make code more readable:
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    fmax = posterior.fmax.sel(type=sample_type)
    cf_input = idata.constant_data.cf_input.sel(input_well=input_well)

    # figure constraints
    x_min = min(idata.constant_data.sgfp_cycle_time.values)
    x_max = max(idata.constant_data.sgfp_cycle_time.values)
    x_center = (x_min + x_max) / 2

    fig, axs = pyplot.subplots(
        dpi=140,
        figsize=(12, 6),
        nrows=2,
        ncols=2,
        sharey="row",
        sharex="col",
        squeeze=False,
    )

    residual_ptp = 0
    for c, assay_well in enumerate(assay_wells):
        fmax_input = fmax * cf_input
        fmax_assay = posterior.fmax_assay.sel(assay_well=assay_well)
        sgfp_assembled_fraction = posterior.sgfp_assembled_fraction.sel(
            assay_well=assay_well
        )
        f_no_decay = fmax_assay * sgfp_assembled_fraction

        # limit in the input well
        pymc3.gp.util.plot_gp_dist(
            ax=axs[0, c],
            samples=numpy.array([fmax_input.values] * 2).T,
            x=numpy.array([x_min, x_center]),
            samples_alpha=0,
            palette="Greys",
        )
        # limit in the assay well (without decay, t->∞)
        pymc3.gp.util.plot_gp_dist(
            ax=axs[0, c],
            samples=numpy.array([fmax_assay.values] * 2).T,
            x=numpy.array([x_center, x_max]),
            samples_alpha=0,
            palette="Blues",
        )

        # fluorescence curve if there was no decay
        pymc3.gp.util.plot_gp_dist(
            ax=axs[0, c],
            samples=f_no_decay.values,
            x=idata.constant_data.sgfp_cycle_time.values,
            samples_alpha=0,
            palette="Greens",
        )

        # and the actual fluorescence
        pymc3.gp.util.plot_gp_dist(
            ax=axs[0, c],
            samples=posterior.sgfp_fluorescence_pred.sel(
                assay_well=assay_well
            ).values.T,
            x=idata.constant_data.sgfp_cycle_time.values,
            samples_alpha=0,
        )
        axs[0, c].scatter(
            idata.constant_data.sgfp_cycle_time.values,
            idata.constant_data.sgfp_fluorescence_obs.sel(assay_well=assay_well),
            marker="x",
        )

        # residuals for the fluorescence
        samples = posterior.sgfp_fluorescence_pred.sel(assay_well=assay_well)
        median = samples.median("sample").values.T
        data = idata.constant_data.sgfp_fluorescence_obs.sel(assay_well=assay_well)
        rel_residuals = ((data - median) / median * 100).values
        residual_ptp = max(residual_ptp, numpy.ptp(rel_residuals))

        pymc3.gp.util.plot_gp_dist(
            ax=axs[1, c],
            samples=(samples.values.T - median) / median * 100,
            x=idata.constant_data.sgfp_cycle_time.values,
            samples_alpha=0,
        )
        axs[1, c].scatter(
            idata.constant_data.sgfp_cycle_time.values, rel_residuals, marker="x"
        )

        # formatting
        axs[0, c].set_title(f"assay well {assay_well}")
        axs[1, c].set_xlabel("time   [h]")
    axs[0, 0].set_ylim(0)
    axs[0, 0].set_ylabel(
        "fluorescence\n"
        "in input well (grey)\n"
        "in assay well (blue)\n"
        "without decay (green)\n"
        "observed (red)"
    )
    axs[1, 0].set_ylim(-residual_ptp, residual_ptp)
    axs[1, 0].set_ylabel("residual   [%]")
    fig.suptitle(
        f"Model fit to '{sample_type}' replicates from input well {input_well}", y=1.01
    )
    fig.tight_layout()
    return fig, axs


def plot_sgfp_fit_interactive(
    idata: arviz.InferenceData,
):
    def plotter(**kwargs):
        plot_sgfp_fit(**kwargs)
        pyplot.show()

    return ipywidgets.interact(
        plotter,
        idata=ipywidgets.fixed(idata),
        sample_type=idata.posterior.type.values,
        input_well=idata.constant_data.input_well.values,
    )


def plot_concentration_error(idata):
    """Creates a plot of scattered to compare desired and actual (modeled) concentrations."""
    # local vars to make code more readable:
    posterior = idata.posterior.stack(sample=("chain", "draw"))

    fig, axs = pyplot.subplots(
        dpi=140, figsize=(8, 8), ncols=2, nrows=2, sharex="col", squeeze=False
    )

    positions = idata.constant_data.cf_input.values
    positions = list(positions) * 2
    for r, (cf_assay, label) in enumerate(
        [
            (posterior.cf_cutinase_assay, "Cutinase"),
            (posterior.cf_sgfp_assay, "split-GFP"),
        ]
    ):
        samples = list(cf_assay.values)

        left, right = axs[r, :]
        left.violinplot(
            dataset=samples,
            positions=positions,
            widths=0.1,
            showextrema=False,
        )
        right.violinplot(
            dataset=list(map(numpy.log10, samples)),
            positions=list(map(numpy.log10, positions)),
            widths=0.1,
            showextrema=False,
        )
        left.plot([0, 1], [0, 1], color="gray", linestyle="--")
        right.plot([-2, 0], [-2, 0], color="gray", linestyle="--")
        left.set_title(label, x=0.15, y=0.9)
        right.set_title(label, x=0.15, y=0.9)

    for (left, right) in axs:
        left.set_ylabel("actual concentration factor   [-]")
        right.set_ylabel("log10(actual concentration factor)   [-]")
    axs[-1, 0].set_xlabel("desired concentration factor   [-]")
    axs[-1, 1].set_xlabel("log10(desired concentration factor)   [-]")
    fig.tight_layout()
    return fig, axs
