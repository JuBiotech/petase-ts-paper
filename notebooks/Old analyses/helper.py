import datetime
import numpy
import matplotlib
from matplotlib import cm, pyplot
import pandas
import pathlib
import retl

import pymc3
import theano.tensor as tt


def replicate_wells(input_wells):
    """ Creates a list of well IDs containing the input well IDs
    and corresponding well ID shifted by 6 columns to the right.
    """
    rwells = list(input_wells).copy()
    for index, iw in enumerate(input_wells):
        r = iw[0]
        c = int(iw[1:])
        if not c <= 6:
            raise ValueError(f"Only wells up to column 6 can be replicated. (Got '{iw}')")
        rwells.append(f"{r}{c + 6:02d}")
    return rwells


def add_or_assert_coords(coords, pmodel):
    for cname, cvalues in coords.items():
        if cname in pmodel.coords:
            numpy.testing.assert_array_equal(pmodel.coords[cname], cvalues)
        else:
            pmodel.coords[cname] = cvalues


def read_cutinase(DP_RUN, input_wells):
    samples = retl.parse(pathlib.Path(DP_RUN, 'Cutinase_Sample.xml'))
    df_samples = samples['Label1_Copy1'].value
    # Time in minutes and not in hours
    df_time = samples['Label1_Copy1'].time * 60

    assay_wells = replicate_wells(input_wells)
    n_assay_wells = len(assay_wells)
    n_time = len(df_samples)

    # reshape data into two 3D arrays:
    t_obs = numpy.empty((n_assay_wells, n_time))
    y_obs = numpy.empty((n_assay_wells, n_time))
    for w, awell in enumerate(assay_wells):
        t_obs[w, :] = df_time[awell].values
        y_obs[w, :] = df_samples[awell].values
    return df_time, df_samples, t_obs, y_obs


def read_gfp(DP_RUN):
    data = []
    time = []
    t0_measured = None
    i = 0
    while True:
        i += 1
        fp_output = pathlib.Path(DP_RUN, f'read_{i+1:02d}', 'Output.xml')
        if not fp_output.exists():
            break
        rdata = retl.parse(fp_output)
        df = rdata['Label1_Copy1_Copy1'].value
        time_end = rdata['Label1_Copy1_Copy1'].end_utc
        if not t0_measured:
            t0_measured = time_end
        runtime = (time_end-t0_measured).seconds/3600
        data.append(df)
        time.append(runtime)
    df_sGFP = pandas.concat(data, ignore_index = True)
    df_sGFP.index = pandas.Index(time, name="time_hours")

    t0_pipetted = None
    with open(pathlib.Path(DP_RUN, "DEBUG.log")) as origin_file:
        for line in origin_file:
            if "Loading reader..." in line:
                dt_load = datetime.datetime.strptime(line[2:21] + "+0000", "%Y-%m-%dT%H:%M:%S%z")
                t0_pipetted = dt_load + datetime.timedelta(minutes=10)
                break
    t0_delta = (t0_measured - t0_pipetted).total_seconds() / 3600
    print(f"The MTP was pipetted at {t0_pipetted} ({t0_delta:.3f} hours before the end of the first measurement).")
    print("!! Overriding t0_delta with 0.25 hours !!")
    t0_delta = 0.25
    return df_sGFP, t0_delta


def to_long_wells(df, value_name:str):
    df = pandas.DataFrame(
        index=[
            f"{r}{int(c):02d}"
            for r in df.index
            for c in df.columns
        ],
        data={
            value_name: df.values.flatten()
        }
    )
    df.index.name = "well"
    return df


def model_concentrations(df_inputs, sd_sgfp=0.1, sd_cutinase=0.3):
    # first check that the input dataframe corresponds to the standard format:
    if not isinstance(df_inputs, pandas.DataFrame):
        raise ValueError("Expected `df_inputs` to be a DataFrame, but got {type(df_inputs)}.")
    if not df_inputs.index.name == "input_well":
        raise ValueError(
            "The inputs index must be named 'input_well', with entries corresponding to "
            f"well IDs in the deep well plate. Index name was '{df_inputs.index.name}'."
        )
    missing_cols = {"type", "concentration_factor"} -  set(df_inputs.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")
    if len(df_inputs) > 48:
        raise ValueError(f"Too many input wells. Expected max 48, got {len(df_inputs)}.")
    if not numpy.array_equal(df_inputs.index, df_inputs.sort_index().index):
        raise ValueError("The input must be sorted by the 'input_well' index.")

    # input checks successful! Can now start building the model...
    # First get a handle on the PyMC3 model
    pmodel = pymc3.modelcontext(None)
    
    # Now create "coords" that describe the relevant dimensions
    add_or_assert_coords({
        "type": list(sorted(set(df_inputs.type))),
        "input_well": df_inputs.index.values,
        "assay_well": numpy.array(replicate_wells(df_inputs.index)),
    }, pmodel)
    
    cf_input = pymc3.Data(
        "cf_input",
        df_inputs.concentration_factor.values,
        dims=("input_well",)
    )
    cf_cutinase_assay = pymc3.Lognormal(
        "cf_cutinase_assay",
        mu=tt.log(tt.concatenate([cf_input]*2)),
        sd=sd_cutinase,
        dims=("assay_well",)
    )
    cf_sgfp_assay = pymc3.Lognormal(
        "cf_sgfp_assay",
        mu=tt.log(tt.concatenate([cf_input]*2)),
        sd=sd_cutinase,
        dims=("assay_well",)
    )
    type_indices = 2 * list(map(pmodel.coords["type"].index, df_inputs.type))
    return type_indices


def model_cutinase(t_obs, y_obs, em_nitrophenol, type_indices):
    pmodel = pymc3.modelcontext(None)
    add_or_assert_coords({
        "cutinase_cycle": numpy.arange(t_obs.shape[-1]),
    }, pmodel)

    cf_input = pmodel["cf_input"]
    cf_cutinase_assay = pmodel["cf_cutinase_assay"]

    # For the cutinase assay, we introduce the specific enzyme activity in the supernatant [µmol/mL/min].
    # Again, we assume one undiluted activity and calculate the activities in the sample/replicate wells
    # deterministically from the relative concentrations above.
    vmax = pymc3.HalfFlat("vmax", dims=("type",))
    # The inputs are diluted 500x into the assay.
    dilution_factor = pymc3.Data('dilution_factor', 500)
    vmax_assay = pymc3.Deterministic(
        "vmax_assay",
        cf_cutinase_assay * vmax[type_indices] / dilution_factor,
        dims=("assay_well",)
    )
    
    # prediction
    cutinase_time = pymc3.Data('cutinase_time', t_obs, dims=("assay_well", "cutinase_cycle"))
    P0 = pymc3.Uniform('P0', -0.5, 1, dims=("assay_well",))
    product_concentration = pymc3.Deterministic(
        "product_concentration",
        P0[:, None] + vmax_assay[:, None] * cutinase_time,
        dims=("assay_well", "cutinase_cycle")
    )
        
    # The reaction product (4-nitrophenol) is measured in [mmol/L] = [µmol/mL] via the error model.
    # Our prediction is also in µmol/mL.
    absorbance = pymc3.Data("absorbance", y_obs, dims=("assay_well", "cutinase_cycle"))
    L_cut = em_nitrophenol.loglikelihood(
        y=absorbance,
        x=product_concentration,
        replicate_id='cutinase_all',
        dependent_key=em_nitrophenol.dependent_key
    )
    return


def model_sgfp(df_sGFP, t0_delta, type_indices):
    pmodel = pymc3.modelcontext(None)
    add_or_assert_coords({
        "sgfp_cycle": numpy.arange(len(df_sGFP)),
    }, pmodel)

    cf_input = pmodel["cf_input"]
    cf_sgfp_assay = pmodel["cf_sgfp_assay"]

    sgfp_cycle_time = pymc3.Data("sgfp_cycle_time", df_sGFP.index.values, dims=("sgfp_cycle",))
    sgfp_time_lag = pymc3.Lognormal("sgfp_time_lag", numpy.log(t0_delta), sd=1, dims=("assay_well",))
    sgfp_time = pymc3.Deterministic(
        "sgfp_time",
        sgfp_time_lag[:, None] + sgfp_cycle_time[None, :],
        dims=("assay_well", "sgfp_cycle")
    )
    
    # fluorescence limits =^= cutinase amount
    # fmax = fluorescence capacity in undiluted supernatant
    fmax = pymc3.Lognormal("fmax", numpy.log(700), sd=1, dims=("type",))
    # fmax_assay = fluorescence capacity in diluted assay well
    fmax_assay = pymc3.Deterministic(
        "fmax_assay",
        cf_sgfp_assay * fmax[type_indices],
        dims=("assay_well",)
    )

    # saturation curve
    sgfp_ht_assembly = pymc3.HalfNormal("sgfp_ht_assembly", 3)
    tau = sgfp_ht_assembly / numpy.log(2)
    sgfp_assembled_fraction = pymc3.Deterministic(
        "sgfp_assembled_fraction",
        (1 - tt.exp(-sgfp_time / tau)),
        dims=("assay_well", "sgfp_cycle")
    )

    # exponential decay:
    sgfp_ht_decay = pymc3.HalfNormal("sgfp_ht_decay", 20)
    sgfp_ht_decay_assay = pymc3.Lognormal(
        "sgfp_ht_decay_assay",
        tt.log(sgfp_ht_decay), sd=0.5,
        dims=("assay_well",)
    )
    tau_decay = sgfp_ht_decay_assay / numpy.log(2)
    sgfp_decay = pymc3.Deterministic(
        "sgfp_decay",
        (tt.exp(-sgfp_time / tau_decay[:, None])),
        dims=("assay_well", "sgfp_cycle")
    )

    # the final fluorescence depends on the well-wise fluorescence capacity, aggregration and decay
    sgfp_fluorescence_pred = pymc3.Deterministic(
        "sgfp_fluorescence_pred",
        fmax_assay[:, None] * sgfp_assembled_fraction * sgfp_decay,
        dims=("assay_well", "sgfp_cycle")
    )
    
    sgfp_fluorescence_obs = pymc3.Data(
        "sgfp_fluorescence_obs",
        df_sGFP[pmodel.coords["assay_well"]].values.T,
        dims=("assay_well", "sgfp_cycle")
    )
    L_sgfp = pymc3.Normal(
        "L_sgfp",
        mu=sgfp_fluorescence_pred, sd=10,
        observed=sgfp_fluorescence_obs,
        dims=("assay_well", "sgfp_cycle")
    )
    return


def print_coords(coords):
    for cname, cvals in coords.items():
        shape = numpy.shape(cvals)
        vals = numpy.asarray(cvals).flatten()
        if numpy.prod(shape) < 5:
            examples = ", ".join([str(v) for v in vals])
        else:
            examples = ", ".join([str(v) for v in vals][:3])
            examples += f", …, {cvals[-1]}"
        print(f"{cname: <20}{shape}\t{examples}")


def plot_concentration_error(idata):
    fig, axs = pyplot.subplots(dpi=140, figsize=(8, 8), ncols=2, nrows=2, sharex="col")

    positions = idata.constant_data.desired_sample_concentration.values

    samples = list(idata.posterior.sample_concentration.stack(sample=("chain", "draw")).values)
    left, right = axs[0, :]
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
    left.plot([0,1], [0,1], color="gray", linestyle="--")
    right.plot([-2,0], [-2,0], color="gray", linestyle="--")

    for r in idata.posterior.replicate.values:
        samples = list(
            idata.posterior.replicate_concentration.stack(sample=("chain", "draw")).sel(
                replicate=r
            ).values
        )
        left, right = axs[1, :]
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
        left.plot([0,1], [0,1], color="gray", linestyle="--")
        right.plot([-2,0], [-2,0], color="gray", linestyle="--")

    for (left, right) in axs:
        left.set_ylabel("actual concentration factor   [-]")
        right.set_ylabel("log10(actual concentration factor)   [-]")
    axs[1, 0].set_xlabel("desired concentration factor   [-]")
    axs[1, 1].set_xlabel("log10(desired concentration factor)   [-]")
    fig.tight_layout()
    return fig, axs


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = pyplot.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(data.shape[1]))
    ax.set_yticks(numpy.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #pyplot.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(numpy.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(numpy.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, numpy.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts