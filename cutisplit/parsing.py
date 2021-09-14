import datetime
import os
import pathlib
import typing

import numpy
import pandas
import retl


def read_nitrophenol_calibration(dp_run: os.PathLike):
    """Parse Nitrophenol calibration data into DataFrame for calibr8 model."""
    df_standards = retl.parse(pathlib.Path(dp_run, "Cutinase_Standard.xml"))[
        "Label1_Copy1"
    ].value

    # concentrations of triplicates, ordered Fortran style:
    standard_concentrations = pandas.read_excel(
        pathlib.Path(dp_run, "Wells.xlsx"), sheet_name="Cutinase_Standard"
    )["Concentration"].values.clip(0.0001)
    # numpy magic to expand it into the same shape as the wells:
    standard_concentrations = numpy.repeat(
        standard_concentrations.reshape(3, 8).flatten("F")[None, :], repeats=3
    ).reshape(8, 3 * 3) / 6 #final dilution of 40 ul in 200 ul

    # the corresponding well IDs:
    standard_wells = numpy.array(
        [f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)]
    ).reshape(8, 12)[:, 3:]

    # compile into a nice DataFrame
    df_calibration = pandas.DataFrame(
        columns=["concentration", "absorbance"],
        index=standard_wells.flatten(),
    )
    df_calibration["concentration"] = standard_concentrations.flatten()
    df_calibration["absorbance"] = df_standards.loc[1, df_calibration.index]
    df_calibration.sort_values("concentration", inplace=True)
    return df_calibration


def read_cutinase(
    dp_run: os.PathLike,
) -> pandas.DataFrame:
    """Parse cutinase data into DataFrame."""
    samples = retl.parse(pathlib.Path(dp_run, "Cutinase_Sample.xml"))
    df_value = samples["Label1_Copy1"].value
    # Time in minutes and not in hours
    df_time = samples["Label1_Copy1"].time * 60

    df_time.columns.name = "well"
    df_value.columns.name = "well"

    df_cutinase = (
        pandas.DataFrame(data={"time": df_time.stack(), "value": df_value.stack()})
        .reorder_levels(["well", "cycle"])
        .sort_index()
    )
    return df_cutinase


def read_sgfp(
    dp_run: os.PathLike,
) -> typing.Tuple[pandas.DataFrame, float]:
    """Parse sGFP data into DataFrame and time delta between pipetting and measurement."""
    data = []
    time = []
    t0_measured = None
    i = 0
    while True:
        i += 1
        fp_output = pathlib.Path(dp_run, f"read_{i+1:02d}", "Output.xml")
        if not fp_output.exists():
            break
        rdata = retl.parse(fp_output)
        df = rdata["Label1_Copy1_Copy1"].value
        time_end = rdata["Label1_Copy1_Copy1"].end_utc
        if not t0_measured:
            t0_measured = time_end
        runtime = (time_end - t0_measured).seconds / 3600
        data.append(df)
        time.append(runtime)
    df_sGFP = pandas.concat(data, ignore_index=True)
    df_sGFP.index = pandas.Index(time, name="time_hours")

    # extract the time delay betwen pipetting and measurement from the log file
    t0_pipetted = None
    with open(pathlib.Path(dp_run, "DEBUG.log")) as origin_file:
        for line in origin_file:
            if "Loading reader..." in line:
                dt_load = datetime.datetime.strptime(
                    line[2:21] + "+0000", "%Y-%m-%dT%H:%M:%S%z"
                )
                t0_pipetted = dt_load + datetime.timedelta(minutes=10)
                break
            elif "sGFP assay: Assay start" in line:
                dt_load = datetime.datetime.strptime(
                    line[2:21] + "+0000", "%Y-%m-%dT%H:%M:%S%z"
                )
                t0_pipetted = dt_load + datetime.timedelta(minutes=10)
                break
    t0_delta = (t0_measured - t0_pipetted).total_seconds() / 3600
    print(
        f"The MTP was pipetted at {t0_pipetted} ({t0_delta:.2f} hours before the end of the first measurement)."
    )
    return df_sGFP, t0_delta
