""" Helper functions that have simple dependencies.
"""
import typing


def replicate_wells_from(input_wells: typing.Sequence[str]) -> typing.List[str]:
    """Creates a list of well IDs containing the input well IDs
    and corresponding well ID shifted by 6 columns to the right.
    """
    rwells = list(input_wells).copy()
    for index, iw in enumerate(input_wells):
        r = iw[0]
        c = int(iw[1:])
        if not c <= 6:
            raise ValueError(
                f"Only wells up to column 6 can be replicated. (Got '{iw}')"
            )
        rwells.append(f"{r}{c + 6:02d}")
    return rwells
