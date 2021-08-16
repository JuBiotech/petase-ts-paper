import numpy
import pandas

import calibr8


class NitrophenolAbsorbanceModel(calibr8.BasePolynomialModelT):
    def __init__(self):
        super().__init__(
            independent_key='4NP_concentration',
            dependent_key='absorbance',
            mu_degree=1, scale_degree=1
        )
