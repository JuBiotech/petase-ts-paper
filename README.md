# Project for PETase secretion screening

[![DOI](https://zenodo.org/badge/518074824.svg)](https://zenodo.org/badge/latestdoi/518074824)

This projects provides the raw data, data analysis notebooks as well as figures for the manuscript __"Explore or exploit? A model-based screening strategy for PETase secretion by _Corynebacterium glutamicum_"__ (2022) by Laura M. Helleckes*, Carolin Müller*, Tim Griesbach, Vera Waffenschmidt, Matthias Moch, Michael Osthege, Wolfgang Wiechert, Marco Oldiges.

*These authors contributed equally.

## Structure
Raw data can be found in the `data` folder. Data analysis is conducted in `data_analysis`.
To load the MCMC traces, please head over to [Zenodo](https://doi.org/10.5281/zenodo.13985584), download and unzip the folder structure and merge with the folder `data_analysis`.
The MCMC files end in `.nc`.
Plots for the accompanying paper can be found in `paper`.
The results for individual experiments can be found by a unique idetentifier, their so-called Run ID.

The following runs were conducted:

| Run ID                | Enzyme | Purpose                              |
| --------------------- | ------ | ------------------------------------ |
| BWA7DQ                | LCC    | TS Round 1.0                         |
| BWNZ99                | PE-H   | TS Round 1.0                         |
| BZACW9                | LCC    | TS Round 1.1                         |
| BZP1XQ                | PE-H   | TS Round 1.1                         |
| C3C1XZ                | LCC    | TS Round 2                           |
| C4PZHQ                | PE-H   | TS Round 2                           |
| CB4MNH                | PE-H   | Improved Assay Effect                |
| Fermentation_LipALipB | LCC    | Comparison to liter-scale bioreactor |
| Fermentation_YoaW     | LCC    | Comparison to liter-scale bioreactor |

In Thompson Sampling (TS), round 1 refers to the combined analysis of runs BWA7DQ & BZACW9 in case of LCC and BWNZ99 & BZP1XQ in case of PE-H.
The results for the combined analysis can be found in the subfolders BZACW9 and BZP1XQ for LCC and PE-H respectively.

Run CB4MNH was conducted after experimental improvements with tip wetting to show that the previous bias in the columns of the assay MTP could be eradicated.

## Citation of code
This repository and the corresponding Python package for data analysis (`cutisplit`) is licensed under the [GNU Affero General Public License v3.0](https://github.com/JuBiotech/petase-ts-paper/blob/main/LICENSE.md).
Head over to [Zenodo](https://doi.org/10.5281/zenodo.6908128) to generate a BibTeX citation for the latest release.
