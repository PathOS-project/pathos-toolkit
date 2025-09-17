# Repository effect: Effects of Data Repositories on Data Usage

This folder contains the code, documentation, and analysis results for the repository effect case study, part of the PathOS project.

## Overview

This case study investigates the effect of data repositories on the use of data for research. How can we establish usage of datasets in research? Are datasets from some repositories more likely to be used for research than other repositories? What factors are relevant for those differences in usage?

We were interested in the effect of the repository where data is shared on the subsequent usage of that data. For research articles, we know that it matters where an article is published for subsequent citations. Does something similar happen with citations to datasets, and are citations affected by the repository where the dataset is stored? That is, would sharing data in a particular repository result in more reuse?

We analyse these effects in general, but also with a particular interest in the social sciences, based on three different studies:

- we study the extent to which data usage can be automatically inferred from scholarly publications in the social sciences;

- we interview scientists in the social sciences about their data usage, and what role data repositories play;

- we study data usage quantitatively on the basis of the Data Citation Corpus.

## Repository Contents

### Analysis Scripts

This repository includes core code to replicate the analysis in the PathOS case study on repositories. This only includes the essential code to reproduce the analysis, and does not include any ancillary code for producing tables and/or figures. Moreover, we do not include any code to run [`datastet`](https://github.com/kermitt2/grobid) or [`SciNoBo`](https://github.com/iNoBo/scinobo-raa), since this requires a custom setup that is documented at their respective repositories. This means that this repository only includes the code for the regression model that is used to analyse the data citations. The data that the model can be fitted to is available from https://doi.org/xxx

We use [Stan](https://mc-stan.org/) to specify the regression model used to analyse the data citations. The model is available from the file `data_citations.stan`. The model is quite simple and rather self-explanatory. Stan offers various interfaces to fit the model to the data, including a command-line, Julia, Python and R interface.

### Data
- See `DATA_ACCESS.md` for information on accessing the data used in this case study
- Processed datasets and results are deposited in Zenodo (see DATA_ACCESS.md for DOI)

## Usage

### To reproduce the analysis:
1. Access the data as described in `DATA_ACCESS.md`
2. Install [Stan](https://mc-stan.org/)
3. Fit the model to the data

## External Dependencies

The analysis relies on [Stan](https://mc-stan.org/), which can be used from various interface.

## Contact

Please contact Vincent Traag (v.a.traag@cwts.leidenuniv.nl) for more information.
