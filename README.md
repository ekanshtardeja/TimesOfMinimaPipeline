
# TimesOfMinimaPipeline

A Python-based pipeline for automatically extracting times of minima from eclipsing binary star light curves using photometric data available through the [MAST Portal](https://mast.stsci.edu/).

## Features

- **Automated Data Retrieval:** Accepts a star name as input and retrieves available light curves from MAST.
- **Light Curve Processing:** Processes and plots light curves obtained from photometric data files.
- **Gaussian Fitting:** Fits Gaussian functions to light curves to identify eclipse minima.
- **Times of Minima Extraction:** Extracts the times of minima from the fitted Gaussian profiles.
- **Cadence-Based Rating:** Assigns a quality rating to each light curve based on its observational cadence, helping prioritize datasets with better temporal resolution.

## How It Works

1. **Input:** Enter the name of an eclipsing binary star.
2. **Data Retrieval:** The pipeline searches MAST for available light curves of the target.
3. **Processing:** Retrieved light curves are processed and plotted.
4. **Fitting:** Gaussian functions are fitted to identify eclipse minima.
5. **Extraction:** Times of minima are calculated from the fitted profiles.
6. **Rating:** Each light curve receives a rating based on its cadence duration.

## Requirements

The pipeline is designed for Python and may use libraries such as:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `astropy`
- `astroquery`



## Author

Developed for the analysis of eclipsing binary star light curves and eclipse-timing variations.
