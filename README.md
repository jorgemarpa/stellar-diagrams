# Stellar Diagrams

`stellar-diagrams` is a Python package designed to fetch astronomical catalog data and effortlessly generate Color-Magnitude Diagrams (CMD) and Hertzsprung-Russell (HR) diagrams. 

It provides an intuitive interface for astronomers and hobbyists to:
* Select between major input catalogs (Gaia DR3 or the TESS Input Catalog).
* Control the sample size and set apparent magnitude limits for faster, optimized querying.
* Automatically apply data-cleaning steps to remove stars with poor parallax (distance) measurements.
* Visualize the stellar populations using different plotting styles, including scatter plots, hexbin histograms, and Kernel Density Estimation (KDE), powered by `matplotlib` and `seaborn`.

## Tutorial

To see step-by-step examples of how to query data and create beautiful CMD and HR diagrams, please check out our interactive Jupyter Notebook tutorial:

👉 **[View the Tutorial](docs/tutorials/tutorial-1.ipynb)**

## Installation

This package is built to work seamlessly with the `uv` package manager. To install the dependencies, run:

```bash
uv sync
```

Or install this package using pip

```bash
pip install .
```

## Quick Start

Here is a brief example of how to use `stellar-diagrams` in your code:

```python
from stellar_diagrams import StellarDiagramMaker

# Initialize the tool for Gaia DR3
maker = StellarDiagramMaker(catalog='gaia', max_stars=10000, mag_limit=11)

# Fetch and clean the data (applies parallax cuts automatically)
maker.fetch_data()

# Plot a Color-Magnitude Diagram using a hexbin style
maker.plot(diagram='cmd', style='hex')
```


## TODO

Future features to implement:
- Add temperature colorbar
- Add labels for HR regions
- Add example of know stars to the diagram with names