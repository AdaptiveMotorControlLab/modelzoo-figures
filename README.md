# DeepLabCut Modelzoo - Figures and Data

Figures for the modelzoo paper.

## Quickstart

Please add `.py` files into `src`; these should be made by downloading your `ipython notebook` to a `.py` file.

Make sure you are in a python>=3.8 environment that supports the `pip install` command (e.g., a virtual environment or a conda environment). Install dependencies, then render of all figures using:

```bash
make -j8 all
```

Figures will be placed in `ipynb` format into the `figures/` directory.
Before a PR, make sure to execute the notebooks with this command to update the `figures/` repository.

## Dependencies

```bash
pip install -r requirements.txt
```

Make sure to install `jupytext` and use this to save notebooks in the `src` folder.

## Repo organization

- ``src``: Jupyter notebooks for reproducing the paper figures, in python format
- ``data``: Folder to data files
- ``figures``: Rendered paper figures in `ipynb` format. Do not edit ipynb here directly.
