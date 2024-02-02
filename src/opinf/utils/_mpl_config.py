# utils/_mpl_config.py
"""Matplotlib configuration for documenation notebooks."""

__all__ = [
    "mpl_config",
]

import pandas as pd
import matplotlib.pyplot as plt


def mpl_config():
    """Set matplotlib and pandas configuration defaults for the
    documentation notebooks.
    """
    # Matplotlib customizations.
    plt.rc("axes.spines", right=False, top=False)
    plt.rc("figure", dpi=300, figsize=(9, 3))
    plt.rc("font", family="serif")
    plt.rc("legend", edgecolor="none", frameon=False)
    plt.rc("text", usetex=True)

    # Pandas display options.
    pd.options.display.float_format = "{:.4%}".format
