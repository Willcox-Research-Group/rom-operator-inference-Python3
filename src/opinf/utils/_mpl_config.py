# utils/_mpl_config.py
"""Matplotlib configuration for documenation notebooks."""

__all__ = [
    "mpl_config",
]

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
    try:
        import pandas as pd
    except ModuleNotFoundError as ex:  # pragma: no cover
        if ex.args[0] != "No module named 'pandas'":
            raise
    else:
        pd.options.display.float_format = "{:.4%}".format
