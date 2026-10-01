import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import fetch_catalog_data


class StellarDiagramMaker:
    """
    A tool to query Gaia DR3 or TIC catalogs and generate
    Color-Magnitude Diagrams (CMD) or Hertzsprung-Russell (HR) diagrams.
    """

    def __init__(self, catalog="gaia", max_stars=10000, mag_limit=12):
        self.catalog = catalog.lower()
        self.max_stars = max_stars
        self.mag_limit = mag_limit
        self.data = None

    def fetch_data(self):
        """
        Queries the selected catalog, utilizing lru_cache from utils for duplicate requests.
        """
        # Check cache hits before calling the function
        pre_cache_info = fetch_catalog_data.cache_info()

        # Call the cached standalone function
        # Append .copy() so plotting manipulations don't mutate the cached DataFrame
        self.data = fetch_catalog_data(
            self.catalog, self.max_stars, self.mag_limit
        ).copy()

        # Check cache hits after calling the function to see if we just loaded from memory
        post_cache_info = fetch_catalog_data.cache_info()

        if post_cache_info.hits > pre_cache_info.hits:
            print(f"Loading {self.catalog.upper()} data from memory cache...")
            print(f"Loaded {len(self.data)} cached stars.")

        return self.data

    def save_catalog(self, filepath):
        """
        Saves the currently loaded catalog data to a local CSV file.

        Parameters:
        - filepath (str): The local file path (e.g., 'data/gaia_sample.csv').
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data to save. Run fetch_data() first.")

        # Ensure the target directory exists
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.data.to_csv(filepath, index=False)
        print(f"Saved {len(self.data)} stars to {filepath}")

    def load_catalog(self, filepath):
        """
        Loads catalog data from a local CSV file into the maker instance.

        Parameters:
        - filepath (str): The local file path to load (e.g., 'data/gaia_sample.csv').
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")

        self.data = pd.read_csv(filepath)
        print(f"Loaded {len(self.data)} stars from {filepath}")
        return self.data

    def plot(self, diagram="hr", style="hex", figsize=(8, 6), ax=None):
        """
        Generates the requested plot.

        Parameters:
        - diagram (str): 'cmd' (Color vs Abs Mag) or 'hr' (Temp vs Abs Mag).
        - style (str): 'scatter', 'hex', or 'kde'.
        - figsize (tuple): Figure dimensions (ignored if ax is provided).
        - ax (matplotlib.axes.Axes, optional): An existing axis to plot on.

        Returns:
        - ax (matplotlib.axes.Axes): The axis containing the plot.
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data found. Run fetch_data() first.")

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        else:
            fig = ax.get_figure()

        if diagram == "cmd":
            x_col = "color"
            y_col = "abs_mag"
            x_label = (
                "Color (BP - RP)"
                if self.catalog == "gaia"
                else "Effective Temperature (K)"
            )
            y_label = "Absolute Magnitude"
        elif diagram == "hr":
            x_col = "teff"
            y_col = "abs_mag"
            x_label = "Effective Temperature (K)"
            y_label = "Absolute Magnitude"
        else:
            raise ValueError("diagram must be 'cmd' or 'hr'")

        plot_df = self.data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[x_col, y_col]
        )

        if style == "scatter":
            sns.scatterplot(
                x=x_col,
                y=y_col,
                data=plot_df,
                s=5,
                color="k",
                alpha=0.3,
                edgecolor=None,
                ax=ax,
            )
        elif style == "hex":
            hb = ax.hexbin(
                plot_df[x_col], plot_df[y_col], gridsize=60, cmap="inferno", bins="log"
            )
            fig.colorbar(hb, ax=ax, label="log(N)")
        elif style == "kde":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.kdeplot(
                    x=x_col,
                    y=y_col,
                    data=plot_df,
                    fill=True,
                    cmap="mako",
                    levels=20,
                    ax=ax,
                )
        else:
            raise ValueError("style must be 'scatter', 'hex', or 'kde'")

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(
            f"{diagram.upper()} Diagram ({self.catalog.upper()}, N={len(plot_df)})",
            fontsize=14,
        )

        if not ax.yaxis_inverted():
            ax.invert_yaxis()

        if diagram == "cmd" and self.catalog == "tic":
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
        elif diagram == "hr":
            if not ax.xaxis_inverted():
                ax.invert_xaxis()

        if created_fig:
            plt.tight_layout()

        return ax
