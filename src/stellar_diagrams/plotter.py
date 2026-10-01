import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from .utils import fetch_catalog_data


class StellarDiagramMaker:
    """
    A tool to query Gaia DR3 or TIC catalogs and generate
    Color-Magnitude Diagrams (CMD) or Hertzsprung-Russell (HR) diagrams.
    """

    def __init__(
        self, catalog: str = "gaia", max_stars: int = 10000, mag_limit: float = 12.0
    ) -> None:
        """
        Initialize the diagram maker.

        Parameters
        ----------
        catalog : str, optional
            The name of the catalog to query ('gaia' or 'tic') (default is 'gaia').
        max_stars : int, optional
            Maximum number of stars to return (default is 10000).
        mag_limit : float, optional
            Faintest apparent magnitude to query (default is 12.0).
        """
        self.catalog = catalog.lower()
        self.max_stars = max_stars
        self.mag_limit = mag_limit
        self.data: Optional[pd.DataFrame] = None

    def fetch_data(self, stratified: bool = False) -> pd.DataFrame:
        """
        Query the selected catalog, utilizing an LRU cache for duplicate requests.

        Parameters
        ----------
        stratified : bool, optional
            If True, uses stratified sampling to ensure representation across
            White Dwarfs, Main Sequence, Giants, etc. (default is False).

        Returns
        -------
        pd.DataFrame
            A dataframe containing the fetched stellar data.
        """
        pre_cache_info = fetch_catalog_data.cache_info()
        self.data = fetch_catalog_data(
            self.catalog, self.max_stars, self.mag_limit, stratified
        ).copy()
        post_cache_info = fetch_catalog_data.cache_info()

        if post_cache_info.hits > pre_cache_info.hits:
            print(f"Loading {self.catalog.upper()} data from memory cache...")
            print(f"Loaded {len(self.data)} cached stars.")

        return self.data

    def save_catalog(self, filepath: str) -> None:
        """
        Save the currently loaded catalog data to a local CSV file.

        Parameters
        ----------
        filepath : str
            The local file path to save the data (e.g., 'data/gaia_sample.csv').

        Raises
        ------
        ValueError
            If no data has been fetched or loaded into the instance yet.
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data to save. Run fetch_data() first.")
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.data.to_csv(filepath, index=False)
        print(f"Saved {len(self.data)} stars to {filepath}")

    def load_catalog(self, filepath: str) -> pd.DataFrame:
        """
        Load catalog data from a local CSV file into the maker instance.

        Parameters
        ----------
        filepath : str
            The local file path to load (e.g., 'data/gaia_sample.csv').

        Returns
        -------
        pd.DataFrame
            The loaded dataframe.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")
        self.data = pd.read_csv(filepath)
        print(f"Loaded {len(self.data)} stars from {filepath}")
        return self.data

    def plot(
        self,
        diagram: str = "hr",
        style: str = "hex",
        figsize: Tuple[float, float] = (8.0, 6.0),
        ax: Optional[Axes] = None,
        plot_radius_lines: bool = False,
    ) -> Axes:
        """
        Generate the requested diagram (CMD or HR).

        Parameters
        ----------
        diagram : str, optional
            Type of diagram to plot, either 'cmd' (Color vs Abs Mag) or
            'hr' (Temp vs Log Luminosity) (default is 'hr').
        style : str, optional
            Plot style, one of 'scatter', 'hex', or 'kde' (default is 'hex').
        figsize : tuple of float, optional
            Figure dimensions (ignored if ax is provided) (default is (8.0, 6.0)).
        ax : matplotlib.axes.Axes, optional
            An existing axis to plot on (default is None).
        plot_radius_lines : bool, optional
            If True, overlay constant stellar radius lines (HR diagram only)
            (default is False).

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the generated plot.

        Raises
        ------
        ValueError
            If no data has been fetched, or if an invalid diagram/style is provided.
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data found. Run fetch_data() or load_catalog() first.")

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
            y_col = "log_lum"
            x_label = "Effective Temperature (K)"
            y_label = r"Log Luminosity ($\log_{10} L / L_\odot$)"

            # Ensure the axis defaults to log scale prior to rendering
            ax.set_xscale("log")
        else:
            raise ValueError("diagram must be 'cmd' or 'hr'")

        plot_df = self.data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[x_col, y_col]
        )

        # Plot the requested style, enforcing log scale transformations natively where needed
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
                zorder=2,
            )
        elif style == "hex":
            xscale_val = "log" if diagram == "hr" else "linear"
            hb = ax.hexbin(
                plot_df[x_col],
                plot_df[y_col],
                gridsize=60,
                cmap="inferno",
                bins="log",
                xscale=xscale_val,
                zorder=2,
            )
            fig.colorbar(hb, ax=ax, label="log(N)")
        elif style == "kde":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_scale_val = (True, False) if diagram == "hr" else False
                sns.kdeplot(
                    x=x_col,
                    y=y_col,
                    data=plot_df,
                    fill=True,
                    cmap="mako",
                    levels=20,
                    ax=ax,
                    zorder=2,
                    log_scale=log_scale_val,
                )
        else:
            raise ValueError("style must be 'scatter', 'hex', or 'kde'")

        # --- Axis Format and Inversions ---
        if diagram == "cmd":
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            if self.catalog == "tic" and not ax.xaxis_inverted():
                ax.invert_xaxis()
        elif diagram == "hr":
            if ax.yaxis_inverted():
                ax.invert_yaxis()
            if not ax.xaxis_inverted():
                ax.invert_xaxis()

        # Force plain numbers (disable scientific notation) for any Temperature axis
        if "Temperature" in x_label:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)

        # --- Overlay Constant Radius Lines ---
        if diagram == "hr" and plot_radius_lines:
            T_sun = 5778
            radii = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            t_max = max(xlim)
            t_min = min(xlim)

            # Use geomspace for a smooth distribution of points on a logarithmic scale
            T_vals = np.geomspace(t_min, t_max, 200)

            for R in radii:
                log_L_vals = 2 * np.log10(R) + 4 * np.log10(T_vals / T_sun)

                if np.min(log_L_vals) < max(ylim) and np.max(log_L_vals) > min(ylim):
                    ax.plot(
                        T_vals,
                        log_L_vals,
                        color="gray",
                        linestyle="--",
                        alpha=0.5,
                        linewidth=1,
                        zorder=1,
                    )

                    y_at_tmax = 2 * np.log10(R) + 4 * np.log10(t_max / T_sun)
                    if min(ylim) < y_at_tmax < max(ylim):
                        ax.text(
                            t_max,
                            y_at_tmax,
                            f" {R} $R_\\odot$",
                            color="gray",
                            alpha=0.8,
                            fontsize=10,
                            va="bottom",
                            ha="left",
                            zorder=3,
                        )
                    else:
                        y_at_tmin = 2 * np.log10(R) + 4 * np.log10(t_min / T_sun)
                        if min(ylim) < y_at_tmin < max(ylim):
                            ax.text(
                                t_min,
                                y_at_tmin,
                                f"{R} $R_\\odot$ ",
                                color="gray",
                                alpha=0.8,
                                fontsize=10,
                                va="bottom",
                                ha="right",
                                zorder=3,
                            )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(
            f"{diagram.upper()} Diagram ({self.catalog.upper()}, N={len(plot_df)})",
            fontsize=14,
        )

        if created_fig:
            plt.tight_layout()

        return ax
