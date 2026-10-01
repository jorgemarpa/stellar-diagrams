import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs


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
        print(f"Querying {self.catalog.upper()} catalog...")

        if self.catalog == "gaia":
            self.data = self._query_gaia()
        elif self.catalog == "tic":
            self.data = self._query_tic()
        else:
            raise ValueError("Catalog must be 'gaia' or 'tic'.")

        print(f"Retrieved and cleaned {len(self.data)} stars.")
        return self.data

    def _query_gaia(self):
        """Executes ADQL query to Gaia DR3."""
        # Removed lum_gspphot and ORDER BY random_index to prevent Server 500 crashes
        query = f"""
        SELECT TOP {self.max_stars}
        source_id, parallax, parallax_over_error, phot_g_mean_mag as app_mag, 
        bp_rp as color, teff_gspphot as teff
        FROM gaiadr3.gaia_source
        WHERE phot_g_mean_mag < {self.mag_limit}
        AND parallax_over_error > 10
        AND parallax > 0
        AND bp_rp IS NOT NULL
        AND teff_gspphot IS NOT NULL
        """
        job = Gaia.launch_job_async(query, dump_to_file=False)
        df = job.get_results().to_pandas()

        # Calculate Absolute Magnitude: M = m - 10 + 5*log10(parallax_in_mas)
        df["abs_mag"] = df["app_mag"] - 10 + 5 * np.log10(df["parallax"])
        return df

    def _query_tic(self):
        """Queries the TESS Input Catalog via MAST."""
        mast_data = Catalogs.query_criteria(
            catalog="Tic", Tmag=[0, self.mag_limit], plx=[5, 1000], objType="STAR"
        )
        df = mast_data.to_pandas()

        # Drop rows missing crucial data (removed 'lum' dependency)
        df = df.dropna(subset=["Tmag", "plx", "e_plx", "Teff"])

        df = df.rename(columns={"Tmag": "app_mag", "plx": "parallax", "Teff": "teff"})

        # Apply data cleaning
        df["parallax_over_error"] = df["parallax"] / df["e_plx"]
        df = df[df["parallax_over_error"] > 10].copy()

        # Calculate Absolute Magnitude
        df["abs_mag"] = df["app_mag"] - 10 + 5 * np.log10(df["parallax"])

        # TIC fallback mapping
        df["color"] = df["teff"]

        return df.head(self.max_stars)

    def plot(self, diagram="hr", style="hex", figsize=(8, 6)):
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data found. Run fetch_data() first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Both diagrams will now use abs_mag as the Y-axis (Standard observational practice)
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

        # Invert axes (Brighter magnitudes are smaller numbers; hotter temps are higher numbers)
        ax.invert_yaxis()

        if diagram == "cmd" and self.catalog == "tic":
            ax.invert_xaxis()
        elif diagram == "hr":
            ax.invert_xaxis()

        plt.tight_layout()
        plt.show()
