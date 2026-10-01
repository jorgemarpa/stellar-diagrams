import numpy as np
import pandas as pd
from functools import lru_cache
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs


@lru_cache(maxsize=32)
def fetch_catalog_data(
    catalog: str, max_stars: int, mag_limit: float, stratified: bool = False
) -> pd.DataFrame:
    """
    Fetch and clean data from the specified astronomical catalog.

    Parameters
    ----------
    catalog : str
        The name of the catalog to query ('gaia' or 'tic').
    max_stars : int
        Maximum number of stars to return.
    mag_limit : float
        Faintest apparent magnitude to query.
    stratified : bool, optional
        If True, query specific stellar populations to ensure a well-populated
        diagram across all evolutionary stages (default is False).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the queried and cleaned stellar data.
    """
    M_SUN = 4.74

    if stratified:
        print(
            f"Querying {catalog.upper()} catalog over network with stratified sampling..."
        )
        sub_limit = max_stars // 4

        if catalog == "gaia":
            populations = {
                "Main Sequence": "bp_rp BETWEEN 0.5 AND 2.0 AND (phot_g_mean_mag - 10 + 5 * log10(parallax)) BETWEEN 4 AND 10",
                "Red Giants": "bp_rp > 1.0 AND (phot_g_mean_mag - 10 + 5 * log10(parallax)) < 4",
                "White Dwarfs": "bp_rp < 1.0 AND (phot_g_mean_mag - 10 + 5 * log10(parallax)) > 10",
                "Horizontal Branch": "bp_rp < 1.0 AND (phot_g_mean_mag - 10 + 5 * log10(parallax)) < 4",
            }

            dfs = []
            for pop_name, condition in populations.items():
                print(f"  -> Fetching {pop_name}...")
                query = f"""
                SELECT TOP {sub_limit}
                source_id, parallax, parallax_over_error, phot_g_mean_mag as app_mag, 
                bp_rp as color, teff_gspphot as teff
                FROM gaiadr3.gaia_source
                WHERE phot_g_mean_mag < {mag_limit}
                AND parallax_over_error > 10
                AND parallax > 0
                AND bp_rp IS NOT NULL
                AND teff_gspphot IS NOT NULL
                AND {condition}
                """
                job = Gaia.launch_job_async(query, dump_to_file=False)
                df_pop = job.get_results().to_pandas()
                if not df_pop.empty:
                    dfs.append(df_pop)

            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            if not df.empty:
                df["abs_mag"] = df["app_mag"] - 10 + 5 * np.log10(df["parallax"])
                df["log_lum"] = -0.4 * (df["abs_mag"] - M_SUN)

        elif catalog == "tic":
            populations = {
                "Main Sequence": {"Teff": [3000, 8000], "logg": [4.0, 5.5]},
                "Giants": {"Teff": [3000, 6000], "logg": [1.0, 3.5]},
                "White Dwarfs": {"Teff": [8000, 50000], "logg": [6.0, 9.0]},
                "Hot Stars / HB": {"Teff": [8000, 50000], "logg": [3.0, 4.5]},
            }

            dfs = []
            for pop_name, kwargs in populations.items():
                print(f"  -> Fetching {pop_name}...")
                try:
                    mast_data = Catalogs.query_criteria(
                        catalog="Tic",
                        Tmag=[0, mag_limit],
                        plx=[5, 1000],
                        objType="STAR",
                        **kwargs,
                    )
                    df_pop = mast_data.to_pandas()

                    if (
                        not df_pop.empty
                        and "Tmag" in df_pop.columns
                        and "plx" in df_pop.columns
                    ):
                        df_pop = df_pop.dropna(subset=["Tmag", "plx", "e_plx", "Teff"])
                        dfs.append(df_pop.head(sub_limit))
                except Exception as e:
                    print(f"     [!] Failed to fetch {pop_name}: {e}")

            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            if not df.empty:
                df = df.rename(
                    columns={"Tmag": "app_mag", "plx": "parallax", "Teff": "teff"}
                )
                df["parallax_over_error"] = df["parallax"] / df["e_plx"]
                df = df[df["parallax_over_error"] > 10].copy()
                df["abs_mag"] = df["app_mag"] - 10 + 5 * np.log10(df["parallax"])
                df["log_lum"] = -0.4 * (df["abs_mag"] - M_SUN)
                df["color"] = df["teff"]

        else:
            raise ValueError("Catalog must be 'gaia' or 'tic'.")

    else:
        print(f"Querying {catalog.upper()} catalog over network (standard sampling)...")
        if catalog == "gaia":
            query = f"""
            SELECT TOP {max_stars}
            source_id, parallax, parallax_over_error, phot_g_mean_mag as app_mag, 
            bp_rp as color, teff_gspphot as teff
            FROM gaiadr3.gaia_source
            WHERE phot_g_mean_mag < {mag_limit}
            AND parallax_over_error > 10
            AND parallax > 0
            AND bp_rp IS NOT NULL
            AND teff_gspphot IS NOT NULL
            """
            job = Gaia.launch_job_async(query, dump_to_file=False)
            df = job.get_results().to_pandas()

            if not df.empty:
                df["abs_mag"] = df["app_mag"] - 10 + 5 * np.log10(df["parallax"])
                df["log_lum"] = -0.4 * (df["abs_mag"] - M_SUN)

        elif catalog == "tic":
            mast_data = Catalogs.query_criteria(
                catalog="Tic", Tmag=[0, mag_limit], plx=[5, 1000], objType="STAR"
            )
            df = mast_data.to_pandas()

            if not df.empty and "Tmag" in df.columns and "plx" in df.columns:
                df = df.dropna(subset=["Tmag", "plx", "e_plx", "Teff"])
                df = df.rename(
                    columns={"Tmag": "app_mag", "plx": "parallax", "Teff": "teff"}
                )
                df["parallax_over_error"] = df["parallax"] / df["e_plx"]
                df = df[df["parallax_over_error"] > 10].copy()
                df["abs_mag"] = df["app_mag"] - 10 + 5 * np.log10(df["parallax"])
                df["log_lum"] = -0.4 * (df["abs_mag"] - M_SUN)
                df["color"] = df["teff"]
                df = df.head(max_stars)

        else:
            raise ValueError("Catalog must be 'gaia' or 'tic'.")

    print(f"Retrieved and cleaned a combined total of {len(df)} stars.")
    return df
