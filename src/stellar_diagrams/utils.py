import numpy as np
from functools import lru_cache
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs

@lru_cache(maxsize=32)
def fetch_catalog_data(catalog, max_stars, mag_limit):
    """
    Fetches and cleans data from the specified astronomical catalog.
    Uses lru_cache to remember previous queries in memory.
    """
    print(f"Querying {catalog.upper()} catalog over network...")
    
    if catalog == 'gaia':
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
        
        # Calculate absolute magnitude
        df['abs_mag'] = df['app_mag'] - 10 + 5 * np.log10(df['parallax'])
        
    elif catalog == 'tic':
        mast_data = Catalogs.query_criteria(
            catalog="Tic", 
            Tmag=[-5, mag_limit], 
            plx=[5, 1000], 
            objType="STAR"
        )
        df = mast_data.to_pandas()
        
        # Clean and rename
        df = df.dropna(subset=['Tmag', 'plx', 'e_plx', 'Teff'])
        df = df.rename(columns={'Tmag': 'app_mag', 'plx': 'parallax', 'Teff': 'teff'})
        
        # Filter for good parallax signal-to-noise
        df['parallax_over_error'] = df['parallax'] / df['e_plx']
        df = df[df['parallax_over_error'] > 10].copy()
        
        # Calculate absolute magnitude and map color fallback
        df['abs_mag'] = df['app_mag'] - 10 + 5 * np.log10(df['parallax'])
        df['color'] = df['teff'] 
        
        # Limit to requested star count
        df = df.head(max_stars)
        
    else:
        raise ValueError("Catalog must be 'gaia' or 'tic'.")
        
    print(f"Retrieved and cleaned {len(df)} stars.")
    return df