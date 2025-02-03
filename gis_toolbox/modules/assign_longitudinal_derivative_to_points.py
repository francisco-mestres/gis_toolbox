import logging
import numpy as np
import geopandas as gpd

# =============================================================================
# Configuration classes for literals and settings
# =============================================================================
class Config:
    LOG_FILE_PATH = "compute_longitudinal_slope_from_discretization.log"  # Pfad zur Log-Datei

class LoggingMessages:
    MISSING_COLUMN = "Column '{}' not found in GeoDataFrame."
    COMPUTING_GROUPS = "Computing longitudinal slope for {} groups (by {})."
    SLOPE_COMPUTED = "Longitudinal slope computed for {} points."
    ERROR_COMPUTING_SLOPE = "Error computing longitudinal slope: {}"

class TqdmConfig:
    # Falls Sie einen Fortschrittsbalken verwenden möchten, definieren Sie hier die Literale
    SLOPE_COMPUTATION_DESC = "Computing longitudinal slope"
    SLOPE_COMPUTATION_UNIT = "groups"


# =============================================================================
# Logger configuration (Logger-Konfiguration)
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Production-ready function to compute longitudinal slope from discretized points
# =============================================================================
def compute_longitudinal_derivative_from_discretized_gdf(
    discretized_gdf: gpd.GeoDataFrame,
    elevation_column: str,
    slope_column: str,
    chainage_column: str,
    row_idx_column: str,
    sub_idx_column: str,
    group_columns: list = None
) -> gpd.GeoDataFrame:
    """
    Computes the derivative of the raster (elevation) values with respect to the chainage,
    i.e. the longitudinal slope along the road. The order of the points is determined based on
    the chainage as well as group indices (e.g., row_idx, sub_idx) from the discretization.
    
    Parameters
    ----------
    discretized_gdf : gpd.GeoDataFrame
        Input GeoDataFrame produced by the discretization pipeline. It must contain the following columns:
            - elevation_column: sampled elevation (from DEM)
            - chainage_column: chainage (distance along the road)
            - group_columns: one or more columns to group points (e.g., row_idx, sub_idx)
    elevation_column : str
        Name of the column that holds the DEM-sampled elevation values.
    slope_column : str
        Name of the new column to store the computed longitudinal slopes.
    chainage_column : str, optional
        Name of the column that holds the chainage/distance values (default "chainage").
    group_columns : list, optional
        List of column names to group the points by. Defaults to ["row_idx", "sub_idx"].
    
    Returns
    -------
    gpd.GeoDataFrame
        The updated GeoDataFrame with a new column (slope_column) containing the computed longitudinal slopes.
    """
    try:
        # Falls keine Gruppierungsspalten angegeben wurden, verwende Standardwerte.
        if group_columns is None:
            group_columns = [row_idx_column, sub_idx_column]

        # Überprüfen, ob alle erforderlichen Spalten vorhanden sind.
        required_columns = [elevation_column, chainage_column] + group_columns
        for col in required_columns:
            if col not in discretized_gdf.columns:
                msg = LoggingMessages.MISSING_COLUMN.format(col)
                logger.error(msg)
                raise ValueError(msg)

        # Sortiere das GeoDataFrame nach den Gruppierungsspalten und der Kettenlänge (chainage)
        sorted_gdf = discretized_gdf.sort_values(by=group_columns + [chainage_column]).copy()
        
        # Gruppiere nach den angegebenen Spalten
        groups = sorted_gdf.groupby(group_columns)
        num_groups = groups.ngroups
        logger.info(LoggingMessages.COMPUTING_GROUPS.format(num_groups, group_columns))

        # Initialisiere ein Array für die computed slopes (float64)
        slopes = np.full(len(sorted_gdf), np.nan, dtype=np.float64)

        # Für jede Gruppe: Berechne den Unterschied der Elevation und des chainage und leite ab.
        # Der erste Punkt jeder Gruppe erhält np.nan.
        for _, group_df in groups:
            # group_df ist bereits nach chainage sortiert
            elev = group_df[elevation_column].to_numpy(dtype=np.float64)
            ch = group_df[chainage_column].to_numpy(dtype=np.float64)
            # Berechne Differenzen: diff_elev / diff_chainage
            delta_elev = np.diff(elev)
            delta_ch = np.diff(ch)
            # Division; falls delta_ch==0 wird np.nan zugewiesen
            group_slopes = np.where(delta_ch != 0, delta_elev / delta_ch, np.nan)
            # Der erste Wert bleibt np.nan; fügen Sie die Ergebnisse in das global slopes-Array ein.
            slopes[group_df.index[1:]] = group_slopes

        # Füge die berechneten Steigungen als neue Spalte in das sortierte GeoDataFrame ein.
        sorted_gdf[slope_column] = slopes

        logger.info(LoggingMessages.SLOPE_COMPUTED.format(len(sorted_gdf)))
        
        # Optional: Sortiere zurück in die ursprüngliche Reihenfolge, falls erforderlich.
        # Hier wird der Index beibehalten.
        return sorted_gdf

    except Exception as e:
        logger.exception(LoggingMessages.ERROR_COMPUTING_SLOPE.format(e))
        raise


