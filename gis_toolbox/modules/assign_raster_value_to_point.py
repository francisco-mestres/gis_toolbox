import logging
from typing import Any
import geopandas as gpd
import rasterio

# =============================================================================
# Konfiguration und Literal-Definitionen
# =============================================================================
class Config:
    LOG_FILE_PATH = "sample_raster_values.log"  # Pfad zur Log-Datei
    DEFAULT_RASTER_BAND = 1                    # Standard-Rasterband (1-basierter Index)

class LoggingMessages:
    READING_RASTER = "Reading raster from '{}'."  
    SAMPLING_POINTS = "Sampling raster values for {} points."  
    ASSIGNING_VALUES = "Assigning raster values to column '{}'."  
    ERROR_OCCURRED = "Error in assign_raster_value_to_points: {}"  

class TqdmConfig:
    # Falls Fortschrittsbalken benötigt werden (hier nicht verwendet)
    SAMPLING_DESC = "Sampling raster values"
    SAMPLING_UNIT = "points"

# =============================================================================
# Logger-Konfiguration
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
# Produktionsreife Funktion: Rasterwert-Abtastung für Punkt-GDF
# =============================================================================
def assign_raster_value_to_points(
    point_gdf: gpd.GeoDataFrame, 
    raster_file: str, 
    new_column: str
) -> gpd.GeoDataFrame:
    """
    Given a point GeoDataFrame and a raster file, this function samples the raster
    value at each point and assigns it to a new column. The column name is provided 
    as an argument.
    
    Parameters
    ----------
    point_gdf : gpd.GeoDataFrame
        GeoDataFrame containing point geometries.
    raster_file : str
        File path to the raster (e.g., a TIFF file).
    new_column : str
        Name of the new column to store the sampled raster values.
        
    Returns
    -------
    gpd.GeoDataFrame
        The updated GeoDataFrame with a new column containing the raster values.
    """
    try:
        # Logge, dass das Raster eingelesen wird
        logger.info(LoggingMessages.READING_RASTER.format(raster_file))
        
        # Öffne das Raster und bestimme den Standardband
        with rasterio.open(raster_file) as src:
            band = Config.DEFAULT_RASTER_BAND
            # Erstelle eine Liste von Koordinaten aus der Geometrie des GeoDataFrame
            # Annahme: Die Geometrien sind Punkte
            coords = [(geom.x, geom.y) for geom in point_gdf.geometry]
            
            # Logge die Anzahl der zu sampelnden Punkte
            logger.info(LoggingMessages.SAMPLING_POINTS.format(len(coords)))
            
            # Verwende die Rasterio sample-Methode, um die Werte für die Koordinaten zu extrahieren
            samples = list(src.sample(coords, indexes=band))
            # Jedes Element in samples ist ein Array; verwende den ersten Wert, falls vorhanden
            sampled_values = [sample[0] if sample.size > 0 else None for sample in samples]
        
        # Logge, dass die Werte der neuen Spalte zugewiesen werden
        logger.info(LoggingMessages.ASSIGNING_VALUES.format(new_column))
        point_gdf[new_column] = sampled_values
        
        return point_gdf
    
    except Exception as e:
        logger.exception(LoggingMessages.ERROR_OCCURRED.format(e))
        raise


