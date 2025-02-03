import logging
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from numba import njit

# =============================================================================
# Konfiguration und Literal-Definitionen
# =============================================================================
class Config:
    LOG_FILE_PATH = "numba_buffer_linestrings.log"   # Pfad zur Log-Datei
    DEFAULT_GEOMETRY_COLUMN = "geometry"             # Standard-Geometriespalte
    BUFFER_APPROXIMATION_EPS = 1e-12                   # Toleranz für Degeneration

class LoggingMessages:
    START_BUFFER = "Starting Numba-based buffering for linestrings with buffer distance {}."
    PROCESSING_GEOMETRY = "Processing geometry index {}."
    BUFFER_COMPLETE = "Buffering completed for {} geometries."
    ERROR_OCCURRED = "Error in buffer_linestrings_gdf_numba: {}"

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
# Numba-jitted Funktion zur Berechnung des Offset-Polygons eines Linestrings
# =============================================================================
@njit
def compute_offset_polygon(coords, d):
    """
    Berechnet ein approximiertes Buffer-Polygon für einen Linestring.
    
    Parameter
    ----------
    coords : 2D np.array of float64
        Array der Koordinaten des Linestrings (n x 2).
    d : float
        Buffer-Distanz.
        
    Returns
    -------
    poly : 2D np.array of float64
        Array der Polygon-Koordinaten (2*n x 2). Die ersten n Punkte entsprechen
        der linken Offset-Kurve, die letzten n der rechten in umgekehrter Reihenfolge.
    """
    n = coords.shape[0]
    left_offsets = np.empty_like(coords)
    right_offsets = np.empty_like(coords)
    normals = np.empty((n - 1, 2), dtype=coords.dtype)
    
    # Berechne Normale für jeden Segment
    for i in range(n - 1):
        dx = coords[i+1, 0] - coords[i, 0]
        dy = coords[i+1, 1] - coords[i, 1]
        seg_len = np.sqrt(dx * dx + dy * dy)
        if seg_len < Config.BUFFER_APPROXIMATION_EPS:
            normals[i, 0] = 0.0
            normals[i, 1] = 0.0
        else:
            # Linksnormale: (-dy, dx) normalisiert
            normals[i, 0] = -dy / seg_len
            normals[i, 1] = dx / seg_len

    # Berechne an jedem Scheitelpunkt den gemittelten Normalenvektor
    for i in range(n):
        if i == 0:
            avg_nx = normals[0, 0]
            avg_ny = normals[0, 1]
        elif i == n - 1:
            avg_nx = normals[n - 2, 0]
            avg_ny = normals[n - 2, 1]
        else:
            avg_nx = (normals[i - 1, 0] + normals[i, 0]) * 0.5
            avg_ny = (normals[i - 1, 1] + normals[i, 1]) * 0.5
            norm = np.sqrt(avg_nx * avg_nx + avg_ny * avg_ny)
            if norm > Config.BUFFER_APPROXIMATION_EPS:
                avg_nx /= norm
                avg_ny /= norm
        left_offsets[i, 0] = coords[i, 0] + d * avg_nx
        left_offsets[i, 1] = coords[i, 1] + d * avg_ny
        right_offsets[i, 0] = coords[i, 0] - d * avg_nx
        right_offsets[i, 1] = coords[i, 1] - d * avg_ny

    # Baue das Polygon: linke Offset-Kurve in normaler Reihenfolge und rechte Offset-Kurve in umgekehrter Reihenfolge
    poly = np.empty((2 * n, 2), dtype=coords.dtype)
    for i in range(n):
        poly[i, 0] = left_offsets[i, 0]
        poly[i, 1] = left_offsets[i, 1]
    for i in range(n):
        poly[n + i, 0] = right_offsets[n - 1 - i, 0]
        poly[n + i, 1] = right_offsets[n - 1 - i, 1]
    return poly

# =============================================================================
# Wrapper-Funktion für ein einzelnes Linestring-Objekt
# =============================================================================
def buffer_linestring_numba(linestring, buffer_distance):
    """
    Wendet die Numba-basierte Buffer-Berechnung auf einen einzelnen Linestring an.
    
    Parameter
    ----------
    linestring : shapely.geometry.LineString
        Der Eingabe-Linestring.
    buffer_distance : float
        Buffer-Distanz.
        
    Returns
    -------
    shapely.geometry.Polygon
        Das approximierte Buffer-Polygon.
    """
    # Konvertiere den Linestring in ein numpy-Array der Koordinaten
    coords = np.array(linestring.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        # Bei weniger als 2 Punkten gibt es keinen sinnvollen Buffer
        return linestring
    poly_coords = compute_offset_polygon(coords, buffer_distance)
    return Polygon(poly_coords)

# =============================================================================
# Produktionsreife Funktion: Buffer für ein GeoDataFrame mit Linestrings unter Verwendung von Numba
# =============================================================================
def buffer_linestrings_gdf_numba(
    gdf: gpd.GeoDataFrame,
    buffer_distance: float,
    geometry_column: str = Config.DEFAULT_GEOMETRY_COLUMN
) -> gpd.GeoDataFrame:
    """
    Given a GeoDataFrame with Linestring geometries, this function computes an
    approximate buffer (as a Polygon) for each geometry using a Numba-accelerated
    algorithm and returns a new GeoDataFrame with the buffered geometries.
    
    Parameter
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing Linestring geometries.
    buffer_distance : float
        Buffer distance to be applied (in CRS units).
    geometry_column : str, optional
        Name of the geometry column (default is "geometry").
        
    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame with the buffered geometries.
    """
    try:
        logger.info(LoggingMessages.START_BUFFER.format(buffer_distance))
        buffered_geoms = []
        # Iteriere über alle Geometrien im GeoDataFrame
        for idx, geom in enumerate(gdf[geometry_column]):
            logger.info(LoggingMessages.PROCESSING_GEOMETRY.format(idx))
            if geom is None:
                buffered_geoms.append(None)
            elif geom.geom_type == "LineString":
                buffered_geom = buffer_linestring_numba(geom, buffer_distance)
                buffered_geoms.append(buffered_geom)
            else:
                # Falls Geometrie nicht vom Typ LineString, verwende Shapely-Fallback
                buffered_geoms.append(geom.buffer(buffer_distance))
        new_gdf = gdf.copy()
        new_gdf[geometry_column] = buffered_geoms
        logger.info(LoggingMessages.BUFFER_COMPLETE.format(len(new_gdf)))
        return new_gdf
    except Exception as e:
        logger.exception(LoggingMessages.ERROR_OCCURRED.format(e))
        raise

# =============================================================================
# Example usage (commented out)
# =============================================================================
# if __name__ == "__main__":
#     # Beispiel: Laden eines GeoDataFrame mit Linestrings
#     gdf_lines = gpd.read_file("lines.shp")
#     # Numba-basierte Buffer-Berechnung mit einem Buffer-Distance von 10.0 (CRS-Einheiten)
#     buffered_gdf = buffer_linestrings_gdf_numba(gdf_lines, buffer_distance=10.0)
#     # Speichern des gepufferten GeoDataFrame
#     buffered_gdf.to_file("buffered_lines.shp")

# =============================================================================
# Example usage (commented out)
# =============================================================================
if __name__ == "__main__":
    # Beispiel: Laden eines GeoDataFrame mit Linestrings
    gdf_lines = gpd.read_file("/mnt/c/Users/fmest/Desktop/Pave for Climate/gis_tools_francisco_mestres/gis_toolbox/data/input/shape/autobahnen.shp")
    # GPU-basierte Buffer-Berechnung mit einem Buffer-Distance von 10.0 (CRS-Einheiten)
    buffered_gdf = buffer_linestrings_gdf_numba(gdf_lines, buffer_distance=10.0)
    # Speichern des gepufferten GeoDataFrame
    buffered_gdf.to_file("buffered_lines.shp")
