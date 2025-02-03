import math
import logging
from typing import List
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from numba import cuda
from tqdm import tqdm

# =============================================================================
# Configuration classes for literals and settings
# =============================================================================
class Config:
    THREADS_PER_BLOCK = 512
    LOG_FILE_PATH = "compute_slopes.log"


class LoggingMessages:
    ATTRIBUTE_NOT_FOUND = "Attribute '{}' not found in GeoDataFrame."
    ERROR_COMPUTING_SLOPES = "Error computing embankment slopes: {}"
    SLOPES_COMPUTED = "Embankment slopes computed for {} points."
    READING_DEM = "Reading DEM from {}"
    COMPUTING_BOUNDING_BOX = "Computing bounding box for sampling coordinates."
    COPYING_TO_GPU = "Transferring data to GPU."
    LAUNCHING_KERNEL = "Launching GPU kernel for slope computation."


class TqdmConfig:
    PROFILE_COMPUTATION_DESC = "Computing slopes"
    PROFILE_COMPUTATION_UNIT = "points"


# =============================================================================
# Logger-Konfiguration
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# GPU Kernel zur Berechnung der Profil-Steigungen
# =============================================================================
@cuda.jit
def compute_profile_slopes_kernel(x0, y0, normal_x, normal_y, distances, dem,
                                  dem_width, dem_height, pixel_width, pixel_height,
                                  origin_x, origin_y, step, min_slope_threshold,
                                  left_slopes, right_slopes):
    # Jeder Thread bearbeitet einen einzelnen Punkt
    i = cuda.grid(1)
    if i < x0.shape[0]:
        valid_left = False
        left_max = 0.0
        valid_right = False
        right_max = 0.0
        left_prev = 0.0
        right_prev = 0.0
        left_initialized = False
        right_initialized = False

        # Schleife über alle Abstände (Proben im Profil)
        for j in range(distances.shape[0]):
            d = distances[j]
            # Berechnung der Abtastkoordinate für den aktuellen Punkt
            x_sample = x0[i] + d * normal_x[i]
            y_sample = y0[i] + d * normal_y[i]

            # Umrechnung in Pixelkoordinaten im DEM-Subset
            col_f = (x_sample - origin_x) / pixel_width
            row_f = (origin_y - y_sample) / pixel_height

            col0 = int(math.floor(col_f))
            row0 = int(math.floor(row_f))
            col1 = col0 + 1
            row1 = row0 + 1

            elev = np.nan
            # Überprüfen, ob die Pixelkoordinaten innerhalb des DEM-Subsets liegen
            if col0 >= 0 and row0 >= 0 and col1 < dem_width and row1 < dem_height:
                a = col_f - col0
                b = row_f - row0
                # Bilineare Interpolation
                v00 = dem[row0, col0]
                v10 = dem[row0, col1]
                v01 = dem[row1, col0]
                v11 = dem[row1, col1]
                elev = v00 * (1 - a) * (1 - b) + v10 * a * (1 - b) + v01 * (1 - a) * b + v11 * a * b

            # Berechnung der Steigung für die linke Seite (d < 0)
            if d < 0:
                if not math.isnan(elev):
                    if not left_initialized:
                        left_prev = elev
                        left_initialized = True
                    else:
                        if not math.isnan(left_prev):
                            slope = abs(elev - left_prev) / step
                            if slope >= min_slope_threshold and slope > left_max:
                                left_max = slope
                                valid_left = True
                        left_prev = elev
            # Berechnung der Steigung für die rechte Seite (d > 0)
            elif d > 0:
                if not math.isnan(elev):
                    if not right_initialized:
                        right_prev = elev
                        right_initialized = True
                    else:
                        if not math.isnan(right_prev):
                            slope = abs(elev - right_prev) / step
                            if slope >= min_slope_threshold and slope > right_max:
                                right_max = slope
                                valid_right = True
                        right_prev = elev

        # Speichern der Ergebnisse für den aktuellen Punkt
        if valid_left:
            left_slopes[i] = left_max
        else:
            left_slopes[i] = np.nan
        if valid_right:
            right_slopes[i] = right_max
        else:
            right_slopes[i] = np.nan


# =============================================================================
# Funktion zur Berechnung der Talud-Steigungen unter Verwendung der GPU
# =============================================================================
def compute_embankment_slope_from_dem(point_data: gpd.GeoDataFrame,
                                       raster_data: str,
                                       lateral_distance: float,
                                       normal_x_col="normal_x",
                                       normal_y_col="normal_y",
                                       step: float = 1.0,
                                       min_slope_threshold: float = 0.0,
                                       side_slope_r_col="side_right",
                                       side_slope_l_col="side_left") -> gpd.GeoDataFrame:
    """
    Compute embankment slopes on both sides of each point by sampling the DEM
    along the normal vector.

    Parameters
    ----------
    point_data : gpd.GeoDataFrame
        Input points with 'geometry', 'normal_x', and 'normal_y'.
    raster_data : str
        Path to the DEM TIFF file.
    lateral_distance : float
        Maximum lateral distance (in CRS units) to sample on each side.
    step : float, optional
        Sampling interval (default is 1.0).
    min_slope_threshold : float, optional
        Minimum slope threshold to consider (default is 0.0).

    Returns
    -------
    gpd.GeoDataFrame
        Updated GeoDataFrame with 'side_slope_left' and 'side_slope_right' columns.
    """
    try:
        # Überprüfen, ob die erforderlichen Attribute vorhanden sind
        if normal_x_col not in point_data.columns or normal_y_col not in point_data.columns:
            msg = LoggingMessages.ATTRIBUTE_NOT_FOUND.format(f"{normal_x_col} or {normal_y_col}")
            logger.error(msg)
            raise ValueError(msg)

        n_points = len(point_data)
        # Erstellen des Abstandsarrays von -lateral_distance bis +lateral_distance
        distances = np.arange(-lateral_distance, lateral_distance + step, step, dtype=np.float32)
        n_samples = distances.shape[0]

        # Extrahieren der Punktkoordinaten und Normalen
        x0 = point_data.geometry.x.values.astype(np.float32)
        y0 = point_data.geometry.y.values.astype(np.float32)
        normal_x = point_data[normal_x_col].values.astype(np.float32)
        normal_y = point_data[normal_y_col].values.astype(np.float32)

        # Berechnung des Bounding Boxes für alle Abtastkoordinaten
        logger.info(LoggingMessages.COMPUTING_BOUNDING_BOX)
        xs = x0[:, None] + distances[None, :] * normal_x[:, None]
        ys = y0[:, None] + distances[None, :] * normal_y[:, None]
        xs_flat = xs.ravel()
        ys_flat = ys.ravel()
        min_x, max_x = xs_flat.min(), xs_flat.max()
        min_y, max_y = ys_flat.min(), ys_flat.max()

        # Öffnen des DEM und Lesen des relevanten Subsets
        logger.info(LoggingMessages.READING_DEM.format(raster_data))
        with rasterio.open(raster_data) as dem_ds:
            dem_transform = dem_ds.transform
            pixel_width = dem_transform.a
            pixel_height = -dem_transform.e  # positiver Wert
            margin_x = pixel_width
            margin_y = pixel_height

            window = from_bounds(min_x - margin_x, min_y - margin_y,
                                 max_x + margin_x, max_y + margin_y,
                                 transform=dem_transform)
            dem_subset = dem_ds.read(1, window=window).astype(np.float32)
            new_transform = dem_ds.window_transform(window)

        # Extrahieren der neuen Transformationsparameter
        pixel_width_new = new_transform.a
        pixel_height_new = -new_transform.e  # positiver Wert
        origin_x = new_transform.c
        origin_y = new_transform.f
        dem_height, dem_width = dem_subset.shape

        # Datenübertragung auf die GPU
        logger.info(LoggingMessages.COPYING_TO_GPU)
        d_dem = cuda.to_device(dem_subset)
        d_x0 = cuda.to_device(x0)
        d_y0 = cuda.to_device(y0)
        d_normal_x = cuda.to_device(normal_x)
        d_normal_y = cuda.to_device(normal_y)
        d_distances = cuda.to_device(distances)

        # Erzeugen von Ausgabearrays auf der GPU
        d_left_slopes = cuda.device_array(n_points, dtype=np.float32)
        d_right_slopes = cuda.device_array(n_points, dtype=np.float32)

        # Kernel-Startparameter definieren
        threadsperblock = Config.THREADS_PER_BLOCK
        blockspergrid = (n_points + threadsperblock - 1) // threadsperblock

        logger.info(LoggingMessages.LAUNCHING_KERNEL)
        # Starten des GPU-Kernels
        compute_profile_slopes_kernel[blockspergrid, threadsperblock](
            d_x0, d_y0, d_normal_x, d_normal_y, d_distances, d_dem,
            dem_width, dem_height, pixel_width_new, pixel_height_new,
            origin_x, origin_y, step, min_slope_threshold,
            d_left_slopes, d_right_slopes
        )
        cuda.synchronize()

        # Kopieren der Ergebnisse von der GPU zur CPU
        left_slopes = d_left_slopes.copy_to_host()
        right_slopes = d_right_slopes.copy_to_host()

        # Hinzufügen der berechneten Steigungen zum GeoDataFrame
        point_data[side_slope_l_col] = left_slopes
        point_data[side_slope_r_col] = right_slopes

        logger.info(LoggingMessages.SLOPES_COMPUTED.format(n_points))
        return point_data

    except Exception as e:
        logger.exception(LoggingMessages.ERROR_COMPUTING_SLOPES.format(e))
        raise


# =============================================================================
# Example usage (uncomment and adapt for production):
# =============================================================================
# updated_gdf = compute_embankment_slope_from_dem(
#     point_data=input_gdf,
#     raster_data="path/to/dem.tif",
#     lateral_distance=10.0,
#     step=1.0,
#     min_slope_threshold=0.1
# )


point_data = gpd.read_file("/mnt/c/Users/fmest/Desktop/Pave for Climate/gis_tools_francisco_mestres/gis_toolbox/output_discretized_points.shp")

updated_gdf = compute_embankment_slope_from_dem(
    point_data=point_data,
    raster_data=r"/mnt/c/Users/fmest/Desktop/Pave for Climate/gis_tools_francisco_mestres/gis_toolbox/data/input/raster/dem.tif",
    lateral_distance=10.0,
    step=1.0,
    min_slope_threshold=0.1
 )

updated_gdf.to_file("output_slopes_1.shp", driver="ESRI Shapefile")




