import geopandas as gpd
import numpy as np
from numba import cuda
import logging
from tqdm import tqdm
from typing import List

from gis_toolbox.enums import GpdEngine, GpdDriver
from gis_toolbox.config import Config, TqdmConfig, LoggingMessages

class Config:
    THREADS_PER_BLOCK = 512
    DEFAULT_GPD_ENGINE = GpdEngine.PYOGRIO.value
    DEFAULT_GPD_DRIVER = GpdDriver.ESRI_SHAPEFILE.value
    LOG_FILE_PATH = "classify_points.log"


class TqdmConfig:
    CLASSIFY_POINTS_DESC = "Klassifizierung der Punkte"  # Beschreibung für die Fortschrittsanzeige
    CLASSIFY_POINTS_UNIT = "Punkt"  


class LoggingMessages:
    ATTRIBUTE_NOT_FOUND = "Attribut '{}' nicht im gdf gefunden."  
    TRANSFERRING_DATA = "Daten werden auf die GPU übertragen."
    STARTING_CLASSIFICATION = "Klassifizierung auf der GPU gestartet." 
    SAVING_OUTPUT = "Speichere klassifizierte Punkte in {}"  
    CLASSIFICATION_COMPLETED = "Klassifizierung abgeschlossen."  
    ERROR_OCCURRED = "Ein Fehler ist während der Klassifizierung aufgetreten: {}" 


# Einstellungen des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@cuda.jit
def classify_kernel(values, thresholds, classes, num_thresholds):
    idx = cuda.grid(1)
    if idx < values.size:
        val = values[idx]
        cls = num_thresholds  # Standardklasse, wenn keine Schwellenwerte erfüllt sind
        for i in range(num_thresholds):
            if val < thresholds[i]:
                cls = i
                break
        classes[idx] = cls


def classify_points_by_attribute(
    input_shp: gpd.GeoDataFrame,
    thresholds: List[float], 
    input_attribute: str,
    output_range_attribute: str,
    threshold_unit="",
    output_shp="",
    driver=Config.DEFAULT_GPD_DRIVER,
    engine=Config.DEFAULT_GPD_ENGINE
):
    
    try:
        
        gdf = input_shp

        if input_attribute not in gdf.columns:
            logger.error(LoggingMessages.ATTRIBUTE_NOT_FOUND.format(input_attribute))
            raise ValueError(LoggingMessages.ATTRIBUTE_NOT_FOUND.format(input_attribute))

        values = gdf[input_attribute].values.astype(np.float32)
        num_points = values.size

        thresholds = np.array(thresholds, dtype=np.float32)
        num_thresholds = thresholds.size

        classes = np.zeros(num_points, dtype=np.int32)

        # CUDA-Einrichtung
        threadsperblock = Config.THREADS_PER_BLOCK
        blockspergrid = (num_points + (threadsperblock - 1)) // threadsperblock

        logger.info(LoggingMessages.TRANSFERRING_DATA)
        d_values = cuda.to_device(values)
        d_thresholds = cuda.to_device(thresholds)
        d_classes = cuda.to_device(classes)

        logger.info(LoggingMessages.STARTING_CLASSIFICATION)
        with tqdm(total=num_points, desc=TqdmConfig.CLASSIFY_POINTS_DESC, unit=TqdmConfig.CLASSIFY_POINTS_UNIT) as pbar:
            classify_kernel[blockspergrid, threadsperblock](
                d_values, d_thresholds, d_classes, num_thresholds
            )
            cuda.synchronize()
            pbar.update(num_points)

        classes = d_classes.copy_to_host()

        # Klassenindizes in Beschriftungen umwandeln
        labels = [f"<{int(thresh)}{threshold_unit}" for thresh in thresholds] + [f">={int(thresholds[-1])}{threshold_unit}"]
        gdf[output_range_attribute] = [labels[cls] for cls in classes]

        if output_shp:
            logger.info(LoggingMessages.SAVING_OUTPUT.format(output_shp))
            gdf.to_file(output_shp, engine=engine, driver=driver)

        logger.info(LoggingMessages.CLASSIFICATION_COMPLETED)

    except Exception as e:
        logger.exception(LoggingMessages.ERROR_OCCURRED.format(e))
        raise  # Stelle sicher, dass die Ausnahme erneut ausgelöst wird

    return gdf