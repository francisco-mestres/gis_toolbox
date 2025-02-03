from numba import cuda
import numpy as np
import geopandas as gpd
import requests
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import math


from gis_toolbox.enums import GeometryType
from gis_toolbox.utils import ensure_folder_exists


class LoggingMessages:
    START_DOWNLOAD = "Beginn des Herunterladens von Dateien..."
    DOWNLOAD_SUCCESS = "Datei erfolgreich heruntergeladen."
    DOWNLOAD_ERROR = "Fehler beim Herunterladen von {url}. Fehler: {error}"
    UNSUPPORTED_GEOMETRY = "Nicht unterstützter Geometrietyp: {geometry_type}"
    MISSING_FOLDER = "Erstelle Ordner: {folder}"
    INTERPOLATING_POINTS = "Interpoliere Punkte für Geometrien..."
    KEY_EXTRACTION_DONE = "Extraktion der Schlüssel abgeschlossen."


class Config:
    LOG_FILE_PATH = "downloads.log"
    DEFAULT_URL_PATTERN = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/\
                        dgm1_tiff/dgm1_tiff//x={x_key}&y={y_key}&spacing={spacing}"
    DEFAULT_SPACING = 1000
    DEFAULT_MAX_WORKERS = 8


# Einstellungen des Loggings
logger = logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)


####### HILFSFUNKTIONEN #######

@cuda.jit(device=True)
def interpolate_points_numba(coords, spacing):
    """
    Interpoliert Punkte entlang eines Liniensegments, falls der Abstand zwischen aufeinanderfolgenden Punkten
    größer als das angegebene Intervall (spacing) ist. Die Optimierung erfolgt durch Numba.

    Parameter:
        coords (np.ndarray): Array der Form (n, 2) mit (x, y)-Koordinaten.
        spacing (float): Der gewünschte Abstand zwischen interpolierten Punkten.

    Rückgabe:
        np.ndarray: Interpolierte (x, y)-Koordinaten.
    """
    result = []
    for i in range(len(coords) - 1):

        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]

        result.append((x1, y1))  # Startpunkt hinzufügen
        
        # Berechnung der Distanz und Richtung
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        
        # Interpolieren, falls die Segmentlänge größer als der Abstand ist
        if dist > spacing:

            num_points = int(dist // spacing)

            for j in range(1, num_points + 1):

                t = j * spacing / dist
                interp_x = x1 + t * dx
                interp_y = y1 + t * dy
                result.append((interp_x, interp_y))
    
    result.append(coords[-1])  # Endpunkt hinzufügen

    return np.array(result)



####### HILFSFUNKTIONEN #######


def get_intersecting_url_with_interpolation(gdf: gpd.GeoDataFrame, spacing=Config.DEFAULT_SPACING):
    """
    Extrahiert die überlappenden Schlüssel aus den Scheitelpunkten von LineString- oder Polygon-Geometrien
    und stellt sicher, dass das Intervall eingehalten wird, auch wenn der Abstand zwischen den Punkten größer ist.

    Parameter:
        gdf (gpd.GeoDataFrame): GeoDataFrame mit Geometrien, die die zu überlappenden Bereiche darstellen.

    Rückgabe:
        set: Eine Menge von Tupeln (x_key, y_key, spacing), wobei x_key und y_key abgeleitete Schlüssel sind.
    """

    bounding_keys = set()

    logger.info(LoggingMessages.INTERPOLATING_POINTS)
    for geometry in gdf.geometry:

        if geometry.geom_type == GeometryType.LINESTRING.value:
            # Umwandeln in ein Numpy-Array zur schnellen Verarbeitung
            coordinates = np.array(geometry.coords)
            interpolated = interpolate_points_numba(coordinates, spacing)

        elif geometry.geom_type == GeometryType.POLYGON.value:
            # Umwandeln der äußeren Begrenzung in ein Numpy-Array
            coordinates = np.array(geometry.exterior.coords)
            interpolated = interpolate_points_numba(coordinates, spacing)

        else:
            raise ValueError(LoggingMessages.UNSUPPORTED_GEOMETRY.format(geometry_type=geometry.geom_type))

        for x, y in interpolated:
            x_key = int(x // spacing)
            y_key = int(y // spacing)
            bounding_keys.add((x_key, y_key, spacing))

    logger.info(LoggingMessages.KEY_EXTRACTION_DONE)

    return bounding_keys


def convert_keys_to_urls(bounding_keys, url_pattern=Config.DEFAULT_URL_PATTERN):
    """
    Konvertiert Schlüssel in eine Liste von URLs, indem die Schlüssel in das URL-Muster eingefügt werden.


    :param param bounding_keys (set): Eine Menge von Tupeln (x_key, y_key, spacing).
    :param url_pattern (str): URL-Muster mit Platzhaltern für x_key, y_key und spacing.

    :return list: Eine Liste von URLs.
    """
    urls = []
    for x_key, y_key, spacing in bounding_keys:
        url = url_pattern.format(x_key=x_key, y_key=y_key, spacing=spacing)
        urls.append(url)
    return urls


def download_single_file(url: str, local_path: str):
    """
    Lädt eine Datei herunter, falls sie nicht existiert. Gibt den lokalen Pfad bei Erfolg zurück, sonst None.

    Parameter:
        url (str): Die URL der Datei.
        local_path (str): Der lokale Speicherort der Datei.

    Rückgabe:
        str oder None: Der lokale Pfad bei Erfolg, None bei Fehler.
    """
    if os.path.exists(local_path):
        return local_path
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(LoggingMessages.DOWNLOAD_SUCCESS)
        return local_path
    
    except Exception as e:
        logger.error(LoggingMessages.DOWNLOAD_ERROR.format(url=url, error=str(e)))
        return None


def download_files_multithreaded(urls, output_folder, max_workers=Config.DEFAULT_MAX_WORKERS):
    """
    Lädt mehrere Dateien parallel mit Threading herunter.

    Parameter:
        urls (list): Eine Liste von URLs der Dateien, die heruntergeladen werden.
        output_folder (str): Der Ordner, in dem die Dateien gespeichert werden sollen.
        max_workers (int): Die maximale Anzahl gleichzeitiger Threads.

    Rückgabe:
        None
    """
    ensure_folder_exists(output_folder, LoggingMessages.MISSING_FOLDER)

    local_paths = [os.path.join(output_folder, os.path.basename(url)) for url in urls]

    logger.info(LoggingMessages.START_DOWNLOAD)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(download_single_file, urls, local_paths))


