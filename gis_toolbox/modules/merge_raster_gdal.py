import os
import subprocess
import logging
from tqdm import tqdm

from gis_toolbox.utils import ensure_folder_exists
from gis_toolbox.enums import GpdEngine, GpdDriver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Config:
    """Konfigurationsklasse für Standardwerte."""
    DEFAULT_IO_ENGINE = GpdEngine.PYOGRIO.value
    DEFAULT_DRIVER = GpdDriver.ESRI_SHAPEFILE.value
    DEFAULT_CHUNK_SIZE = 10_000


class LoggingMessages:
    """Nachrichten für Logging-Zwecke."""
    CREATING_OUTPUT_DIR = "Erstelle Ausgabeverzeichnis: {folder}"
    SEARCHING_FOR_TIFFS = "Suche nach .tif-Dateien in: {input_dir}"
    NO_TIFFS_FOUND = "Keine .tif-Dateien im Eingabeverzeichnis gefunden."
    TIFFS_FOUND = "{count} .tif-Dateien gefunden. Erstelle Ausgabeverzeichnis, falls benötigt..."
    WRITING_FILE_LIST = "Schreibe Dateiliste nach {filelist_path}"
    BUILDING_VRT = "Erstelle VRT (virtuelles Mosaik)..."
    TRANSLATING_VRT = "Konvertiere VRT in finale zusammengeführte GeoTIFF-Datei..."
    MERGE_SUCCESS = "Erfolgreich Rasterdateien zu {final_tif} zusammengeführt"
    DEFAULT_WRITING_CHUNKS = "Schreibe Daten in Blöcken..."


class OutputConfig:
    OUTPUT_FILENAME = "merged.tif"
    TEMP_VRT_NAME = "merged.vrt"
    TIFF_LIST_NAME = "tiff_list.csv"


class TqdmConfig:
    DEFAULT_UNIT_CHUNKS = "Dateien"
    DESC = "Mosaik erstellen"



def merge_rasters_with_gdal(
    input_dir,
    output_dir,
    output_filename=OutputConfig.OUTPUT_FILENAME,
    temp_vrt_name=OutputConfig.TEMP_VRT_NAME
):
    """
    Führt alle .tif-Dateien in 'input_dir' zu einer einzigen .tif-Datei zusammen, indem GDAL verwendet wird.

    1. Durchsucht 'input_dir' nach .tif-Dateien.
    2. Schreibt diese Dateinamen in eine --optfile-Textdatei.
    3. Erstellt ein VRT (virtuelles Mosaik) aus diesen Dateien mit gdalbuildvrt.
    4. Konvertiert das VRT in eine einzige GeoTIFF-Datei mit Kompression und BigTIFF.

    :param input_dir: Verzeichnis, das die zu zusammenführenden .tif-Dateien enthält
    :param output_dir: Verzeichnis, in das die zusammengeführte .tif-Datei (und die Zwischen-.vrt-Datei) geschrieben wird
    :param output_filename: Name der finalen zusammengeführten TIFF-Datei
    :param temp_vrt_name: Name der Zwischen-.vrt-Datei
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    logger.info(LoggingMessages.SEARCHING_FOR_TIFFS.format(input_dir=input_dir))
    tiff_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".tif")
    )
    if not tiff_files:
        logger.error(LoggingMessages.NO_TIFFS_FOUND)
        return

    logger.info(LoggingMessages.TIFFS_FOUND.format(count=len(tiff_files)))
    ensure_folder_exists(output_dir, LoggingMessages.CREATING_OUTPUT_DIR.format(folder=output_dir))

    # Erstelle eine Textdatei mit der Liste der TIFF-Dateien (eine pro Zeile)
    filelist_path = os.path.join(output_dir, OutputConfig.TIFF_LIST_NAME)
    logger.info(LoggingMessages.WRITING_FILE_LIST.format(filelist_path=filelist_path))
    with open(filelist_path, "w", encoding="utf-8") as f:
        for tif in tqdm(tiff_files, desc=TqdmConfig.DESC, unit=TqdmConfig.UNIT):
            full_path = os.path.join(input_dir, tif)
            f.write(f"{full_path}\n")

    vrt_path = os.path.join(output_dir, temp_vrt_name)
    final_tif = os.path.join(output_dir, output_filename)


    # Erstelle ein VRT mit --optfile, um alle TIFF-Dateien zu referenzieren
    logger.info(LoggingMessages.BUILDING_VRT)
    # Verwende ALL_CPUS für Multi-Threaded-Lesen, falls möglich
    vrt_command = [
        "gdalbuildvrt",
        "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
        vrt_path,
        "--optfile", filelist_path
    ]
    subprocess.run(vrt_command, check=True)


    # Konvertiere VRT in eine finale komprimierte BigTIFF-Datei
    logger.info(LoggingMessages.TRANSLATING_VRT)
    translate_command = [
        "gdal_translate",
        "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=DEFLATE",   # oder LZW, etc.
        "-co", "TILED=YES",          # optional, gut für schnelleres Lesen
        vrt_path,
        final_tif
    ]
    subprocess.run(translate_command, check=True)

    logger.info(LoggingMessages.MERGE_SUCCESS.format(final_tif=final_tif))

