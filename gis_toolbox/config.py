import os
from gis_toolbox.enums import GpdEngine, GpdDriver

class Config:
    ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    MAX_DISTANCE_FACTOR = 1.8
    DEFAULT_GPD_DRIVER = GpdDriver.ESRI_SHAPEFILE.value
    DEFAULT_GPD_ENGINE = GpdEngine.PYOGRIO.value
    LOG_FILE_PATH = os.path.join(ROOT_DIRECTORY, "logs", "gis_toolbox.log")
    DEFAULT_SPACING = 10.0
    DEFAULT_LINE_LENGTH_THRESHOLD = 0.1
    DEFAULT_THREADS_PER_BLOCK = 512
    DEFAULT_BATCH_SIZE = 100_000
    DEFAULT_URL_PATTERN = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_tiff/dgm1_tiff//x={x_key}&y={y_key}&spacing={spacing}"
    DEFAULT_MAX_WORKERS = 8

class LoggingMessages:
    READING_INPUT = "Lese Punkte-Shapefile: {}"
    INPUT_EMPTY = "Das Eingabe-Shapefile ist leer. Nichts zu verarbeiten."
    MISSING_COLUMN = "Fehlende Spalte {} in den Punktdaten."
    CREATING_LINES = "Erstelle {} Liniensegmente mit Übergängen."
    SAVING_OUTPUT = "Speichere zusammengeführte Linien in {}"
    STARTING_DISTANCE_CALCULATION = "Starte Distanzberechnung..."
    READING_POINT_LAYER = "Lese Punkt-Layer: {}"
    POINT_LAYER_EMPTY = "Der Punkt-Layer ist leer."
    COMBINING_POLYGON_BOUNDARIES = "Kombiniere Polygon-Grenzen."
    NO_LINE_SEGMENTS_WARNING = "Keine Liniensegmente gefunden."
    CHECKING_POINTS_INSIDE_POLYGONS = "Überprüfe Punkte innerhalb von Polygonen."
    CALCULATING_DISTANCES = "Berechne Distanzen."
    DISTANCE_CALCULATION_COMPLETED = "Distanzberechnung abgeschlossen."
    ERROR_OCCURRED = "Ein Fehler ist während der Klassifizierung aufgetreten: {}"
    POINTS_GENERATED = "{} Punkte mit Normalrichtungen generiert."
    WRITING_POINTS = "Schreiben von {} Punkten in {}"
    SUCCESSFULLY_WRITTEN_POINTS = "Punkte erfolgreich in {} geschrieben."
    INVALID_INPUT_SHAPEFILE = "Ungültige Eingabe-Shapefile."
    EMPTY_INPUT_SHAPEFILE = "Eingabe-Shapefile ist leer. Keine Linien zu verarbeiten."
    START_DOWNLOAD = "Beginn des Herunterladens von Dateien..."
    DOWNLOAD_SUCCESS = "Datei erfolgreich heruntergeladen."
    DOWNLOAD_ERROR = "Fehler beim Herunterladen von {url}. Fehler: {error}"
    UNSUPPORTED_GEOMETRY = "Nicht unterstützter Geometrietyp: {geometry_type}"
    MISSING_FOLDER = "Erstelle Ordner: {folder}"
    INTERPOLATING_POINTS = "Interpoliere Punkte für Geometrien..."
    KEY_EXTRACTION_DONE = "Extraktion der Schlüssel abgeschlossen."
    ATTRIBUTE_NOT_FOUND = "Attribut '{}' nicht im Shapefile gefunden."
    TRANSFERRING_DATA = "Daten werden auf die GPU übertragen."
    STARTING_CLASSIFICATION = "Klassifizierung auf der GPU gestartet."
    CLASSIFICATION_COMPLETED = "Klassifizierung abgeschlossen."

class TqdmConfig:
    DESC = "Klassifizierung der Punkte"
    UNIT = "Punkt"
    DEFAULT_UNIT_CHUNKS = "rows"
    DISCRETIZING_LINES = "Diskretisieren von Linien"

class OutputConfig:
    ROW_IDX_COL = "row_idx"
    SUB_IDX_COL = "sub_idx"
    CHAINAGE_COL = "chainage"
    NORMAL_X_COL = "normal_x"
    NORMAL_Y_COL = "normal_y"
    GEOMETRY_COL = "geometry"
    DEFAULT_DISTANCE_LABEL = "Entfernung"
    DEFAULT_INSIDE_LABEL = "Innerhalb"
    OUTPUT_FILE_DRIVER = "ESRI Shapefile"
    OUTPUT_FILENAME = "merged.tif"
    TEMP_VRT_NAME = "merged.vrt"
    TIFF_LIST_NAME = "tiff_list.csv"