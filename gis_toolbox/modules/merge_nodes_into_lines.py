import geopandas as gpd
from shapely.geometry import LineString
from math import sqrt
import logging

from gis_toolbox.enums import GpdEngine, GpdDriver  # Importiere Enums


# Klassen zur Speicherung von Literalen
class Config:
    MAX_DISTANCE_FACTOR = 1.8  # Maximaler Faktor für den Abstand zwischen Punkten
    DEFAULT_GPD_DRIVER = GpdDriver.ESRI_SHAPEFILE.value  
    DEFAULT_GPD_ENGINE = GpdEngine.PYOGRIO.value
    LOG_FILE_PATH = "merge_points.log"  

class LoggingMessages:
    READING_INPUT = "Lese Punkte-Shapefile: {}"  
    INPUT_EMPTY = "Das Eingabe-Shapefile ist leer. Nichts zu verarbeiten."  
    MISSING_COLUMN = "Fehlende Spalte {} in den Punktdaten." 
    CREATING_LINES = "Erstelle {} Liniensegmente mit Übergängen."  
    SAVING_OUTPUT = "Speichere zusammengeführte Linien in {}" 


class OutputConfig:
    DEFAULT_ROW_IDX_COL = "row_idx"  # Standard-Spalte für die Zeilenkennung
    DEFAULT_SUB_IDX_COL = "sub_idx"  # Standard-Spalte für die Unterzeilenkennung
    DEFAULT_CHAINAGE_COL = "chainage"  # Standard-Spalte für die Entfernung entlang der Linie


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def merge_points_into_lines(
    input_points_shp: gpd.GeoDataFrame,
    range_col: str,
    discretization_distance: float,
    output_lines_path = "",
    row_idx_col=OutputConfig.DEFAULT_ROW_IDX_COL,
    sub_idx_col=OutputConfig.DEFAULT_SUB_IDX_COL,
    chainage_col=OutputConfig.DEFAULT_CHAINAGE_COL,
    max_distance_factor=Config.MAX_DISTANCE_FACTOR,
    engine=Config.DEFAULT_GPD_ENGINE,
    driver=Config.DEFAULT_GPD_DRIVER
):
    """
    Fasst Punkte zu Linien zusammen, indem:
      - Gruppierung nach (row_idx_col, sub_idx_col, range_col)
      - Sortierung jeder Gruppe nach 'chainage_col'
      - Erstellung eines LineString für jede zusammenhängende Gruppe
      - Hinzufügen von Übergangslinien zwischen Gruppen mit unterschiedlichen 'dist_range'-Werten

    Jedes resultierende Liniensegment enthält seinen 'dist_range'-Wert. Übergangslinien
    werden mit dem minimalen 'dist_range'-Wert zwischen zwei benachbarten Gruppen gekennzeichnet.

    :param input_points_shp: Pfad zum Eingabe-Punkte-Shapefile (muss row_idx, sub_idx, chainage, dist_range enthalten).
    :param output_lines_shp: Pfad zum Speichern der zusammengeführten Linien.
    :param row_idx_col: Spalte zur Identifizierung der ursprünglichen Zeile im Linien-Shapefile.
    :param sub_idx_col: Spalte zur Identifizierung des Unterzeilenindex (für MultiLineStrings).
    :param chainage_col: Spalte zur Speicherung der Entfernung entlang der Unterzeile (0..Linienlänge).
    :param range_col: Spalte zur Speicherung des `dist_range`- oder Klassifizierungsattributs.
    :param discretization_distance: Erwarteter Abstand zwischen aufeinanderfolgenden Punkten.
    :param max_distance_factor: Maximal zulässiger Abstand zwischen aufeinanderfolgenden Punkten,
                                als Vielfaches von `discretization_distance`. Darüber hinausgehende
                                Abstände werden als ungültig betrachtet.
    :param engine: GeoPandas-Engine zum Lesen/Schreiben von Dateien (aus GpdEngine-Enum).
    :param driver: GeoPandas-Treiber zum Schreiben von Dateien (aus GpdDriver-Enum).
    :return: GeoDataFrame der zusammengeführten Linien mit Spalten: [row_idx, sub_idx, dist_range, geometry].
    """

    gdf_points = input_points_shp

    if gdf_points.empty:
        logger.error(LoggingMessages.INPUT_EMPTY)
        return gpd.GeoDataFrame()

    # Überprüfe erforderliche Spalten
    required_columns = [row_idx_col, sub_idx_col, chainage_col, range_col]
    for req_col in required_columns:
        if req_col not in gdf_points.columns:
            raise ValueError(LoggingMessages.MISSING_COLUMN.format(req_col))

    # Maximal zulässiger Abstand zwischen aufeinanderfolgenden Punkten
    max_distance = discretization_distance * max_distance_factor

    # Hier werden die resultierenden Linien gespeichert
    all_lines = []
    all_row_idx = []
    all_sub_idx = []
    all_range = []

    # Gruppiere nach (row_idx, sub_idx)
    grouped = gdf_points.groupby([row_idx_col, sub_idx_col])

    for (r_i, s_i), group_df in grouped:
        # Sortiere Punkte in dieser Gruppe nach chainage
        group_sorted = group_df.sort_values(by=chainage_col)

        # Iteriere über aufeinanderfolgende Punkte, um Linien zu erstellen
        segment_coords = []
        current_range = None
        last_point = None

        for idx, row in group_sorted.iterrows():
            pt = row.geometry
            this_range = row[range_col]

            # Wenn ein neues Segment beginnt
            if current_range is None:
                current_range = this_range
                segment_coords = [(pt.x, pt.y)]
                last_point = pt
                continue

            # Überprüfe den Abstand zwischen aufeinanderfolgenden Punkten
            dist = sqrt((pt.x - last_point.x)**2 + (pt.y - last_point.y)**2)
            if dist > max_distance:
                # Beende das aktuelle Segment als LineString
                if len(segment_coords) > 1:
                    line = LineString(segment_coords)
                    all_lines.append(line)
                    all_row_idx.append(r_i)
                    all_sub_idx.append(s_i)
                    all_range.append(current_range)

                # Starte ein neues Segment
                segment_coords = [(pt.x, pt.y)]
                current_range = this_range
            else:
                # Füge den Punkt zum aktuellen Segment hinzu
                segment_coords.append((pt.x, pt.y))
                last_point = pt

            # Wenn sich der `dist_range`-Wert ändert, beende das aktuelle Segment und erstelle einen Übergang
            if this_range != current_range:
                # Beende das aktuelle Segment
                if len(segment_coords) > 1:
                    line = LineString(segment_coords)
                    all_lines.append(line)
                    all_row_idx.append(r_i)
                    all_sub_idx.append(s_i)
                    all_range.append(current_range)

                # Erstelle eine Übergangslinie
                if last_point is not None:
                    transition_line = LineString([segment_coords[-1], (pt.x, pt.y)])
                    transition_range = min(current_range, this_range)
                    all_lines.append(transition_line)
                    all_row_idx.append(r_i)
                    all_sub_idx.append(s_i)
                    all_range.append(transition_range)

                # Starte ein neues Segment
                segment_coords = [(pt.x, pt.y)]
                current_range = this_range

        # Beende das letzte Segment
        if len(segment_coords) > 1:
            line = LineString(segment_coords)
            all_lines.append(line)
            all_row_idx.append(r_i)
            all_sub_idx.append(s_i)
            all_range.append(current_range)


    gdf_out = gpd.GeoDataFrame({
        row_idx_col: all_row_idx,
        sub_idx_col: all_sub_idx,
        range_col: all_range
    }, geometry=all_lines, crs=gdf_points.crs)

    logger.info(LoggingMessages.CREATING_LINES.format(len(gdf_out)))


    if output_lines_path:
        gdf_out.to_file(output_lines_path, engine=engine, driver=driver)
        logger.info(LoggingMessages.SAVING_OUTPUT.format(output_lines_path))

    return gdf_out