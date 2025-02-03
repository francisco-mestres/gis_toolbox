import geopandas as gpd
import numpy as np
import pandas as pd
from numba import cuda
import math
import logging
from typing import List
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString

from gis_toolbox.enums import GeometryType, GpdDriver, GpdEngine


####### EINSTELLUNGEN #######

class Config:
    DEFAULT_IO_ENGINE = GpdEngine.PYOGRIO.value
    DEFAULT_DRIVER = GpdDriver.ESRI_SHAPEFILE.value
    DEFAULT_BATCH_SIZE = 100_000
    DEFAULT_SPACING = 10.0
    DEFAULT_THREADS_PER_BLOCK = 512

    LOG_FILE_PATH = "distance_calculation.log"


class LoggingMessages:
    """
    Diese Klasse speichert verschiedene Logging-Nachrichten als Konstanten,
    damit sie zentral verwaltet werden können.
    """
    # Allgemeine Meldungen
    POINT_LAYER_EMPTY = "Punkt-Layer ist leer. Bitte stellen Sie sicher, dass die Eingabe korrekt ist."
    COMBINING_POLYGON_BOUNDARIES = "Kombiniere Polygon-Grenzen und Linien."
    NO_LINE_SEGMENTS_WARNING = "Keine Liniensegmente oder Polygon-Grenzen gefunden. Weise NaN für Abstände zu."
    SKIPPING_UNSUPPORTED_GEOMETRY = "Überspringe nicht unterstützten Geometrietyp: {}."
    DISTANCE_CALCULATION_COMPLETED = "Abstandsberechnung erfolgreich abgeschlossen."
    STARTING_DISTANCE_CALCULATION = "Starte Distanzberechnung..."
    ERROR_OCCURRED = "Während der Distanzberechnung ist ein Fehler aufgetreten: {}"

    # Meldungen für GPU-Prozesse
    CHECKING_POINTS_INSIDE_POLYGONS = "Prüfe, ob Punkte in Polygonen liegen (GPU)."
    CALCULATING_DISTANCES = "Berechne Abstände zu Polygon-Grenzen und Linien (GPU)."

class TqdmConfig:
    COMPUTE_DISTANCE_DESC = "Distanzberechnung"  # Beschreibung für die Fortschrittsanzeige
    COMPUTE_DISTANCE_UNIT = "Punkt"  # Einheit für die Fortschrittsanzeige

class OutputConfig:
    DEFAULT_DISTANCE_LABEL = "Entfernung"
    DEFAULT_INSIDE_LABEL = "Innerhalb"


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


####### CUDA KERNEL #######

@cuda.jit(device=True)
def d_point_in_polygon(px, py, polygon_x, polygon_y, start_idx, end_idx):
    """
    Überprüft mittels Ray-Casting, ob der Punkt (px, py) innerhalb des 
    Polygons liegt, das von polygon_x/ polygon_y repräsentiert wird.
    """
    inside = False
    j = end_idx - 1

    for i in range(start_idx, end_idx):

        xi = polygon_x[i]
        yi = polygon_y[i]
        xj = polygon_x[j]
        yj = polygon_y[j]

        # Ray-Casting-Bedingung
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-10) + xi):
            
            inside = not inside

        j = i

    return inside


@cuda.jit
def point_in_polygon_kernel(
    points_x, points_y,
    polygon_x, polygon_y,
    polygon_starts, polygon_ends,
    inside_flags
):
    """
    Bestimmt für jeden Punkt, ob er sich innerhalb eines beliebigen Polygons befindet.
    """
    idx = cuda.grid(1)
    if idx >= points_x.size:
        return

    px = points_x[idx]
    py = points_y[idx]

    inside = False
    for poly_idx in range(polygon_starts.size):
        if d_point_in_polygon(
            px, py,
            polygon_x, polygon_y,
            polygon_starts[poly_idx],
            polygon_ends[poly_idx]
        ):
            inside = True
            break
    inside_flags[idx] = inside

@cuda.jit(device=True)
def d_point_on_line(px, py, x1, y1, x2, y2, epsilon=1e-6):
    """
    Überprüft, ob der Punkt (px, py) auf dem Liniensegment (x1, y1)-(x2, y2) liegt.
    """
    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross) > epsilon:
        return False
    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= 0

@cuda.jit(device=True)
def d_point_to_segment_distance(px, py, x1, y1, x2, y2):
    """
    Berechnet den kürzesten Abstand von einem Punkt zu einem Liniensegment.
    """
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1

    c1 = vx * wx + vy * wy
    c2 = vx * vx + vy * vy

    if c2 == 0:
        # Das Liniensegment ist in Wirklichkeit nur ein einzelner Punkt
        return math.hypot(px - x1, py - y1)

    b = c1 / c2
    if b < 0:
        # Nächster Punkt ist das Anfangsende des Segments
        return math.hypot(px - x1, py - y1)
    elif b > 1:
        # Nächster Punkt ist das Endende des Segments
        return math.hypot(px - x2, py - y2)
    else:
        # Nächster Punkt liegt im Inneren des Segments
        proj_x = x1 + b * vx
        proj_y = y1 + b * vy
        return math.hypot(px - proj_x, py - proj_y)

@cuda.jit
def calculate_distances_kernel(
    points_x, points_y,
    line_x, line_y,
    line_indices, cell_starts, cell_ends,
    grid_dim_x, grid_dim_y,
    cell_size_x, cell_size_y,
    x_min, y_min,
    distances
):
    """
    Berechnet den kürzesten Abstand jedes Punktes zu den Liniensegmenten 
    (Polygongrenzen und Linien) anhand eines räumlichen Rasters (Spatial Index).
    """
    idx = cuda.grid(1)
    if idx >= points_x.size:
        return

    # Punkt-Koordinaten
    px = points_x[idx]
    py = points_y[idx]
    min_dist = 1e20

    # Falls keine Linien vorhanden sind
    if line_indices.size == 0:
        distances[idx] = math.nan
        return

    # Bestimme Raster-Zelle, in der sich der Punkt befindet
    cell_x = int((px - x_min) / cell_size_x)
    cell_y = int((py - y_min) / cell_size_y)

    # Sicherstellen, dass die Raster-Zellenindizes nicht außerhalb liegen
    cell_x = max(0, min(cell_x, grid_dim_x - 1))
    cell_y = max(0, min(cell_y, grid_dim_y - 1))

    # Untersuche Nachbarzellen (3x3 Umgebung)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx = cell_x + dx
            ny = cell_y + dy

            if 0 <= nx < grid_dim_x and 0 <= ny < grid_dim_y:
                cell_idx = ny * grid_dim_x + nx
                start = cell_starts[cell_idx]
                end = cell_ends[cell_idx]

                # Berechne Abstand des Punktes zu allen Segmenten in dieser Raster-Zelle
                for i in range(start, end):
                    line_idx = line_indices[i]

                    # Endpunkte des Segments
                    x1 = line_x[2 * line_idx]
                    y1 = line_y[2 * line_idx]
                    x2 = line_x[2 * line_idx + 1]
                    y2 = line_y[2 * line_idx + 1]

                    # Vektor-Komponenten
                    vx = x2 - x1
                    vy = y2 - y1
                    length_sq = vx * vx + vy * vy

                    if length_sq > 0:
                        # Parameter t für die Projektion
                        t = ((px - x1) * vx + (py - y1) * vy) / length_sq

                        if t < 0:
                            # Nächste Stelle: Segment-Anfang
                            dist = math.hypot(px - x1, py - y1)
                        elif t > 1:
                            # Nächste Stelle: Segment-Ende
                            dist = math.hypot(px - x2, py - y2)
                        else:
                            # Projektion liegt innerhalb des Segments
                            proj_x = x1 + t * vx
                            proj_y = y1 + t * vy
                            dist = math.hypot(px - proj_x, py - proj_y)

                        min_dist = min(min_dist, dist)

    # Endergebnis speichern
    distances[idx] = min_dist if min_dist < 1e20 else math.nan


####### HILFSFUNKTIONEN #######


def prepare_polygon_data(gdf_polygons: gpd.GeoDataFrame):
    """
    Wandelt Polygongeometrien in Koordinatenarrays um, die für GPU-Berechnungen
    geeignet sind (Format: polygon_x, polygon_y, polygon_starts, polygon_ends).
    """
    polygon_x = []
    polygon_y = []
    polygon_starts = []
    polygon_ends = []
    current_index = 0

    for geom in gdf_polygons.geometry:
        if geom.is_empty:
            continue

        # Stelle sicher, dass nur Polygon- oder MultiPolygon-Geometrien verarbeitet werden
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            # Nicht unterstützte Geometrietypen werden übersprungen
            logger.warning(LoggingMessages.SKIPPING_UNSUPPORTED_GEOMETRY.format(geom.geom_type))
            continue

        for poly in polys:
            # Außenring (exterior)
            coords = np.array(poly.exterior.coords)
            num_coords = coords.shape[0]
            polygon_starts.append(current_index)
            polygon_x.extend(coords[:, 0])
            polygon_y.extend(coords[:, 1])
            current_index += num_coords
            polygon_ends.append(current_index)

            # Innenringe (Holes)
            for interior in poly.interiors:
                coords = np.array(interior.coords)
                num_coords = coords.shape[0]
                polygon_starts.append(current_index)
                polygon_x.extend(coords[:, 0])
                polygon_y.extend(coords[:, 1])
                current_index += num_coords
                polygon_ends.append(current_index)

    # Konvertiere Listen in numpy-Arrays
    polygon_x = np.array(polygon_x, dtype=np.float32)
    polygon_y = np.array(polygon_y, dtype=np.float32)
    polygon_starts = np.array(polygon_starts, dtype=np.int32)
    polygon_ends = np.array(polygon_ends, dtype=np.int32)

    return polygon_x, polygon_y, polygon_starts, polygon_ends

def extract_line_segments(gdf_lines: gpd.GeoDataFrame, gdf_polygons):
    """
    Extrahiert Liniensegmente aus Linienobjekten und Polygon-Grenzen.
    """
    line_coords = []

    # Linien verarbeiten, falls vorhanden
    if gdf_lines is not None:
        for geom in gdf_lines.geometry:
            if geom.is_empty:
                continue
            # Unterscheide zwischen Single- und MultiLineString
            lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]
            for line in lines:
                coords = np.array(line.coords)
                for i in range(len(coords) - 1):
                    line_coords.append([coords[i, 0], coords[i, 1], coords[i+1, 0], coords[i+1, 1]])

    # Polygon-Grenzen extrahieren
    for geom in gdf_polygons.geometry:
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
            for poly in polys:
                # Außenring
                coords = np.array(poly.exterior.coords)
                for i in range(len(coords) - 1):
                    line_coords.append([coords[i, 0], coords[i, 1], coords[i+1, 0], coords[i+1, 1]])

                # Innenringe
                for interior in poly.interiors:
                    coords = np.array(interior.coords)
                    for i in range(len(coords) - 1):
                        line_coords.append([coords[i, 0], coords[i, 1], coords[i+1, 0], coords[i+1, 1]])
        else:
            logger.warning(LoggingMessages.SKIPPING_UNSUPPORTED_GEOMETRY.format(geom.geom_type))

    if not line_coords:
        return None

    return np.array(line_coords, dtype=np.float32)


def build_spatial_index(line_coords, bounds, num_points):
    """
    Erstellt einen einfachen Gitter-basierten (Grid) Spatial Index, 
    um die Abstandsberechnungen zu beschleunigen.
    """
    x_min, y_min, x_max, y_max = bounds
    # Etwas Puffer hinzufügen
    padding = 1.0
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Eine einfache Heuristik, um die Gittergröße zu bestimmen
    grid_dim = int(math.sqrt(num_points))
    grid_dim_x = max(16, min(256, grid_dim))
    grid_dim_y = grid_dim_x
    cell_size_x = x_range / grid_dim_x
    cell_size_y = y_range / grid_dim_y
    total_cells = grid_dim_x * grid_dim_y

    cell_line_indices = [[] for _ in range(total_cells)]
    for seg_idx, (x1, y1, x2, y2) in enumerate(line_coords):
        # Begrenzendes Rechteck des Segments
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        # Bestimme, in welchem Zellbereich das Segment liegt
        cell_x_min = int((min_x - x_min) / cell_size_x)
        cell_x_max = int((max_x - x_min) / cell_size_x)
        cell_y_min = int((min_y - y_min) / cell_size_y)
        cell_y_max = int((max_y - y_min) / cell_size_y)

        # Clampen der Zellindizes (Randbehandlung)
        cell_x_min = max(0, min(cell_x_min, grid_dim_x - 1))
        cell_x_max = max(0, min(cell_x_max, grid_dim_x - 1))
        cell_y_min = max(0, min(cell_y_min, grid_dim_y - 1))
        cell_y_max = max(0, min(cell_y_max, grid_dim_y - 1))

        # Segment in alle relevanten Zellen eintragen
        for cy in range(cell_y_min, cell_y_max + 1):
            for cx in range(cell_x_min, cell_x_max + 1):
                cell_idx = cy * grid_dim_x + cx
                cell_line_indices[cell_idx].append(seg_idx)

    cell_starts = np.zeros(total_cells, dtype=np.int32)
    cell_ends = np.zeros(total_cells, dtype=np.int32)
    line_indices = []
    position = 0
    for idx in range(total_cells):
        indices = cell_line_indices[idx]
        cell_starts[idx] = position
        line_indices.extend(indices)
        position += len(indices)
        cell_ends[idx] = position

    index_data = {
        'line_indices': np.ascontiguousarray(line_indices, dtype=np.int32),
        'cell_starts': cell_starts,
        'cell_ends': cell_ends,
        'x_min': np.float64(x_min),
        'y_min': np.float64(y_min),
        'grid_dim_x': grid_dim_x,
        'grid_dim_y': grid_dim_y,
        'cell_size_x': np.float64(cell_size_x),
        'cell_size_y': np.float64(cell_size_y)
    }

    return index_data


####### HAUPTFUNKTION #######


def compute_distances_points_to_polygons(
    point_layer: gpd.GeoDataFrame,
    polygon_layers: List[gpd.GeoDataFrame],
    output_label=OutputConfig.DEFAULT_DISTANCE_LABEL,
    inside_label=OutputConfig.DEFAULT_INSIDE_LABEL,
    output_layer="",
    batch_size=Config.DEFAULT_BATCH_SIZE,
    engine=Config.DEFAULT_IO_ENGINE,
    threads_per_block=Config.DEFAULT_THREADS_PER_BLOCK
):
    """
    Berechnet die Abstände von Punkten zur nächsten Polygon-Grenze bzw. Linie
    mithilfe von GPU-Beschleunigung (Numba CUDA).
    
    :param point_layer: Pfad zu den Punktdaten (oder GeoDataFrame).
    :param polygon_layers: Liste von Pfaden zu Polygon-Dateien oder polygon-/linienhaltigen Dateien.
    :param output_layer: Pfad zur Ausgabedatei, in die das Ergebnis geschrieben wird.
    :param output_label: Name des Attributs, in dem der Abstand gespeichert wird.
    :param inside_label: Name des Attributs, das kennzeichnet, ob ein Punkt innerhalb eines Polygons liegt.
    :param batch_size: Größe des Pakets, das in den Spatial-Index einfließt (Heuristik).
    :return: Ein GeoDataFrame mit den Ergebnissen.
    """
    logger.info(LoggingMessages.STARTING_DISTANCE_CALCULATION)

    from gis_toolbox.utils import write_gdf_in_chunks

    try:

        gdf_points = point_layer

        if gdf_points.empty:
            raise ValueError(LoggingMessages.POINT_LAYER_EMPTY)

        # Polygone und Linien aus den Eingabe-Layern extrahieren
        combined_polygons = []
        combined_lines = []
        for layer in tqdm(polygon_layers, desc=TqdmConfig.COMPUTE_DISTANCE_DESC, unit=TqdmConfig.COMPUTE_DISTANCE_UNIT):
            
            if layer.crs != gdf_points.crs:
                layer = layer.to_crs(gdf_points.crs)

            polygon_mask = layer.geometry.apply(lambda g: isinstance(g, (Polygon, MultiPolygon)))
            if polygon_mask.any():
                combined_polygons.append(layer[polygon_mask])

            line_mask = layer.geometry.apply(lambda g: isinstance(g, (LineString, MultiLineString)))
            if line_mask.any():
                combined_lines.append(layer[line_mask])


        # Kombinierte Polygone
        if combined_polygons:
            gdf_polygons = gpd.GeoDataFrame(pd.concat(combined_polygons, ignore_index=True), crs=gdf_points.crs)
        else:
            gdf_polygons = gpd.GeoDataFrame(columns=['geometry'], crs=gdf_points.crs)

        # Kombinierte Linien
        if combined_lines:
            gdf_lines = gpd.GeoDataFrame(pd.concat(combined_lines, ignore_index=True), crs=gdf_points.crs)
        else:
            gdf_lines = gpd.GeoDataFrame(columns=['geometry'], crs=gdf_points.crs)

        logger.info(LoggingMessages.COMBINING_POLYGON_BOUNDARIES)
        # Nur zum Bound-Bestimmen (Alternative: extrahierte line_coords nutzen)
        polygon_boundaries = gdf_polygons.boundary
        combined_line_geometries = pd.concat([polygon_boundaries, gdf_lines.geometry], ignore_index=True)
        combined_line_geometries_mask = combined_line_geometries.apply(lambda g: isinstance(g, (LineString, MultiLineString)))
        combined_line_geometries = combined_line_geometries[combined_line_geometries_mask]

        # Liniensegmente extrahieren
        line_coords = extract_line_segments(gdf_lines, gdf_polygons)
        if line_coords is None or len(line_coords) == 0:
            logger.warning(LoggingMessages.NO_LINE_SEGMENTS_WARNING)
            gdf_points[output_label] = np.nan
            if output_layer:
                gdf_points.to_file(output_layer, driver=Config.DEFAULT_DRIVER, engine=engine)
            return

        # Linien-Koordinaten für die GPU aufbereiten
        line_x = np.ascontiguousarray(line_coords[:, [0, 2]].flatten(), dtype=np.float32)
        line_y = np.ascontiguousarray(line_coords[:, [1, 3]].flatten(), dtype=np.float32)
        bounds = combined_line_geometries.total_bounds
        index_data = build_spatial_index(line_coords, bounds, batch_size)

        # Transfer der Linien-/Index-Daten auf die GPU
        d_line_x = cuda.to_device(line_x)
        d_line_y = cuda.to_device(line_y)
        d_line_indices = cuda.to_device(index_data['line_indices'])
        d_cell_starts = cuda.to_device(index_data['cell_starts'])
        d_cell_ends = cuda.to_device(index_data['cell_ends'])

        # Extrahieren der Punkt-Koordinaten
        points_x = np.ascontiguousarray(gdf_points.geometry.x.values, dtype=np.float32)
        points_y = np.ascontiguousarray(gdf_points.geometry.y.values, dtype=np.float32)
        num_points = points_x.size

        d_points_x = cuda.to_device(points_x)
        d_points_y = cuda.to_device(points_y)
        d_distances = cuda.device_array(num_points, dtype=np.float32)
        d_inside_flags = cuda.device_array(num_points, dtype=np.int32)

        # Konfiguration der CUDA-Kernel

        blockspergrid = (num_points + (threads_per_block - 1)) // threads_per_block

        # Schritt 1: Punkt-in-Polygon-Test
        if not gdf_polygons.empty:
            logger.info(LoggingMessages.CHECKING_POINTS_INSIDE_POLYGONS)
            polygon_x, polygon_y, polygon_starts, polygon_ends = prepare_polygon_data(gdf_polygons)
            d_polygon_x = cuda.to_device(polygon_x)
            d_polygon_y = cuda.to_device(polygon_y)
            d_polygon_starts = cuda.to_device(polygon_starts)
            d_polygon_ends = cuda.to_device(polygon_ends)

            point_in_polygon_kernel[blockspergrid, threads_per_block](
                d_points_x, d_points_y,
                d_polygon_x, d_polygon_y,
                d_polygon_starts, d_polygon_ends,
                d_inside_flags
            )
            cuda.synchronize()

        # Schritt 2: Distanzberechnung zu Polygon-Grenzen und Linien
        logger.info(LoggingMessages.CALCULATING_DISTANCES)
        calculate_distances_kernel[blockspergrid, threads_per_block](
            d_points_x, d_points_y,
            d_line_x, d_line_y,
            d_line_indices, d_cell_starts, d_cell_ends,
            index_data['grid_dim_x'], index_data['grid_dim_y'],
            index_data['cell_size_x'], index_data['cell_size_y'],
            index_data['x_min'], index_data['y_min'],
            d_distances
        )
        cuda.synchronize()

        # Ergebnisse vom GPU-Speicher abziehen
        distances = d_distances.copy_to_host()
        inside_flags = d_inside_flags.copy_to_host()

        # Für alle Punkte, die innerhalb von Polygonen liegen, Distance = -1
        distances[inside_flags == 1] = -1

        # Ergebnisse in GeoDataFrame einfügen
        gdf_points[inside_label] = inside_flags
        gdf_points[output_label] = distances
        logger.info(LoggingMessages.DISTANCE_CALCULATION_COMPLETED)

        # Ausgabe speichern
        if output_layer:
            write_gdf_in_chunks(gdf_points, output_layer)

    except Exception as e:
        logger.exception(LoggingMessages.ERROR_OCCURRED.format(e))
        raise e

    return gdf_points

