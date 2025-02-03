import geopandas as gpd
import logging
import os


from modules.discretize_lines import discretize_lines_gpu_exact_spacing
from modules.compute_distance_to_polygon import compute_distances_points_to_polygons
from modules.classify_points_by_attr import classify_points_by_attribute
from modules.merge_nodes_into_lines import merge_points_into_lines


logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)

# Einstellungen des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/logfile.log"),
        logging.StreamHandler()
    ]
)

root_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root_folder, "../data")
input_folder = os.path.join(data_folder, "input")
output_folder = os.path.join(data_folder, "output")

input_lines_filename = "shape/autobahnen.shp"
input_lines_filepath = os.path.join(input_folder, input_lines_filename)

input_polygon_1_filename = "shape/stream_lines.shp"
input_polygon_1_filepath = os.path.join(input_folder, input_polygon_1_filename)

input_polygon_2_filename = "shape/stream_network.shp"
input_polygon_2_filepath = os.path.join(input_folder, input_polygon_2_filename)

output_filename = "shape/output.shp"
output_filepath = os.path.join(output_folder, output_filename)


def main():
    # Pfade zu den Eingabedateien
    input_lines_shp = input_lines_filepath
    input_polygons_shp = [input_polygon_2_filepath, input_polygon_1_filepath]
    output_discretized_points_shp = "output_discretized_points.shp"
    output_distances_shp = "output_distances.shp"
    output_classified_points_shp = "output_classified_points.shp"
    output_merged_lines_shp = "output_merged_lines_1.shp"

    # Schritt 1: Diskretisieren der Linien
    logging.info("Diskretisieren der Linien...")
    gdf_lines = gpd.read_file(input_lines_shp)
    gdf_discretized_points = discretize_lines_gpu_exact_spacing(gdf_lines, 2)
    gdf_discretized_points.to_file(output_discretized_points_shp)

    # Schritt 2: Berechnung der Abstände zu Polygonen
    logging.info("Berechnung der Abstände zu Polygonen...")
    gdf_polygons = [gpd.read_file(polygon) for polygon in input_polygons_shp]
    gdf_distances = compute_distances_points_to_polygons(gdf_discretized_points, gdf_polygons, "distance", output_distances_shp)

    # Schritt 3: Klassifizierung der Punkte basierend auf Attributen
    logging.info("Klassifizierung der Punkte basierend auf Attributen...")
    thresholds = [10, 25, 50, 100, 150, 200, 250, 500, 1000]  # Beispiel-Schwellenwerte
    input_attribute = "distance"
    output_range_attribute = "distance_range"
    gdf_classified_points = classify_points_by_attribute(gdf_distances, thresholds, input_attribute, output_range_attribute)

    # Schritt 4: Zusammenführen der Punkte zu Linien
    logging.info("Zusammenführen der Punkte zu Linien...")
    range_col = "distance_range"
    discretization_distance = 2
    gdf_merged_lines = merge_points_into_lines(gdf_classified_points, range_col, discretization_distance, output_merged_lines_shp)

    logging.info("Pipeline erfolgreich abgeschlossen.")

if __name__ == "__main__":
    main()