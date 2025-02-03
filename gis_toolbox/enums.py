from enum import Enum


class GeometryType(Enum):
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOLYGON = "MultiPolygon"
    MULTILINESTRING = "MultiLineString"


class OSType(Enum):
    WINDOWS = "nt"
    UNIX = "posix"


class GpdEngine(Enum):
    FIONA = "fiona"
    PYOGRIO = "pyogrio"


class GpdDriver(Enum):
    ESRI_SHAPEFILE = "ESRI Shapefile"
    GEOJSON = "GeoJSON"
    GPKG = "GPKG"
    CSV = "CSV"
