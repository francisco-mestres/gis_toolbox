import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from numba import cuda, int32, float64
import logging

from enums import GpdEngine, GpdDriver  # Importiere Enums


class Config:
    MAX_DISTANCE_FACTOR = 1.8
    DEFAULT_GPD_DRIVER = GpdDriver.ESRI_SHAPEFILE.value
    DEFAULT_GPD_ENGINE = GpdEngine.PYOGRIO.value
    DEFUALT_THREADS_PER_BLOCK = 256
    DEFAULT_TINY_FACTOR = 0.01
    LARGE_BYTES_USAGE = 4_000_000_000  # 4 GB

    LOG_FILE_PATH = "discretize_lines.log"


class LoggingMessages:
    EMPTY_GDF = "Empty GeoDataFrame."
    NO_LINES = "No valid lines after preprocessing."
    NO_CHAINAGES = "No chainages => no points generated."
    LARGE_GPU_USAGE = "Estimated GPU usage ~ {:.2f} GB. Could be too large."


class OutputConfig:
    DEFAULT_ROW_IDX_COL = "row_idx"
    DEFAULT_SUB_IDX_COL = "sub_idx"
    DEFAULT_CHAINAGE_COL = "chainage"
    DEFAULT_NORMAL_X_COL = "normal_x"
    DEFAULT_NORMAL_Y_COL = "normal_y"


logger = logging.getLogger(__name__)



####### KERNEL #######

@cuda.jit(device=True)
def _binary_search_segment(prefix_sums, start_idx, end_idx, target):
    """
    Binary search in prefix_sums[start_idx:end_idx] to find
    the largest index k where prefix_sums[k] <= target.

    Returns k, start_idx <= k < end_idx.
    """
    left = start_idx
    right = end_idx - 1
    while left < right:
        mid = (left + right + 1) // 2
        if prefix_sums[mid] <= target:
            left = mid
        else:
            right = mid - 1
    return left


@cuda.jit
def interpolate_chainages_kernel(
    prefix_sums,          # (total_vertices,) float64
    global_coords,        # (total_vertices,2) float64
    line_vertex_start,    # (num_lines,) int32
    line_vertex_end,      # (num_lines,) int32
    line_first_prefix,    # (num_lines,) float64 => prefix_sums offset for first vertex
    line_lengths,         # (num_lines,) float64 => total length of each line
    line_info,            # (num_lines,2) int32 => (row_idx, sub_idx)
    chainage_offsets,     # (num_lines,) int32 => offset into chainages array
    chainages_global,     # (total_chainages,) float64
    tiny_distance,        # float64
    out_points,           # (total_chainages,2) float64
    out_normals,          # (total_chainages,2) float64
    out_actual_dist       # (total_chainages,) float64
):
    """
    Each thread processes exactly ONE chainage in chainages_global.
    Steps:
      1) Determine which line this chainage belongs to.
      2) Interpolate X,Y at that chainage (piecewise linear).
      3) Approximate normal by sampling chainage +/- tiny_distance.
    """

    idx = cuda.grid(1)
    total = chainages_global.shape[0]
    if idx >= total:
        return

    # 1) Find line index. We do a linear scan of chainage_offsets for demonstration.
    #    For large # lines, you'd do a more advanced approach or store a separate array
    #    that maps each chainage index -> line index. But let's keep it simple.

    num_lines = line_vertex_start.shape[0]
    line_i = 0
    for i in range(num_lines):
        start_ch_ofs = chainage_offsets[i]
        if i == num_lines - 1:
            end_ch_ofs = total
        else:
            end_ch_ofs = chainage_offsets[i+1]
        if start_ch_ofs <= idx < end_ch_ofs:
            line_i = i
            break

    # Retrieve line-specific data
    vstart = line_vertex_start[line_i]
    vend   = line_vertex_end[line_i]
    line_len = line_lengths[line_i]
    first_pref = line_first_prefix[line_i]  # prefix_sums[vstart]

    # chainage requested
    chainage = chainages_global[idx]
    # clamp
    if chainage < 0.0:
        chainage = 0.0
    elif chainage > line_len:
        chainage = line_len

    # local_target is in prefix_sums domain
    local_target = first_pref + chainage

    # 2) Binary search for segment
    seg_idx = _binary_search_segment(prefix_sums, vstart, vend, local_target)
    # Interpolate
    if seg_idx == vend - 1:
        px = global_coords[seg_idx,0]
        py = global_coords[seg_idx,1]
    else:
        distA = prefix_sums[seg_idx]
        distB = prefix_sums[seg_idx+1]
        denom = distB - distA
        if denom < 1e-12:
            # degenerate
            px = global_coords[seg_idx,0]
            py = global_coords[seg_idx,1]
        else:
            t = (local_target - distA)/denom
            xA = global_coords[seg_idx,0]
            yA = global_coords[seg_idx,1]
            xB = global_coords[seg_idx+1,0]
            yB = global_coords[seg_idx+1,1]
            px = xA + t*(xB - xA)
            py = yA + t*(yB - yA)

    # store point & dist
    out_points[idx,0] = px
    out_points[idx,1] = py
    out_actual_dist[idx] = chainage

    # 3) Approx normal => sample chainage +/- tiny
    c_before = chainage - tiny_distance
    c_after  = chainage + tiny_distance
    if c_before < 0.0: c_before = 0.0
    if c_after  > line_len: c_after = line_len

    # function to get x,y for c
    def interp_dist(cval):
        loc_tar = first_pref + cval
        sidx = _binary_search_segment(prefix_sums, vstart, vend, loc_tar)
        if sidx == vend - 1:
            return (global_coords[sidx,0], global_coords[sidx,1])
        dA = prefix_sums[sidx]
        dB = prefix_sums[sidx+1]
        dd = dB - dA
        if dd < 1e-12:
            return (global_coords[sidx,0], global_coords[sidx,1])
        tt = (loc_tar - dA)/dd
        xxA = global_coords[sidx,0]
        yyA = global_coords[sidx,1]
        xxB = global_coords[sidx+1,0]
        yyB = global_coords[sidx+1,1]
        xx = xxA + tt*(xxB - xxA)
        yy = yyA + tt*(yyB - yyA)
        return (xx, yy)

    bx, by = interp_dist(c_before)
    ax, ay = interp_dist(c_after)
    tx = ax - bx
    ty = ay - by
    mag = math.sqrt(tx*tx + ty*ty)
    if mag < 1e-12:
        nx = 0.0
        ny = 0.0
    else:
        tx /= mag
        ty /= mag
        nx = -ty
        ny =  tx
    out_normals[idx,0] = nx
    out_normals[idx,1] = ny





####### LINE PRE-PROCESSOR #######

class LinePreprocessor:
    """
    Responsible for extracting lines from a GeoDataFrame, expanding
    MultiLineStrings, building coordinate arrays and prefix sums.
    """
    def __init__(self, gdf):
        self.gdf = gdf
        self.coords_list = []      # list of np arrays of shape (n,2)
        self.prefix_sums_list = [] # matching list of prefix sums
        self.line_info = []        # (row_idx, sub_idx)
        self.line_lengths = []     # total length of each line
        self.line_vertex_start = []
        self.line_vertex_end = []
        self.line_first_prefix = []

        self.total_vertices = 0
        self.num_lines = 0

    def run(self):
        if self.gdf.empty:
            logger.warning(LoggingMessages.EMPTY_GDF)
            return

        current_vertex_pos = 0
        for row_i, geom in enumerate(self.gdf.geometry):
            if geom.is_empty:
                continue
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                # skip unsupported
                continue

            for sub_i, line in enumerate(lines):
                if line.is_empty:
                    continue
                coords_np = np.array(line.coords, dtype=np.float64)
                if coords_np.shape[0] < 2:
                    continue
                p_sums = self._compute_prefix_sums(coords_np)
                length = p_sums[-1]
                if length < 1e-12:
                    continue

                self.coords_list.append(coords_np)
                self.prefix_sums_list.append(p_sums)
                self.line_info.append((row_i, sub_i))
                self.line_lengths.append(length)

                self.line_vertex_start.append(current_vertex_pos)
                vsz = coords_np.shape[0]
                self.line_vertex_end.append(current_vertex_pos + vsz)
                # the prefix at the first vertex
                first_pref = 0.0
                if current_vertex_pos > 0:
                    # We'll make them strictly increasing by offsetting
                    first_pref = self._last_prefix_val() 
                # offset this line's prefix sums
                # so prefix_sums for line i goes from [first_pref..(first_pref+length)]
                # We'll store the "base" (first_pref) separately,
                # then in the final array we do prefix_sums + base.
                self.line_first_prefix.append(first_pref)

                current_vertex_pos += vsz
                self.num_lines += 1

        self.total_vertices = current_vertex_pos

    def _compute_prefix_sums(self, coords):
        """
        For an array coords of shape (n,2),
        prefix_sums[i] = distance along line from coords[0] to coords[i].
        prefix_sums[0] = 0.
        """
        n = coords.shape[0]
        out = np.zeros(n, dtype=np.float64)
        dist_acc = 0.0
        for i in range(1, n):
            dx = coords[i,0] - coords[i-1,0]
            dy = coords[i,1] - coords[i-1,1]
            seg_len = math.sqrt(dx*dx + dy*dy)
            dist_acc += seg_len
            out[i] = dist_acc
        return out

    def _last_prefix_val(self):
        """
        Return the last prefix value in the global sense.
        i.e. the prefix sum at the final vertex from the previous line.
        """
        if not self.prefix_sums_list:
            return 0.0
        # the last line's prefix sums
        p_sums = self.prefix_sums_list[-1]
        base = self.line_first_prefix[-1]
        return base + p_sums[-1]


####### CHAINAGE BUILDER #######
class ChainageBuilder:
    """
    Builds arrays of chainages for each line: [0, d, 2d, ... line_length].
    Also accumulates total chainages to build a global array of chainages.
    """
    def __init__(self, line_lengths, distance=10.0):
        self.line_lengths = line_lengths
        self.distance = distance
        self.all_chainages = []
        self.chainage_offsets = []
        self.total_chainages = 0

    def run(self):
        """
        Populate all_chainages[i] for line i,
        then build chainage_offsets to flatten them into a single array.
        """
        total = 0
        for length in self.line_lengths:
            c = self._build_for_line(length)
            self.all_chainages.append(c)
            self.chainage_offsets.append(total)
            total += len(c)
        self.total_chainages = total

    def _build_for_line(self, length):
        if length < 1e-12:
            return np.array([], dtype=np.float64)
        c = np.arange(0, length, self.distance, dtype=np.float64)
        if len(c) == 0:
            c = np.array([0.0], dtype=np.float64)
        if c[-1] < length:
            c = np.append(c, length)
        return c
    

####### GPU DISCRETIZER #######
class GPUDiscretizer:
    """
    Takes the preprocessed lines (coords, prefix sums) and the chainage arrays,
    then allocates GPU memory, runs the kernel, and returns the final arrays.
    """

    def __init__(self, 
                 coords_list, prefix_sums_list, line_vertex_start, line_vertex_end,
                 line_first_prefix, line_lengths, line_info,
                 chainages_global, chainage_offsets,
                 threads_per_block=Config.DEFUALT_THREADS_PER_BLOCK, tiny_factor=Config.DEFAULT_TINY_FACTOR):
        self.coords_list = coords_list
        self.prefix_sums_list = prefix_sums_list
        self.line_vertex_start = line_vertex_start
        self.line_vertex_end = line_vertex_end
        self.line_first_prefix = line_first_prefix
        self.line_lengths = line_lengths
        self.line_info = line_info
        self.chainages_global = chainages_global
        self.chainage_offsets = chainage_offsets

        self.threads_per_block = threads_per_block
        self.tiny_factor = tiny_factor

        # We'll build the big unified arrays
        self.global_coords = None  # shape (total_vertices,2)
        self.global_prefixes = None  # shape (total_vertices,)
        self.num_lines = len(line_info)
        self.total_vertices = 0
        self.total_chainages = len(chainages_global)
    
    def prepare_data(self):
        """
        Convert coords_list and prefix_sums_list into single big arrays,
        applying line_first_prefix offsets so they're strictly increasing.
        """
        self.total_vertices = sum(arr.shape[0] for arr in self.coords_list)
        self.global_coords = np.zeros((self.total_vertices, 2), dtype=np.float64)
        self.global_prefixes = np.zeros(self.total_vertices, dtype=np.float64)

        current_pos = 0
        for i, coords_np in enumerate(self.coords_list):
            n = coords_np.shape[0]
            # copy coords
            self.global_coords[current_pos:current_pos+n] = coords_np

            # offset prefix sums
            p_sums = self.prefix_sums_list[i]
            base = self.line_first_prefix[i]
            for k in range(n):
                self.global_prefixes[current_pos + k] = base + p_sums[k]

            current_pos += n

    def check_memory_requirements(self):
        """
        Basic memory check: if total_chainages or total_vertices is huge,
        we might exceed GPU memory. This is only a rough check.
        """
        # Rough estimate: each float64 is 8 bytes.
        # We'll be storing coords, prefix sums, chainages, output arrays, etc.
        # Let's do a naive sum:
        usage_bytes = (self.total_vertices * 3 * 8)  # coords(2), prefix(1)
        usage_bytes += (self.total_chainages * (1+2+2) * 8)  # chainage, out_points(2), out_normals(2)
        # This is not exact, but a ballpark.

        # For large usage, warn:
        if usage_bytes > Config.LARGE_BYTES_USAGE:
            logger.warning(LoggingMessages.LARGE_GPU_USAGE.format(usage_bytes / 1e9))
    
    def run_kernel(self):
        """
        Allocates device arrays, launches kernel, copies results back.
        Returns (h_points, h_normals, h_dists, line_indices).
        """
        # Device allocations
        d_prefix_sums = cuda.to_device(self.global_prefixes)
        d_coords      = cuda.to_device(self.global_coords)

        d_line_vstart = cuda.to_device(np.array(self.line_vertex_start, dtype=np.int32))
        d_line_vend   = cuda.to_device(np.array(self.line_vertex_end,   dtype=np.int32))
        d_line_firstp = cuda.to_device(np.array(self.line_first_prefix, dtype=np.float64))
        d_line_lengths= cuda.to_device(np.array(self.line_lengths,      dtype=np.float64))
        d_line_info   = cuda.to_device(np.array(self.line_info,         dtype=np.int32))

        d_chain_offsets = cuda.to_device(np.array(self.chainage_offsets, dtype=np.int32))
        d_chainages     = cuda.to_device(self.chainages_global)

        d_out_points  = cuda.device_array((self.total_chainages,2), dtype=np.float64)
        d_out_normals = cuda.device_array((self.total_chainages,2), dtype=np.float64)
        d_out_dists   = cuda.device_array((self.total_chainages,),   dtype=np.float64)

        # Kernel config
        blocks = (self.total_chainages + self.threads_per_block - 1)//self.threads_per_block
        tiny_distance = self.tiny_factor * (self.chainages_global[1] if len(self.chainages_global)>1 else 1.0)
        # ^ you might pick a different reference if chainages_global is sparse, or pass a separate param.

        # Launch
        interpolate_chainages_kernel[blocks, self.threads_per_block](
            d_prefix_sums,
            d_coords,
            d_line_vstart,
            d_line_vend,
            d_line_firstp,
            d_line_lengths,
            d_line_info,
            d_chain_offsets,
            d_chainages,
            tiny_distance,
            d_out_points,
            d_out_normals,
            d_out_dists
        )

        # Copy back
        h_pts    = d_out_points.copy_to_host()
        h_norms  = d_out_normals.copy_to_host()
        h_dists  = d_out_dists.copy_to_host()

        # We'll also replicate row/sub for each chainage, same approach as in the kernel.
        # For large # lines, do a better approach (like storing direct line_i map).
        row_idxs = np.zeros(self.total_chainages, dtype=np.int32)
        sub_idxs = np.zeros(self.total_chainages, dtype=np.int32)
        for i in range(self.num_lines):
            start_ofs = self.chainage_offsets[i]
            if i == self.num_lines - 1:
                end_ofs = self.total_chainages
            else:
                end_ofs = self.chainage_offsets[i+1]
            (r_i, s_i) = self.line_info[i]
            row_idxs[start_ofs:end_ofs] = r_i
            sub_idxs[start_ofs:end_ofs] = s_i

        return (h_pts, h_norms, h_dists, row_idxs, sub_idxs)




####### MAIN FUNCTION #######
def discretize_lines_gpu_exact_spacing(
    gdf_lines: gpd.GeoDataFrame,
    distance: float,
    row_idx_col=OutputConfig.DEFAULT_ROW_IDX_COL,
    sub_idx_col=OutputConfig.DEFAULT_SUB_IDX_COL,
    chainage_col=OutputConfig.DEFAULT_CHAINAGE_COL,
    normal_x_col=OutputConfig.DEFAULT_NORMAL_X_COL,
    normal_y_col=OutputConfig.DEFAULT_NORMAL_Y_COL,
    threads_per_block=Config.DEFUALT_THREADS_PER_BLOCK,
    tiny_factor=Config.DEFAULT_TINY_FACTOR
):
    """
    Refactored "production-ish" pipeline:
     1) Preprocess lines -> coords, prefix sums.
     2) Build chainage arrays for each line => flatten.
     3) GPU discretizer => kernel => piecewise interpolation => normal approx.
     4) Build final GeoDataFrame with columns row_idx, sub_idx, chainage, normal_x, normal_y, geometry
    """

    if gdf_lines.empty:
        logger.warning(LoggingMessages.EMPTY_GDF)
        return gpd.GeoDataFrame()

    # 1) Preprocess
    pre = LinePreprocessor(gdf_lines)
    pre.run()
    if pre.num_lines == 0:
        logger.warning(LoggingMessages.NO_LINES)
        return gpd.GeoDataFrame()

    # 2) Chainage arrays
    cb = ChainageBuilder(pre.line_lengths, distance=distance)
    cb.run()
    total_chainages = cb.total_chainages
    if total_chainages == 0:
        logger.warning(LoggingMessages.NO_CHAINAGES)
        return gpd.GeoDataFrame()

    # Flatten chainages
    chainages_global = np.concatenate(cb.all_chainages) if len(cb.all_chainages)>0 else np.array([], dtype=np.float64)

    # 3) GPU discretizer
    disc = GPUDiscretizer(
        coords_list=pre.coords_list,
        prefix_sums_list=pre.prefix_sums_list,
        line_vertex_start=pre.line_vertex_start,
        line_vertex_end=pre.line_vertex_end,
        line_first_prefix=pre.line_first_prefix,
        line_lengths=pre.line_lengths,
        line_info=pre.line_info,
        chainages_global=chainages_global,
        chainage_offsets=np.array(cb.chainage_offsets, dtype=np.int32),
        threads_per_block=threads_per_block,
        tiny_factor=tiny_factor
    )
    disc.prepare_data()
    disc.check_memory_requirements()
    (h_pts, h_norms, h_dists, row_idxs, sub_idxs) = disc.run_kernel()

    # 4) Build final GeoDataFrame
    g_points = [Point(xy[0], xy[1]) for xy in h_pts]
    out_gdf = gpd.GeoDataFrame({
        row_idx_col: row_idxs,
        sub_idx_col: sub_idxs,
        chainage_col: h_dists,
        normal_x_col: h_norms[:,0],
        normal_y_col: h_norms[:,1]
    }, geometry=g_points, crs=gdf_lines.crs)

    return out_gdf
