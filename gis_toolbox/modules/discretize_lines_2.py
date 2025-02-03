import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from numba import cuda
import logging

logger = logging.getLogger(__name__)

# ---------------------------
# GPU Kernel
# ---------------------------

@cuda.jit(device=True)
def _find_segment_index(prefix_sums, start_idx, end_idx, target):
    """
    Device function: binary search in prefix_sums[start_idx:end_idx]
    to find the largest prefix_sums[k] <= target.
    Returns k, where start_idx <= k < end_idx.
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
def interpolate_linestring_kernel(
    # Prefix sums of segment lengths for *all* lines combined
    prefix_sums,       # shape (total_vertices,)

    # XY coords for each vertex in all lines combined
    global_coords,     # shape (total_vertices, 2)

    # For each line, we store the range in [line_vertex_start, line_vertex_end)
    line_vertex_starts,  # shape (num_lines,)
    line_vertex_ends,    # shape (num_lines,)

    # For each line, we store the start index in chainages_global where
    # that line's chainages are. The next line starts in chainages_global
    # at chainage_offsets[i+1].
    chainage_offsets,  # shape (num_lines,) -> index in chainages_global

    # All chainages for *all* lines in a single array.
    chainages_global,  # shape (total_chainages,)

    # Additional info we want to carry per line
    line_info,         # shape (num_lines, 2) -> (row_idx, sub_idx)

    # The "tiny" offset for normal approximation
    tiny_distance,

    # OUTPUT arrays
    out_points,      # (total_chainages, 2)
    out_normals,     # (total_chainages, 2)
    out_actual_dist  # (total_chainages,)  (the chainage)
):
    """
    Each thread handles exactly ONE chainage in chainages_global.

    Steps:
    1) Identify which line this thread belongs to (by reverse-engineering offsets).
    2) Binary search in prefix_sums to find which segment the chainage falls in.
    3) Interpolate point at that chainage.
    4) For normal, sample chainage - tiny and chainage + tiny, clamp to [0, L].
       Interpolate those points, subtract => tangent => rotate => normal.
    """

    idx = cuda.grid(1)
    total = chainages_global.shape[0]
    if idx >= total:
        return

    chainage = chainages_global[idx]

    # First, we need to figure out which line we're dealing with.
    # We'll do a linear or binary search over chainage_offsets to find i
    # s.t. chainage belongs to line i.
    # But we also stored them in a structured manner so we might do a "reverse"
    # approach: Suppose we store line_index_for_chainage[idx] on CPU?
    # For simplicity, let's do a linear search here (inefficient for large num_lines).
    # A more advanced approach: build a prefix sum of chainage_counts and do a binary search.

    # For demonstration, let's do a naive linear approach:
    num_lines = line_vertex_starts.shape[0]
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

    # Now we know the line index is line_i.
    vstart = line_vertex_starts[line_i]
    vend   = line_vertex_ends[line_i]
    # prefix_sums in [vstart : vend], each prefix_sums[k] is the distance at vertex k (global index k).

    # 1) clamp chainage to [0, line_length]
    # line_length = prefix_sums[vend-1]  # distance at last vertex of the line
    line_length = prefix_sums[vend - 1] - prefix_sums[vstart] + (prefix_sums[vstart])  
    # Actually simpler: prefix_sums[vend-1] is the distance from "start_of_global_0" not from vstart.
    # We want local distance. We'll handle that carefully.

    # local_target = prefix_sums[vstart] + chainage
    # Because prefix_sums[vstart] is the "absolute" distance at the first vertex of the line
    # if lines are concatenated. 
    first_vertex_dist = prefix_sums[vstart]  # absolute distance of line's first vertex
    local_target = first_vertex_dist + chainage
    if chainage < 0.0:  # clamp
        chainage = 0.0
        local_target = first_vertex_dist
    if chainage > line_length:
        chainage = line_length
        local_target = first_vertex_dist + line_length

    # 2) binary search in prefix_sums[vstart : vend] for local_target
    seg_idx = _find_segment_index(prefix_sums, vstart, vend, local_target)
    if seg_idx == vend - 1:
        # chainage is at the last vertex
        px = global_coords[seg_idx, 0]
        py = global_coords[seg_idx, 1]
    else:
        # prefix_sums[seg_idx] <= local_target <= prefix_sums[seg_idx+1]
        distA = prefix_sums[seg_idx]
        distB = prefix_sums[seg_idx + 1]
        # fraction:
        denom = distB - distA
        if denom < 1e-12:
            # degenerate
            px = global_coords[seg_idx, 0]
            py = global_coords[seg_idx, 1]
        else:
            t = (local_target - distA) / denom
            xA = global_coords[seg_idx, 0]
            yA = global_coords[seg_idx, 1]
            xB = global_coords[seg_idx + 1, 0]
            yB = global_coords[seg_idx + 1, 1]
            px = xA + t*(xB - xA)
            py = yA + t*(yB - yA)

    # Store the final point
    out_points[idx, 0] = px
    out_points[idx, 1] = py
    out_actual_dist[idx] = chainage

    # ============= Normal Approximation ==============
    # We'll do the same approach for chainage +/- tiny. 
    # Then tangent = pt_after - pt_before => rotate => normal
    c_before = chainage - tiny_distance
    c_after  = chainage + tiny_distance
    if c_before < 0.0:
        c_before = 0.0
    if c_after > line_length:
        c_after = line_length

    # Interp c_before
    local_targetB = first_vertex_dist + c_before
    segB = _find_segment_index(prefix_sums, vstart, vend, local_targetB)
    if segB == vend - 1:
        pBx = global_coords[segB, 0]
        pBy = global_coords[segB, 1]
    else:
        dA = prefix_sums[segB]
        dB = prefix_sums[segB + 1]
        denom = dB - dA
        if denom < 1e-12:
            pBx = global_coords[segB, 0]
            pBy = global_coords[segB, 1]
        else:
            t = (local_targetB - dA) / denom
            xA = global_coords[segB, 0]
            yA = global_coords[segB, 1]
            xB = global_coords[segB + 1, 0]
            yB = global_coords[segB + 1, 1]
            pBx = xA + t*(xB - xA)
            pBy = yA + t*(yB - yA)

    # Interp c_after
    local_targetA = first_vertex_dist + c_after
    segA = _find_segment_index(prefix_sums, vstart, vend, local_targetA)
    if segA == vend - 1:
        pAx = global_coords[segA, 0]
        pAy = global_coords[segA, 1]
    else:
        dA = prefix_sums[segA]
        dB = prefix_sums[segA + 1]
        denom = dB - dA
        if denom < 1e-12:
            pAx = global_coords[segA, 0]
            pAy = global_coords[segA, 1]
        else:
            t = (local_targetA - dA) / denom
            xA = global_coords[segA, 0]
            yA = global_coords[segA, 1]
            xB = global_coords[segA + 1, 0]
            yB = global_coords[segA + 1, 1]
            pAx = xA + t*(xB - xA)
            pAy = yA + t*(yB - yA)

    tx = pAx - pBx
    ty = pAy - pBy
    mag = math.sqrt(tx*tx + ty*ty)
    if mag < 1e-12:
        nx = 0.0
        ny = 0.0
    else:
        # normalized tangent
        tx /= mag
        ty /= mag
        # rotate 90 deg left
        nx = -ty
        ny =  tx

    out_normals[idx, 0] = nx
    out_normals[idx, 1] = ny


# ---------------------------
# High-Level Function
# ---------------------------

def discretize_lines_gpu_exact_spacing(
    gdf_lines: gpd.GeoDataFrame,
    distance=10.0,
    row_idx_col="row_idx",
    sub_idx_col="sub_idx",
    chainage_col="chainage",
    tiny_factor=0.01,
    threads_per_block=256
):
    """
    Discretize each (Multi)LineString in 'gdf_lines' exactly at chainages
    [0, distance, 2*distance, ... , line_length]. Then approximate normals
    using +/- tiny offsets. All done with a GPU kernel.

    Returns a GeoDataFrame of points with columns:
      row_idx, sub_idx, chainage, normal_x, normal_y, geometry

    Steps:
    1) For each feature -> each linestring, gather vertices -> prefix sums.
    2) Build an array of desired chainages [0, d, 2d, ... L].
    3) Use a single GPU kernel that does a piecewise interpolation.
    4) Return the combined result.

    NOTE: This code is advanced and might require additional checks for
    extremely large data. The memory usage can be high if many lines
    produce many chainages.
    """
    if gdf_lines.empty:
        logger.warning("Input GeoDataFrame is empty.")
        return gpd.GeoDataFrame()

    # 1. Collect line(s) -> arrays of prefix sums & store row/sub info
    # We'll store each linestring as [vstart, vend) in a global "coords" array.
    # Also store how many chainages each line will produce.

    line_vertex_starts = []
    line_vertex_ends = []
    line_info = []  # (row_idx, sub_idx)
    coords_list = []
    prefix_sums_list = []  # for each vertex, the absolute distance from "some origin"
                           # We'll unify them in a single big array but we need
                           # to keep them separate first to compute offsets.

    # We'll also collect chainages (like CPU approach) for each line in a separate list
    all_chainages = []

    total_vertices = 0
    total_chainages = 0

    # Helper function to compute prefix sums of an array of line segments
    def partial_prefix_sums(xy):
        # xy shape (n, 2)
        # prefix_sums[i] = sum of segment lengths from 0..i-1
        # prefix_sums[0] = 0 (by definition, distance at first vertex)
        # prefix_sums[1] = distance( xy[0], xy[1] )
        # ...
        n = xy.shape[0]
        out = np.zeros(n, dtype=np.float64)
        dist_acc = 0.0
        for i in range(1, n):
            dx = xy[i,0] - xy[i-1,0]
            dy = xy[i,1] - xy[i-1,1]
            seg_len = math.sqrt(dx*dx + dy*dy)
            dist_acc += seg_len
            out[i] = dist_acc
        return out

    # Expand MultiLineStrings
    # For each row, we might have a single linestring or multiple.
    # We'll treat each linestring as a separate "line" in this approach.
    for row_i, geom in enumerate(gdf_lines.geometry):
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            line_list = [geom]
        elif isinstance(geom, MultiLineString):
            line_list = list(geom.geoms)
        else:
            # skip
            continue

        for sub_i, line in enumerate(line_list):
            if line.is_empty:
                continue
            coords_np = np.array(line.coords, dtype=np.float64)
            if coords_np.shape[0] < 2:
                continue
            # prefix sums for this linestring:
            p_sums = partial_prefix_sums(coords_np)
            length = p_sums[-1]
            if length < 1e-12:
                continue

            # create array of chainages for this line
            # [0, distance, 2*distance, ...] plus the end if not included
            c = np.arange(0, length, distance, dtype=np.float64)
            if len(c) == 0:
                c = np.array([0.0], dtype=np.float64)
            if c[-1] < length:
                c = np.append(c, length)

            line_vertex_starts.append(total_vertices)
            total_vertices_for_line = coords_np.shape[0]
            line_vertex_ends.append(total_vertices + total_vertices_for_line)
            line_info.append((row_i, sub_i))

            coords_list.append(coords_np)
            prefix_sums_list.append(p_sums)

            all_chainages.append(c)
            total_vertices += total_vertices_for_line
            total_chainages += len(c)

    if total_vertices == 0:
        logger.warning("No valid lines found.")
        return gpd.GeoDataFrame()

    # 2. Convert everything to big unified arrays
    global_coords = np.zeros((total_vertices, 2), dtype=np.float64)
    prefix_sums   = np.zeros((total_vertices,), dtype=np.float64)

    # We'll fill them in line by line
    current_vertex_pos = 0
    for i, (coords_np) in enumerate(coords_list):
        n = coords_np.shape[0]
        global_coords[current_vertex_pos:current_vertex_pos+n] = coords_np
        # But prefix sums need an offset, i.e. we want them continuous globally
        # so each line's prefix_sums starts from prefix_sums of last line_end?
        # Actually, we need them separate per line. However, for the kernel,
        # we can store them in a single array but with an "absolute offset".
        # We'll do the simpler approach: we add the last prefix_sums's final distance
        # to the next line? That merges them physically. We only want them separate if we
        # truly want a single continuous space. Alternatively, we can store them as is
        # but we must keep track of line boundaries. We'll do "as is" but we must offset:

        if i == 0:
            offset = 0.0
        else:
            # offset = prefix_sums[current_vertex_pos - 1] # last in the previous line
            # Actually, let's just store them strictly increasing across all lines
            offset = prefix_sums[current_vertex_pos - 1]

        p_sums_local = prefix_sums_list[i]
        for k in range(n):
            prefix_sums[current_vertex_pos + k] = offset + p_sums_local[k]

        current_vertex_pos += n

    line_vertex_starts_arr = np.array(line_vertex_starts, dtype=np.int32)
    line_vertex_ends_arr   = np.array(line_vertex_ends,   dtype=np.int32)
    line_info_arr          = np.array(line_info,          dtype=np.int32)  # shape (num_lines, 2)

    # Flatten chainages
    chainage_offsets = np.zeros(len(all_chainages), dtype=np.int32)
    total_so_far = 0
    for i, c_arr in enumerate(all_chainages):
        chainage_offsets[i] = total_so_far
        total_so_far += len(c_arr)
    chainages_global = np.zeros(total_chainages, dtype=np.float64)
    pos = 0
    for c_arr in all_chainages:
        chainages_global[pos:pos+len(c_arr)] = c_arr
        pos += len(c_arr)

    num_lines = len(line_info)
    # 3. GPU memory
    d_prefix_sums       = cuda.to_device(prefix_sums)
    d_global_coords     = cuda.to_device(global_coords)
    d_line_vertex_starts= cuda.to_device(line_vertex_starts_arr)
    d_line_vertex_ends  = cuda.to_device(line_vertex_ends_arr)
    d_line_info         = cuda.to_device(line_info_arr)
    d_chainage_offsets  = cuda.to_device(chainage_offsets)
    d_chainages_global  = cuda.to_device(chainages_global)

    # Output
    d_out_points    = cuda.device_array((total_chainages, 2), dtype=np.float64)
    d_out_normals   = cuda.device_array((total_chainages, 2), dtype=np.float64)
    d_out_distances = cuda.device_array((total_chainages,), dtype=np.float64)

    # Kernel config
    threads = threads_per_block
    blocks  = (total_chainages + threads - 1) // threads

    tiny_distance = tiny_factor * distance

    # Launch
    interpolate_linestring_kernel[blocks, threads](
        d_prefix_sums,
        d_global_coords,
        d_line_vertex_starts,
        d_line_vertex_ends,
        d_chainage_offsets,
        d_chainages_global,
        d_line_info,
        tiny_distance,
        d_out_points,
        d_out_normals,
        d_out_distances
    )

    # Copy back
    h_pts = d_out_points.copy_to_host()
    h_norms = d_out_normals.copy_to_host()
    h_dists = d_out_distances.copy_to_host()

    # Now we must also figure out each chainage's (row_idx, sub_idx).
    # We do the same naive approach we used in the kernel to identify line_i,
    # or we can do a direct approach if chainage_offsets is sorted.

    row_idxs = np.zeros(total_chainages, dtype=np.int32)
    sub_idxs = np.zeros(total_chainages, dtype=np.int32)
    # We'll do a linear approach (inefficient, but simpler). For large #lines, consider prefix sums.
    for i in range(num_lines):
        start_ch_ofs = chainage_offsets[i]
        if i == num_lines - 1:
            end_ch_ofs = total_chainages
        else:
            end_ch_ofs = chainage_offsets[i+1]
        (r_i, s_i) = line_info[i]
        row_idxs[start_ch_ofs:end_ch_ofs] = r_i
        sub_idxs[start_ch_ofs:end_ch_ofs] = s_i

    # Build final GDF
    points_list = [Point(xy[0], xy[1]) for xy in h_pts]
    gdf_out = gpd.GeoDataFrame({
        row_idx_col: row_idxs,
        sub_idx_col: sub_idxs,
        chainage_col: h_dists,
        "normal_x": h_norms[:,0],
        "normal_y": h_norms[:,1]
    }, geometry=points_list, crs=gdf_lines.crs)

    return gdf_out
