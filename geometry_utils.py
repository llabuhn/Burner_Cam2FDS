"""
Rotates the fire map by angle_deg around its centre onto a new axis-aligned
grid: active pixels are turned into polygons, rotated, and intersected with
the target grid; FRE (fre_f/fre_s) is redistributed area-weighted so the
total is conserved, while arrival/residence/burning times are aggregated per
cell (min/max). Returns a new structured array on the rotated grid.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box

def resample_rotated_grid(subset, dx, dy, angle_deg):
    if angle_deg == 0.0:
        return subset

    # 1. Mask active pixels
    mask = (subset['fre_f'] + subset['fre_s'] > 0) & (~np.isnan(subset['arrivalTime']))
    v_e, v_n = subset['grid_e'][mask], subset['grid_n'][mask]

    # 2. Domain bounds & center point calculation
    x_min, x_max = subset['grid_e'].min(), subset['grid_e'].max() + dx
    y_min, y_max = subset['grid_n'].min(), subset['grid_n'].max() + dy
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    # 3. Create & rotate active source polygons around center
    src_polys = [box(e, n, e + dx, n + dy) for e, n in zip(v_e, v_n)]
    src_gdf = gpd.GeoDataFrame({
        'fre_f': subset['fre_f'][mask], 'fre_s': subset['fre_s'][mask],
        'aT': subset['arrivalTime'][mask], 'rT': subset['residenceTime'][mask],
        'bT': subset['burningTime'][mask]
    }, geometry=src_polys)

    src_gdf['geometry'] = src_gdf.geometry.rotate(angle_deg, origin=(cx, cy))
    src_gdf['src_area'] = src_gdf.geometry.area

    # 4. Target domain     
    dom_in = gpd.GeoSeries([box(x_min, y_min, x_max, y_max)])
    x_min_rot, y_min_rot, x_max_rot, y_max_rot = dom_in.rotate(angle_deg, origin=(cx, cy))[0].bounds

    nx = int(np.ceil((x_max_rot - x_min_rot) / dx))
    ny = int(np.ceil((y_max_rot - y_min_rot) / dy))
    te, tn = np.meshgrid(np.arange(nx) * dx + x_min_rot, np.arange(ny) * dy + y_min_rot, indexing='ij')

    tgt_polys = [box(x, y, x + dx, y + dy) for x, y in zip(te.ravel(), tn.ravel())]
    tgt_gdf = gpd.GeoDataFrame({'tgt_id': np.arange(len(tgt_polys))}, geometry=tgt_polys)

    # 5. Intersect & area-weighted allocation
    inter = gpd.overlay(tgt_gdf, src_gdf, how='intersection')
    inter['area_frac'] = inter.geometry.area / inter['src_area']
    inter['fre_f_part'] = inter['fre_f'] * inter['area_frac']
    inter['fre_s_part'] = inter['fre_s'] * inter['area_frac']

    grouped = inter.groupby('tgt_id').agg({
        'fre_f_part': 'sum', 'fre_s_part': 'sum', 'aT': 'min', 'rT': 'max', 'bT': 'max'
    }).reset_index()

    # 6. Reconstruct 2D array
    out = np.zeros((nx, ny), dtype=subset.dtype)
    out['grid_e'], out['grid_n'] = te, tn
    out['arrivalTime'] = np.nan 

    idx_i, idx_j = np.unravel_index(grouped['tgt_id'].values, (nx, ny), order='C')
    out['fre_f'][idx_i, idx_j] = grouped['fre_f_part'].values
    out['fre_s'][idx_i, idx_j] = grouped['fre_s_part'].values
    out['arrivalTime'][idx_i, idx_j] = grouped['aT'].values
    out['residenceTime'][idx_i, idx_j] = grouped['rT'].values
    out['burningTime'][idx_i, idx_j] = grouped['bT'].values

    # Shift coordinates to start at (0,0) for FDS mesh
    out['grid_e'] -= np.nanmin(out['grid_e'])
    out['grid_n'] -= np.nanmin(out['grid_n'])

    return out
