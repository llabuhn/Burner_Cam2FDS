"""
Converts a fire map (structured .npy) containing per-pixel fire radiative
energy (FRE) with distinct flaming and residual smouldering contributions into a 
Fire Dynamics Simulator (FDS) input file, in which every active fire pixel 
is represented as a 'burner cell' (SURF/VENT pair with time-dependent RAMPs).
"""

import os
import re 
import copy
import warnings
import numpy as np
import multiprocessing as mp
import f90nml
import downgradereso as downgradeReso
import geometry_utils

from pathlib import Path
from datetime import datetime

# ================= CONFIGURATION =================
CASE_NAME = 'burner'                    # FDS case ID (CHID); names the output .fds and all FDS result files
INPUT_NPY = 'skukuza4_4ForeFire.npy'    # ForeFire fire map (structured .npy) to convert
TEMPLATE_FDS = 'burner_template.fds'    # FDS template with the base namelists and BURNER_template marker

TARGET_SCALE = 2            # Defines resolution downgrade in downgradereso
RAMP_UP = 0.5               # Ramp up time of 0.5 s before arrivalTime
RAMP_DOWN = 0.1             # Ramp down time of 0.1 s after residenceTime and burningTime
RF_F = 0.14                 # (Apparent) Radiative fraction in the flaming phase. THIS IS DIFFERENT FROM LOCAL RF IN FDS.
RF_S = 0.34                 # Radiative fraction in the smouldering phase
ARRIVAL_TIME_SHIFT = 100    # Starts fire in the simulation earlier
EMISSIVITY = 0.95           # Emissivity for smoldering surface
TMPA = 33.0                 # Ambient temperature (Celsius)

HOC_GRASS = 18140.0         # kJ/kg
HOC_CARBON = 30500.0        # kJ/kg

ROTATION_ANGLE_DEG = 20.0   # Rotates the fire map from UTM north to align the domain's Y-axis with the main wind direction
DOMAIN_BUFFER_M = 50.0      # Buffer in meters for boundaries in x and y directions
DOMAIN_Z_MAX = 80.0         # Target total height
DOMAIN_Z_LOWER = 3.0        # Height of the high-resolution lower mesh

TOTAL_MPI_CORES = 72        # Total available CPU cores for calculation

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / 'output'

warnings.filterwarnings('ignore', r'All-NaN.*') # Suppress NumPy NaN warnings (NaN pixels are expected)

# ================= HELPER FUNCTIONS =================
def downsample_structured(src, target_res):
    fields = {'grid_e': 'min', 'grid_n': 'min', 'plotMask': 'max', 'fre_f': 'sum', 
              'fre_s': 'sum', 'arrivalTime': 'min', 'residenceTime': 'max', 
              'burningTime': 'max', 'moisture': 'conservative'}
    out = np.zeros(target_res, dtype=src.dtype)
    for name, method in fields.items():
        out[name] = downgradeReso.downgrade_resolution_4nadir(src[name], target_res, method)
    return out

def active_mask(data):
    return (data['fre_f'] + data['fre_s'] > 0) & (~np.isnan(data['arrivalTime']))

def apply_domain_cropping(subset, dx, dy):
    """Crops the domain to the active fire area plus a uniform buffer in x and y directions"""
    i, j = np.where(active_mask(subset))
    if not i.size: return subset

    bi, bj = int(np.ceil(DOMAIN_BUFFER_M / dx)), int(np.ceil(DOMAIN_BUFFER_M / dy))
    h, w = subset.shape
    
    subset = subset[max(0, i.min() - bi) : min(h, i.max() + bi + 1), 
                    max(0, j.min() - bj) : min(w, j.max() + bj + 1)]
    
    subset['grid_e'] -= np.nanmin(subset['grid_e'])
    subset['grid_n'] -= np.nanmin(subset['grid_n'])
    
    return subset

def get_2tier_geometry(lx, ly, cores):
    """Generates a compact 2-tier Z-mesh (Lower: 0.5m res, Upper: 1m res) using MULT."""
    n = cores // 2  
    nx, ny = min(((i, n//i) for i in range(1, n+1) if n % i == 0), key=lambda x: abs(x[0]-x[1]))

    bx, by = int(np.ceil(lx / nx)), int(np.ceil(ly / ny))

    z_cells_lower = int(DOMAIN_Z_LOWER / 0.5)
    z_cells_upper = int((DOMAIN_Z_MAX - DOMAIN_Z_LOWER) / 1.0)

    meshes = [
        {'ijk': [bx*2, by*2, z_cells_lower], 'xb': [0., float(bx), 0., float(by), 0., DOMAIN_Z_LOWER], 'mult_id': 'grid'},
        {'ijk': [bx, by, z_cells_upper], 'xb': [0., float(bx), 0., float(by), DOMAIN_Z_LOWER, DOMAIN_Z_MAX], 'mult_id': 'grid'}
    ]
    mult = {'id': 'grid', 'dx': float(bx), 'dy': float(by), 'dz': 0., 'i_upper': nx-1, 'j_upper': ny-1, 'k_upper': 0}

    return (bx * nx, by * ny), (nx, ny, 2), meshes, mult

# ================= PIXEL PROCESSING =================
def process_pixel(i, j, cell, template, params):
    ramp_up, ramp_down, rf_f, rf_s, shift, dx, dy, eps, tmpa = params
    aT, rT, bT = cell['arrivalTime'], cell['residenceTime'], cell['burningTime']
    
    if np.isnan(aT) or aT < 0: return None
    aT -= shift
    area = dx * dy
    
    hrrpua_f = round(cell['fre_f'] / rf_f * 1e3 / (rT * area), 10) if rT > 0 else 0.0
    hrrpua_s     = round(cell['fre_s'] / rf_s * 1e3 / ((bT - rT) * area), 10) if bT > rT else 0.0  # total HRRPUA: drives mf_s 
    hrrpua_s_rad = round(cell['fre_s']        * 1e3 / ((bT - rT) * area), 10) if bT > rT else 0.0  # radiative HRRPUA: drives T_s 
    mf_s = hrrpua_s / HOC_CARBON if hrrpua_s > 0 else 0.0
    mf_f = round(hrrpua_f / HOC_GRASS, 6) if hrrpua_f > 0 else 0.0

    t_surf = round(((hrrpua_s_rad * 1000.0) / (eps * 5.67e-8) + (tmpa + 273.15)**4)**0.25 - 273.15, 1) if hrrpua_s_rad > 0 else tmpa
    
    if hrrpua_f < 1e-10 and mf_s < 1e-10: return None
    
    surf, _, vent = copy.deepcopy(template)
    bid = f'{i}_{j}'

    for key in ('COLOR', 'color', 'SPEC_ID', 'spec_id', 'HRRPUA', 'hrrpua',
                'RAMP_Q', 'ramp_q', 'MASS_FLUX', 'mass_flux',
                'RAMP_MF', 'ramp_mf', 'TMP_FRONT', 'tmp_front',
                'EMISSIVITY', 'emissivity', 'RAMP_T', 'ramp_t'): surf.pop(key, None)

    surf.update({'ID': f'BURNER_{bid}'})
    spec_idx = 1
    if mf_f > 0:
        surf.update({f'SPEC_ID({spec_idx})': 'GRASS_FUEL', f'MASS_FLUX({spec_idx})': mf_f, f'RAMP_MF({spec_idx})': f'rmf_f_{bid}'})
        spec_idx += 1
    if mf_s > 0:
        surf.update({f'SPEC_ID({spec_idx})': 'C_GAS', f'MASS_FLUX({spec_idx})': round(mf_s, 6), f'RAMP_MF({spec_idx})': f'rmf_s_{bid}',
                     'TMP_FRONT': t_surf, 'EMISSIVITY': eps, 'RAMP_T': f'rt_{bid}'})

    t = [round(x, 2) for x in [0.0, aT - ramp_up, aT, aT + rT, aT + rT + ramp_down, aT + bT, aT + bT + ramp_down]]
    for k in range(1, 7): t[k] = round(max(t[k-1] + 0.01, t[k]), 2)

    ramps = []
    if mf_f > 0:
        ramps.extend([{'ID': f'rmf_f_{bid}', 'T': x, 'F': y} for x, y in zip(t[1:5], [0.0, 1.0, 1.0, 0.0])])
    if mf_s > 0:
        t_s = t[3:]
        ramps.extend([{'ID': f'rt_{bid}',    'T': x, 'F': y} for x, y in zip(t_s, [0.0, 1.0, 1.0, 0.0])])
        ramps.extend([{'ID': f'rmf_s_{bid}', 'T': x, 'F': y} for x, y in zip(t_s, [0.0, 1.0, 1.0, 0.0])])
    
    ge, gn = cell['grid_e'], cell['grid_n']
    vent.update({'SURF_ID': surf['ID'], 'XB': [round(x, 3) for x in [ge, ge + dx, gn, gn + dy, 0.0, 0.0]]})
    
    return (surf, ramps, vent)

# ================= DATA EXPORT =================
def format_fds_block(block_type, params):
    def fmt_elem(k, x):
        if isinstance(x, bool): return '.TRUE.' if x else '.FALSE.'
        if isinstance(x, float):
            return '0.0' if np.isnan(x) else str(round(x, 6 if 'MASS_FLUX' in k.upper() else 4))
        if isinstance(x, str): return x if x.startswith('.') else f"'{x}'"
        return str(x)

    def fmt(k, v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return ", ".join(fmt_elem(k, x) for x in v)
        return fmt_elem(k, v)

    lines = [f" {k.upper()} = {fmt(k, v)}" for k, v in params.items()]
    return "\n".join([f"&{block_type.upper()}"] + lines + ["/\n"])

def write_fds(nml, results, fds_outfile, domain_sz):
    for key in ('TAIL', 'tail'): nml.pop(key, None)

    header = f"! Generated computationally with cam2fds in Python (https://github.com/3dfirelab/Burner_Cam2FDS)\n" \
             f"! Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
             f"! Domain: {domain_sz[0]:.1f} m (X) x {domain_sz[1]:.1f} m (Y) x {domain_sz[2]:.1f} m (Z)\n\n"

    preferred_order = ['head', 'time', 'mesh', 'mult', 'dump', 'misc', 'reac', 'spec', 'surf',
                       'vent', 'wind', 'devc', 'slcf', 'bndf']
    nml_keys = sorted(nml.keys(), key=lambda k: preferred_order.index(k.lower()) if k.lower() in preferred_order else 99)

    with open(fds_outfile, 'w') as f:
        f.write(header)

        for grp_name in nml_keys:
            blocks = nml[grp_name]
            if not isinstance(blocks, list): blocks = [blocks]
            for block in blocks:
                if block: f.write(format_fds_block(grp_name, block))

        for surf, ramps, vent in (r for r in results if r):
            f.write(format_fds_block('SURF', surf))
            f.write("".join(format_fds_block('RAMP', r) for r in ramps))
            f.write(format_fds_block('VENT', vent))

        f.write("&TAIL /\n")


def to_dict(txt, tag):
    """Parses a single FDS namelist block string into a plain dict."""
    if not txt: return {}
    v = f90nml.reads(txt).get(tag.lower(), {})
    return copy.deepcopy(v[0] if isinstance(v, list) else v)

# ================= MAIN PIPELINE =================
def main():
    print(f"--- Generating FDS case '{CASE_NAME}' ---", flush=True)
    maps_fire = np.load(INPUT_DIR / INPUT_NPY)
    
    target_res = (maps_fire.shape[0] // TARGET_SCALE, maps_fire.shape[1] // TARGET_SCALE)
    subset = downsample_structured(maps_fire, target_res)
    
    subset['grid_e'] -= np.nanmin(subset['grid_e'])
    subset['grid_n'] -= np.nanmin(subset['grid_n'])
    dx, dy = float(np.nanmedian(np.diff(subset['grid_e'], axis=0))), float(np.nanmedian(np.diff(subset['grid_n'], axis=1)))

    subset = geometry_utils.resample_rotated_grid(subset, dx, dy, ROTATION_ANGLE_DEG)
    subset = apply_domain_cropping(subset, dx, dy)

    with open(INPUT_DIR / TEMPLATE_FDS, 'r') as f: raw_txt = f.read()
    def extract_template_block(tag):
        m = re.search(rf'(?i)&{tag}[\s\S]*?BURNER_template[\s\S]*?/', raw_txt)
        return m.group(0) if m else ""
    surf_txt, vent_txt = extract_template_block('SURF'), extract_template_block('VENT')
    
    nml = f90nml.reads(raw_txt.replace(surf_txt, '').replace(vent_txt, ''))
    if 'head' in nml: nml['head'].update({'chid': CASE_NAME, 'title': CASE_NAME})

    lx, ly = subset.shape[0] * dx, subset.shape[1] * dy
    for k in ('mesh', 'MESH', 'mult', 'MULT'): nml.pop(k, None)
    (fds_xmax, fds_ymax), _, nml['mesh'], nml['mult'] = get_2tier_geometry(lx, ly, TOTAL_MPI_CORES)
    
    def update_nml_group(name, func):
        if name in nml:
            for item in (nml[name] if isinstance(nml[name], list) else [nml[name]]): func(item)

    update_nml_group('slcf', lambda s: s.update({'xb': [0.0, float(fds_xmax), 0.0, float(fds_ymax), 0.0, DOMAIN_Z_MAX]}))

    vent_xb_map = {
        '[XMAX]': [fds_xmax, fds_xmax, 0.0, fds_ymax, 0.0, DOMAIN_Z_MAX],
        '[XMIN]': [0.0, 0.0, 0.0, fds_ymax, 0.0, DOMAIN_Z_MAX],
        '[YMAX]': [0.0, fds_xmax, fds_ymax, fds_ymax, 0.0, DOMAIN_Z_MAX],
        '[YMIN]': [0.0, fds_xmax, 0.0, 0.0, 0.0, DOMAIN_Z_MAX],
        '[ZMAX]': [0.0, fds_xmax, 0.0, fds_ymax, DOMAIN_Z_MAX, DOMAIN_Z_MAX]
    }
    update_nml_group('vent', lambda v: v.update({'xb': vent_xb_map[str(v.get('id', '')).upper()]}) if str(v.get('id', '')).upper() in vent_xb_map else None)

    domain_sz = (fds_xmax, fds_ymax, DOMAIN_Z_MAX)

    template = (to_dict(surf_txt, 'SURF'), {}, to_dict(vent_txt, 'VENT'))

    ntasks = int(os.getenv('SLURM_NTASKS', 1))
    params = (RAMP_UP, RAMP_DOWN, RF_F, RF_S, ARRIVAL_TIME_SHIFT, dx, dy, EMISSIVITY, TMPA)

    mask_post_crop = active_mask(subset)
    args = [(i, j, subset[i, j], template, params) for i, j in zip(*np.where(mask_post_crop))]
    
    with mp.Pool(ntasks) as pool:
        results = pool.starmap(process_pixel, args) if ntasks > 1 else [process_pixel(*a) for a in args]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fds_outfile = OUTPUT_DIR / f"{CASE_NAME}.fds"
    write_fds(nml, results, fds_outfile, domain_sz)

if __name__ == "__main__":
    main()

