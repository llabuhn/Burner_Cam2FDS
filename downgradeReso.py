"""
=========================================================================
Downgrade 2D Array Resolution 

- Reduces a 2D array to a target resolution using block aggregation.
- Methods: 'average', 'min', 'max', 'sum' (ignore nodata), 'conservative' (nodata=0).
- Nodata (-999) is internally converted to NaN.
=========================================================================
"""

import numpy as np

def downgrade_resolution_4nadir(a, out_shape, flag_interpolation, nodata=-999):
    a = np.where(a == nodata, np.nan, a)
    reducer = {
        'min': np.nanmin,
        'max': np.nanmax,
        'sum': np.nansum,
        'conservative': lambda x, axis: np.mean(np.nan_to_num(x, nan=0.0), axis=axis)
    }.get(flag_interpolation, np.nanmean)   # fallback np.nanmean == 'average'

    bx, by = a.shape[0] // out_shape[0], a.shape[1] // out_shape[1]
    a = a[:bx*out_shape[0], :by*out_shape[1]]
    return reducer(a.reshape(out_shape[0], bx, out_shape[1], by), axis=(1,3))
