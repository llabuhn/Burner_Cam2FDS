import numpy as np 
import matplotlib.pyplot as plt

######################################################
def downgrade_resolution_4nadir(arr, diag_res_cte_shape, flag_interpolation='conservative'):

    '''
    flag_interpolation is conservative, max, min, average or sum 
    '''
    factor = (1.*arr.shape[0]/diag_res_cte_shape[0])
    if factor == int(np.floor(factor)): factor = int(factor)
    else: factor = int(factor) + 1
    #factor = int(np.round(old_div(1.*arr.shape[0],diag_res_cte_shape[0]),0))
    
    if np.mod( arr.shape[0], factor )!=0:
        extra_pixel0 = factor-np.mod( arr.shape[0], factor )
        extra_pixel1 = factor-np.mod( arr.shape[1], factor )
    else: 
        extra_pixel0 = 0
        extra_pixel1 = 0
  
    if (extra_pixel0>0) |  (extra_pixel1>0):
        x = np.arange(0,arr.shape[0],1)
        y = np.arange(0,arr.shape[1],1)
        z = arr.flatten()
        f = interpolate.interp2d(x, y, z, kind='linear')
        
        grid_x = np.arange(0-int(0.5*extra_pixel0),extra_pixel0-int(0.5*extra_pixel0)+arr.shape[0],1)
        grid_y = np.arange(0-int(0.5*extra_pixel1),extra_pixel1-int(0.5*extra_pixel1)+arr.shape[1],1) 
        arr = f(grid_x, grid_y)
        arr = arr.T
    
    if flag_interpolation == 'max':
        return shrink_max(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
   
    elif flag_interpolation == 'min':
        return shrink_min(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'conservative':
        
        mask = np.where(arr!=-999, 1, 0)
        sum_pixel = shrink_sum(mask, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        
        sum = shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        return np.where(sum != -999, (sum/sum_pixel), sum)

    elif flag_interpolation == 'average':
        return shrink_average(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'sum':
        return shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])

    else:
        print('bad flag')
        pdb.set_trace()


######################################################
def shrink_sum(data, nx, ny, nodata=-999):
    
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.sum(3).sum(1).mask==False, rshp.sum(3).sum(1).data, nodata)

######################################################
def shrink_max(data, nx, ny, nodata=-999):
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.max(3).max(1).mask==False, rshp.max(3).max(1).data, nodata)
    
    #return min3    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).max(axis=1).max(axis=2)


######################################################
def shrink_min(data, nx, ny, nodata=-999):
    
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.min(3).min(1).mask == False, rshp.min(3).min(1).data, nodata)

    #return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).min(axis=1).min(axis=2)


######################################################
def shrink_average(data, nx, ny, nodata=-999.):
   
    data_masked = np.ma.array(data, mask = np.where(data==nodata, 1, 0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    mean3 = rshp.mean(3)
    count3 = rshp.count(3)

    # Compute weighted mean along axis 1
    mean1 = ((count3*mean3).sum(1)/count3.sum(1))
    
    return np.where( mean1.mask, nodata, mean1.data)


if __name__ == '__main__':
    arr = np.random.random([10,10])
    mm = downgrade_resolution_4nadir(arr, [5,5], flag_interpolation='sum')

    ax = plt.subplot(121)
    plt.imshow(arr.T, origin='lower')
    ax = plt.subplot(122)
    plt.imshow(mm.T, origin='lower')
    plt.show()

