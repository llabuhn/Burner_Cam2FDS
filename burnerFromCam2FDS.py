import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import f90nml
import sys, os
import copy
import pandas as pd
from scipy import integrate
from numpy import trapz
import multiprocessing as mp
import pdb 
import downgradeReso

###########################################
class Burner:
    def __init__(self, nml):
        self.surf_template = nml['SURF'][0]
        self.ramp_template = nml['ramp']
        self.vent_template = nml['vent'][0]

    def copy(self):
        return copy.deepcopy(self)


###########################################
# Function to capitalize lines that start with '&'
def capitalize_ampersand_strings(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for line in lines:
            words = line.split()
            updated_words = [word.upper() if word.startswith('&') else word for word in words]
            file.write(" ".join(updated_words) + "\n")

###########################################
def sum_HRR_per_pixel(burner):
    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))
    
    hrr_act_pix = []
    total_HRR = 0
    
    for i, j in zip(*act_pixels):
        hrr_value = float((burner.fre_f[i, j] + burner.fre_s[i, j])*1.e3)
        burner_id = f"Burner_{i}_{j}"
        hrr_act_pix.append((burner_id, hrr_value))
        total_HRR += hrr_value

    return hrr_act_pix, total_HRR

###########################################
def color_burner(f):
    if f > 300: 
        return 'RED'
    elif f > 150: 
        return 'ORANGE'
    elif f > 75: 
        return 'YELLOW'
    elif f > 40: 
        return 'GREEN'
    else:
        return 'BLUE'
    
###########################################
def process_pixel(args):
    #for i,j in zip(*act_pixels):
    
    i, j, subset, template, w, v = args

    if subset[i, j]['arrivalTime'] < 0: return None, None, None 

    arrivalT = subset[i, j]['arrivalTime']-arrivalTime_shift
    residenceT = subset[i, j]['residenceTime']
    burningT = subset[i, j]['burningTime']
    fre_f = subset[i, j]['fre_f']
    fre_s = subset[i, j]['fre_s']
    grid_e = subset[i, j]['grid_e']
    grid_n = subset[i, j]['grid_n']

    dx = subset[i+1, j]['grid_e'] - subset[i, j]['grid_e']
    dy = subset[i, j+1]['grid_n'] - subset[i, j]['grid_n']


    if residenceT == 0:
        HRRPUA_f = 0

    else:
        HRRPUA_f = round(fre_f/Rf_f * 1.e3 / (residenceT * dx * dy), 10)

    if abs(HRRPUA_f) < 1e-10:  
        HRRPUA_f = 0

    if (burningT - residenceT) <= 0:  
        HRRPUA_s = 0
    else:
        HRRPUA_s = round(fre_s/Rf_s * 1.e3 / ((burningT - residenceT) * dx * dy), 10)

    if abs(HRRPUA_s) < 1e-10:  # Evitar valores negativos cercanos a cero
        HRRPUA_s = 0
        
    surf_id = f'BURNER_{i}_{j}'
    ramp_id = f'ramp__{i}_{j}'
    
    template_here = template.copy()
    
    if HRRPUA_f!=0:

        template_here.surf_template['hrrpua'] = HRRPUA_f  
        template_here.surf_template['id'] = surf_id
        template_here.surf_template['ramp_q'] = ramp_id
        
        template_here.surf_template['color'] = color_burner(HRRPUA_f)  

        ramp_f_f = HRRPUA_f / HRRPUA_f if HRRPUA_f != 0 else 0
        ramp_s_f = HRRPUA_s / HRRPUA_f if HRRPUA_f != 0 else 0
  
    else:
        
        template_here.surf_template['hrrpua'] = HRRPUA_s
        template_here.surf_template['id'] = surf_id
        template_here.surf_template['ramp_q'] = ramp_id
        
        template_here.surf_template['color'] = color_burner(HRRPUA_f)  

        ramp_f_f = 0
        ramp_s_f = 1 


    #if ramp_id == 'ramp__26_49': 
    #    pdb.set_trace()

    for k in range(7):
        template_here.ramp_template[k]['id'] = ramp_id
        
    if ramp_s_f > ramp_f_f: print (ramp_id+': higher smoldering')
    template_here.ramp_template[1]['t'] = arrivalT - w
    
    template_here.ramp_template[2]['t'] = arrivalT
    template_here.ramp_template[2]['f'] = ramp_f_f
    
    template_here.ramp_template[3]['t'] = arrivalT + residenceT if residenceT>0 else arrivalT + residenceT + v/2
    template_here.ramp_template[3]['f'] = ramp_f_f
    
    template_here.ramp_template[4]['t'] = arrivalT + residenceT + v
    template_here.ramp_template[4]['f'] = ramp_s_f
    
    template_here.ramp_template[5]['t'] = arrivalT + burningT if burningT>residenceT else arrivalT + residenceT + 2*v
    template_here.ramp_template[5]['f'] = ramp_s_f
    
    template_here.ramp_template[6]['t'] = arrivalT + burningT + v if burningT>residenceT else arrivalT + residenceT + 3*v

    time_ = [template_here.ramp_template[ii]['t'] for ii in range(1,7)]
    if np.diff(np.array(time_)).min()<0: pdb.set_trace()

    template_here.vent_template['xb'] = np.round(np.array([grid_e, grid_e+dx, grid_n, grid_n+dy, 0.0, 0.0]), 3)
    template_here.vent_template['surf_id'] = surf_id
    print (i,j)
    

    return template_here, template_here.ramp_template[1]['t'], template_here.ramp_template[6]['t']


###########################################
if __name__ == '__main__':
###########################################

    # Duration change between fires
    w = .5
    v = .1
    Rf_f = .10 #Radioactive fraction of flaming
    Rf_s = .3  #Radioactive fraction of smoldering
    l = 1
    
    knpdir='/data/paugam/Data/2014_SouthAfrica/'
    maps_fire = np.load(knpdir+'4ForeFire/Skukuza4/skukuza4_4ForeFire.npy')
    
    #divide resolution by 2, dx=2m
    nn = int(maps_fire.shape[0]/2)
    maps2 = np.zeros([nn,nn], dtype=maps_fire.dtype)
    maps2['grid_e'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['grid_e'], 
                                                               maps2.shape, flag_interpolation='min')
    maps2['grid_n'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['grid_n'], 
                                                               maps2.shape, flag_interpolation='min')
    maps2['plotMask'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['plotMask'], 
                                                               maps2.shape, flag_interpolation='max')

    maps2['fre_f'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['fre_f'], 
                                                               maps2.shape, flag_interpolation='sum')
    maps2['fre_s'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['fre_s'], 
                                                               maps2.shape, flag_interpolation='sum')
    
    maps2['arrivalTime'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['arrivalTime'], 
                                                               maps2.shape, flag_interpolation='min')
    maps2['residenceTime'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['residenceTime'], 
                                                               maps2.shape, flag_interpolation='max')
    maps2['burningTime'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['burningTime'], 
                                                               maps2.shape, flag_interpolation='max')
    maps2['moisture'] = downgradeReso.downgrade_resolution_4nadir(maps_fire['moisture'], 
                                                               maps2.shape, flag_interpolation='conservative')

    arrivalTime_shift=100

    #subset_size = 288
    #start = (mm.shape[0]-subset_size)//2
    #end = start + subset_size
    #subset = mm[start:end, start:end]
    #np.save("mid_subset.npy", subset)
    subset = maps2
    subset_size = subset.shape[0]

    inputDirSimu = '/data/paugam/FDS/BurnerSku4/'
    fileConfigIn = inputDirSimu+'input_fixed_burner.fds'

    nml = f90nml.read(fileConfigIn)
    surf_templates_original = nml['surf']
    ramp_templates_original = nml['ramp']
    vent_templates_original = nml['vent']
    del nml['TAIL'], surf_templates_original, ramp_templates_original, vent_templates_original
    

    x = subset['grid_e'][:,0]-subset['grid_e'][0,0]
    y = subset['grid_n'][0,:]-subset['grid_n'][0,0]
    grid_n, grid_e = subset['grid_n']-subset['grid_n'][0,0], subset['grid_e']-subset['grid_e'][0,0]

    burner = subset.view(np.recarray)
    burner.grid_e = grid_e
    burner.grid_n = grid_n

    template = Burner(nml)
    act_pixels = np.where( (burner.fre_f +burner.fre_s) > 0 )
    args = [(i, j, subset, template, w, v) for i, j in zip(*act_pixels)]

    #sys.exit()
    
    # Verificar si $SLURM_NTASKS existe
    if 'SLURM_NTASKS' in os.environ:
        # Leer el valor y convertirlo a un entero
        ntasks = int(os.getenv('SLURM_NTASKS'))
        print(f"SLURM_NTASKS existe y su valor es: {ntasks}")
    else:
        # Acción alternativa si no existe
        ntasks = int(os.getenv('ntask'))
        print("SLURM_NTASKS no está definido en las variables de entorno.")


    # Use multiprocessing to process pixels
    if False:
        with mp.Pool(ntasks) as pool:
            results = pool.map(process_pixel, args)
    else: 
        results = []
        for arg in args: 
            results.append(process_pixel(arg))

    print (len(results),'burners will be written in fds config file')
    # Add processed templates to the .fds file
    minArrivalT = 1.e6
    maxArrivalT = -1.e6
    for template_here, mint, maxt in results:
        if template_here is None: continue
        
        minArrivalT = min([minArrivalT,mint])
        maxArrivalT = max([maxArrivalT,maxt])

        nml.add_cogroup('SURF', template_here.surf_template)
        [nml.add_cogroup('RAMP', ramp_) for ramp_ in template_here.ramp_template]
        nml.add_cogroup('VENT', template_here.vent_template)

    print('min max time in burner:')
    print(minArrivalT)
    print(maxArrivalT)

    nml['tail'] = {}
    del nml['SURF'][0]
    for i in range(7):
        del nml['RAMP'][0]
    del nml['VENT'][0]

    f90nml.write(nml, 'tmp.fds')
    capitalize_ampersand_strings('tmp.fds', inputDir+os.path.basename(fileConfigIn).split('.')[0]+'_withBurner.fds')    
    os.remove('tmp.fds')

    hrr_act_pix, total_HRR = sum_HRR_per_pixel(burner)
    print(f"La suma total de HRR es: {total_HRR:.4f} KJ")

