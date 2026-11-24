#Reproducing Figure 4 from: Van De Putte et al. (2026)
#JWST observations of photodissociation regions. IV. Carbonaceous emission band sub-components in NGC 7023  have distinct spatial distributions
#https://doi.org/10.1051/0004-6361/202658975

#import statements
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
# import random
# import pandas
# import statistics
# import scipy
# import scipy.io as sio
# from scipy.optimize import curve_fit
# from astropy.io import ascii
# from scipy.interpolate import UnivariateSpline
# from matplotlib.backends.backend_pdf import PdfPages
# import copy
# from math import floor, ceil
# from PIL import Image
# from astropy import units as u
# from astropy.utils.data import get_pkg_data_filename
# # from ismwestern.io import save_Spectrum1D_to_fits, FITSFormat
# from specutils import Spectrum1D
# from ismwestern import io
# from astropy.io import fits
# from astropy.wcs import WCS
# from astropy.units import Unit
# from astropy.nddata import StdDevUncertainty
# from astropy.modeling import models, fitting
# from astropy.stats import gaussian_sigma_to_fwhm
from ismwestern.general.mathutils.functions import gaussian, lorentzian, skewed_gaussian
from ismwestern.fitting.profile_fitter import ProfileFitter
# from tqdm import tqdm
from scipy.interpolate import make_splrep

#Functions
def _wave_to_ind(wavelengths, x):
    ind = np.zeros(x.shape).astype(np.int64)
    for i, wave in enumerate(x):
        ind[i] = np.argmin(abs(wavelengths - wave))
    return ind

def _anchor_point(
        data, 
        ap_ind, 
        ap_method=None, 
        ext=None, 
        ap_lb_ind=None, 
        ap_ub_ind=None
        ):
    """
    Calculates the y value of a single anchor point.

    Parameters
    ----------
    data : numpy.ndarray
        spectra data cube. assumes that the spectral axis is 0.
    ap_ind : int
        anchor point x index corresponding to the y val to be calculated.
    ap_method : int, optional
        approach to be used for determining the y val.
    ext : int, optional
        extent of bounds to be used in median calculations. if ext=2, median has 5 entires.
    ap_lb_ind : int, optional
        index of lower bound used for some methods.
    ap_ub_ind : int, optional
        index of upper bound used for some methods.
        
    Returns
    -------
    ap_y : int
        y val corresponding to ap_x, together making an anchor point used for spline calculation.
    """    
    # convinience variable
    N = data.shape[0]

    # determining logic of all extra conditions now to avoid excessive nesting
    # trading efficiency for readability of code
    
    # ap_ind cannot be too close to edges
    ec_2 = ap_ind > 9 and ap_ind < N-10
    
    if ext is not None:
        # ap_ind cannot be too close to edges
        ec_ext = ap_ind > ext-1 and ap_ind < N-ext
    else:
        ec_ext = False
        
    if ap_lb_ind is not None and ap_ub_ind is not None:
        # bounds need to be in correct order and not too close to edges
        ec_b = ap_ub_ind > ap_lb_ind and ap_lb_ind > ext-1 and ap_ub_ind < N-ext
    else:
        ec_b = False

    # calculating ap_y
    if ap_method == 1 and ec_ext == True:
        # median using ext
        ap_y = np.nanmedian(data[ap_ind - ext : ap_ind + ext + 1], axis=0)

    elif ap_method == 2 and ec_2 == True:
        # median with 10 on either side of ap_ind, 21 total
        ap_y = np.nanmedian(data[ap_ind - 10 : ap_ind + 11], axis=0)
    
    elif ap_method == 3 and ec_b == True:
        # use bounds to calculate linear function, anchor point on this function.
        d1 = np.nanmedian(data[ap_lb_ind - ext : ap_lb_ind + ext + 1], axis=0)
        d2 = np.nanmedian(data[ap_ub_ind - ext : ap_ub_ind + ext + 1], axis=0)
        # define line in terms of indices
        m = (d2 - d1)/(ap_ub_ind - ap_lb_ind)
        ap_y = m*(ap_ind - ap_lb_ind) + d1
        
    elif ap_method == 4 and ec_b == True: 
        # use bounds to calculate a flat line, anchor point on this line.
        d1 = np.nanmedian(data[ap_lb_ind - ext : ap_lb_ind + ext + 1], axis=0)
        d2 = np.nanmedian(data[ap_ub_ind - ext : ap_ub_ind + ext + 1], axis=0)
        ap_y = (d1 + d2)/2
        
    else:
        # use data[ind] only, unless the conditions are met for something more complex.
        # this is the expected output if ap_method=0, or if the intended args are incorrect.
        ap_y = data[ap_ind]
            
    return ap_y

def spline_from_anchor_points(
        wavelengths, 
        data, 
        ap_x, 
        ap_method=None, 
        ext=None, 
        ap_lb=None, 
        ap_ub=None
        ):
    """
    Turns a list of wavelengths into anchor points, and fits a spline to them.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        wavelengths array. Units should match ap_x, ap_lb, ap_ub.
    data : numpy.ndarray
        spectra data cube. assumes that the spectral axis is 0.
    ap_x : numpy.ndarray
        anchor point wavelengths. All ap are assumed to have the same length.
    ap_method : list of int, optional
        approach to be used for determining the anchor point y val.
    ext : list of int, optional
        extent of bounds to be used in median calculations. if ext=2, median has 5 entires.
    ap_lb : list of int, optional
        wavelengths of lower bounds used for some methods.
    ap_ub : list of int, optional
        wavelengths of upper bounds used for some methods.
        
    Returns
    -------
    spl : numpy.ndarray
        spline fits applied to wavelengths, same shape as data.
    """   
    # shape of data
    shape_y, shape_x = data[0].shape
    M = ap_x.shape[0]
    
    # data needs to have no nans, leave data intact 
    data_nonan = np.copy(data)
    data_nonan[np.isnan(data_nonan)] = 0
    
    # convert ap_x to ap_ind
    ap_ind = _wave_to_ind(wavelengths, ap_x)
    
    # need both ap_lb and ap_ub to not be None for their routines to function
    if ap_lb is not None and ap_ub is not None:
        ap_lb_ind = _wave_to_ind(wavelengths, ap_lb)
        ap_ub_ind = _wave_to_ind(wavelengths, ap_ub)
    else:
        ap_lb_ind = [None]*M
        ap_ub_ind = [None]*M
    
    # make ap_method iterable if it is None
    if ap_method is None:
        ap_method = [None]*M
        
    # make ext iterable if it is None
    if ext is None:
        ext = [None]*M
    
    # calculating the anchor point y vals
    ap_y = np.zeros((M, shape_y, shape_x))
    # each input is a list/ndarray of the same length
    for w, ind in enumerate(ap_ind):
        ap_y[w] = _anchor_point(
            data_nonan, 
            ind, 
            ap_method=ap_method[w], 
            ext=ext[w], 
            ap_lb_ind=ap_lb_ind[w], 
            ap_ub_ind=ap_ub_ind[w]
            )
    
    # calculate BSpline instance
    spl_func = make_splrep(ap_x, ap_y)
    spl = spl_func(wavelengths)
    return spl, spl_func

def drude(x, x0, amp, fwhm):

    return amp * (fwhm / x0)**2 / ((x/x0 - x0/x)**2 + (fwhm/x0)**2)

#File imports
# flux = np.load('NGC7023_Files/NGC7023_ATM_Fluxes.npy') #this version has atomic lines
fluxes_7023_atm = np.load('NGC7023_Files/NGC7023_ATM_Continuum.npy')
waves_7023_atm = np.load('NGC7023_Files/NGC7023_ATM_Wavelengths.npy')

# estimating and removing the global continuum contirbution
anchors_g = np.array([1.10, 1.23, 1.67, 3.05, 5.08, 10.30, 15.00, 23.00, 24.85, 25.50, 26.15])
spline2, spl_func2 = spline_from_anchor_points(waves_7023_atm, fluxes_7023_atm[:, np.newaxis, np.newaxis], anchors_g)
spline_7023_atm = spline2[:, 0, 0]
fluxes_7023_atm_gsub = fluxes_7023_atm - spline_7023_atm

# making a plot to show what the estimated continuum looks like and the visualize anchor points
anchors_g_yaxis = [
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 1.10)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 1.23)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 1.67)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 3.05)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 5.08)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 10.30)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 15.00)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 23.00)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 24.85)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 25.50)[0][0]],
    spline_7023_atm[np.where(np.round(waves_7023_atm, 2) == 26.15)[0][0]]
]

fig = plt.figure(figsize=(10,7))
ax = plt.gca()
plt.plot(waves_7023_atm, fluxes_7023_atm, color='grey')
plt.plot(waves_7023_atm, spline_7023_atm, color='black')
plt.scatter(anchors_g, anchors_g_yaxis, zorder=2)

ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 6))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=7, width=1, direction='in')
ax.tick_params(axis='x', which='minor', labelbottom=True, top=True, length=5, width=1, direction='in')
ax.tick_params(axis='y', which='major', labelleft=True, right=True, length=7, width=1, direction='in')
ax.tick_params(axis='y', which='minor', labelleft=True, right=True, length=5, width=1, direction='in')

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.title(r'NGC 7023 ATM Template Spectra')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$F_\nu$ (MJy sr$^{-1}$)')

# plt.show()
plt.savefig(f'global_continuum_overlay', dpi=300, bbox_inches='tight', pad_inches=0.1)

# Van De Putte et al. (2026) fits the PAH band profiles after removing both the local and global contribution
# They fit the '16-18' micron region using 9 components
# The wavelength of the centre of the profile and the FWHM are given in Table 2

# [wavelength of max amplitude, FWHM]
# wavelength values have been rounded to two decimal places compared to Table 2
# this is done to account for np.where() not being able to find indecies out to 3 decimal places
tabvals_16p0a = [15.86, 0.289]
tabvals_16p0b = [16.03, 0.114]
tabvals_16p4a = [16.40, 0.040]
tabvals_16p4b = [16.44, 0.182]
tabvals_16p7 = [16.72, 0.352]
tabvals_17p0 = [17.00, 0.464]
tabvals_17p2 = [17.17, 0.219]
tabvals_17p4 = [17.37, 0.224]
tabvals_17p7 = [17.77, 0.362]

# Drude Profile Replication
# [index of maximum amplitude wavelength, fwhm]
comp_16p0a = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p0a[0])[0][0],
    tabvals_16p0a[1]
]
comp_16p0b = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p0b[0])[0][0],
    tabvals_16p0b[1]
]
comp_16p4a = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p4a[0])[0][0],
    tabvals_16p4a[1]
]
comp_16p4b = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p4b[0])[0][0],
    tabvals_16p4b[1]
]
comp_16p7 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p7[0])[0][0],
    tabvals_16p7[1] / 2
]
comp_17p0 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p0[0])[0][0],
    tabvals_17p0[1]
]
comp_17p2 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p2[0])[0][0],
    tabvals_17p2[1]
]
comp_17p4 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p4[0])[0][0],
    tabvals_17p4[1]
]
comp_17p7 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p7[0])[0][0],
    tabvals_17p7[1]
]

# We will replicate using a superposition of drude profiles
bounds = [
    np.where(np.round(waves_7023_atm, 2) == 15.00)[0][0],
    np.where(np.round(waves_7023_atm, 2) == 18.50)[0][0]
]
vary = [False, True, False]
profile = ProfileFitter()
#16.0A component
profile.add_component(
    drude,
    ['x01', 'amp1', 'fwhm1'],
    initial_values=[tabvals_16p0a[0], comp_16p0a[0], comp_16p0a[1]],
    min_values=[15.8, 10, 0],
    max_values=[15.9, 1000, comp_16p0a[1] + 0.001],
    allow_vary=vary
)

#16.0B component
profile.add_component(
    drude,
    ['x02', 'amp2', 'fwhm2'],
    initial_values=[tabvals_16p0b[0], comp_16p0b[0], comp_16p0b[1]],
    min_values=[16.0, 10, 0],
    max_values=[16.05, 1000, comp_16p0b[1] + 0.001],
    allow_vary=vary
)

#16.4A component
profile.add_component(
    drude,
    ['x03', 'amp3', 'fwhm3'],
    initial_values=[tabvals_16p4a[0], comp_16p4a[0], comp_16p4a[1]],
    min_values=[16.40, 10, 0],
    max_values=[16.41, 1000, comp_16p4a[1] + 0.001],
    allow_vary=vary
)

#16.4B component
profile.add_component(
    drude,
    ['x04', 'amp4', 'fwhm4'],
    initial_values=[tabvals_16p4b[0], comp_16p4b[0], comp_16p4b[1]],
    min_values=[16.43, 10, 0],
    max_values=[16.44, 1000, comp_16p0b[1] + 0.001],
    allow_vary=vary
)

#16.7 component
profile.add_component(
    drude,
    ['x05', 'amp5', 'fwhm5'],
    initial_values=[tabvals_16p7[0], comp_16p7[0], comp_16p7[1]],
    min_values=[16.7, 10, 0],
    max_values=[16.75, 1000, comp_16p7[1]],
    allow_vary=vary
)

#17.0 component
profile.add_component(
    drude,
    ['x06', 'amp6', 'fwhm6'],
    initial_values=[tabvals_17p0[0], comp_17p0[0], comp_17p0[1]],
    min_values=[16.9, 10, 0],
    max_values=[17.1, 1000, comp_17p0[1] + 0.001],
    allow_vary=vary
)

#17.2 component
profile.add_component(
    drude,
    ['x07', 'amp7', 'fwhm7'],
    initial_values=[tabvals_17p2[0], comp_17p2[0], comp_17p2[1]],
    min_values=[17.1, 10, 0],
    max_values=[17.2, 1000, comp_17p2[1] + 0.001],
    allow_vary=vary
)

#17.4 component
profile.add_component(
    drude,
    ['x08', 'amp8', 'fwhm8'],
    initial_values=[tabvals_17p4[0], comp_17p4[0], comp_17p4[1]],
    min_values=[17.3, 10, 0],
    max_values=[17.4, 1000, comp_17p4[1] + 0.001],
    allow_vary=vary
)

#17.7 component
profile.add_component(
    drude,
    ['x09', 'amp9', 'fwhm9'],
    initial_values=[tabvals_17p7[0], comp_17p7[0], comp_17p7[1]],
    min_values=[17.7, 10, 0],
    max_values=[17.8, 1000, comp_17p7[1] + 0.001],
    allow_vary=vary
)

wave_range = waves_7023_atm[bounds[0]:bounds[1]]
flux_range = fluxes_7023_atm_gsub[bounds[0]:bounds[1]]
global_range = spline_7023_atm[bounds[0]:bounds[1]]
profile_fit = profile.fit(wave_range, flux_range)

col_16p0a = '#cd6578'
col_16p0b = '#332288'
col_16p4a = '#016d25'
col_16p4b = '#88cbf0'
col_16p7 = '#82144a'
col_17p0 = '#43aa97'
col_17p2 = '#999932'
col_17p4 = '#aa4498'
col_17p7 = '#ca6678'

fig = plt.figure(figsize=(5, 6))
ax = plt.gca()
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range), color='black')
plt.plot(wave_range, flux_range + global_range, color='grey')

plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[True, False, False, False, False, False, False, False, False]), color=col_16p0a, label='16.0A')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, True, False, False, False, False, False, False, False]), color=col_16p0b, label='16.0B')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, True, False, False, False, False, False, False]), color=col_16p4a, label='16.4A')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, True, False, False, False, False, False]), color=col_16p4b, label='16.4B')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, True, False, False, False, False]), color=col_16p7, label='16.7')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, True, False, False, False]), color=col_17p0, label='17.0')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, False, True, False, False]), color=col_17p2, label='17.2')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, False, False, True, False]), color=col_17p4, label='17.4')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, False, False, False, True]), color=col_17p7, label='17.7')

ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 6))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=7, width=1, direction='in')
ax.tick_params(axis='x', which='minor', labelbottom=True, top=True, length=5, width=1, direction='in')
ax.tick_params(axis='y', which='major', labelleft=True, right=True, length=7, width=1, direction='in')
ax.tick_params(axis='y', which='minor', labelleft=True, right=True, length=5, width=1, direction='in')

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)

plt.title(r'16-18 $\mu$m decomposition - Drude')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$F_\nu$ (MJy sr$^{-1}$)')

plt.xlim(15.5, 18.3)
plt.ylim(125, 1150)
plt.legend()
# plt.show()
plt.savefig(f'fig_4_reproduction_drudeprofile', dpi=300, bbox_inches='tight', pad_inches=0.1)

#Lorentzian replication
# [index of maximum amplitude wavelength, gamma]
comp_16p0a = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p0a[0])[0][0],
    tabvals_16p0a[1] / 2
]
comp_16p0b = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p0b[0])[0][0],
    tabvals_16p0b[1] / 2
]
comp_16p4a = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p4a[0])[0][0],
    tabvals_16p4a[1] / 2
]
comp_16p4b = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p4b[0])[0][0],
    tabvals_16p4b[1] / 2
]
comp_16p7 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_16p7[0])[0][0],
    tabvals_16p7[1] / 2
]
comp_17p0 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p0[0])[0][0],
    tabvals_17p0[1] / 2
]
comp_17p2 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p2[0])[0][0],
    tabvals_17p2[1] / 2
]
comp_17p4 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p4[0])[0][0],
    tabvals_17p4[1] / 2
]
comp_17p7 = [
    np.where(np.round(waves_7023_atm, 2) == tabvals_17p7[0])[0][0],
    tabvals_17p7[1] / 2
]

# We will replicate using a superposition of lorentzian profiles
bounds = [
    np.where(np.round(waves_7023_atm, 2) == 15.00)[0][0],
    np.where(np.round(waves_7023_atm, 2) == 18.50)[0][0]
]
vary = [False, True, False]
profile = ProfileFitter()
#16.0A component
profile.add_component(
    lorentzian,
    ['x01', 'amp1', 'gamma1'],
    initial_values=[tabvals_16p0a[0], comp_16p0a[0], comp_16p0a[1]],
    min_values=[15.8, 10, 0],
    max_values=[15.9, 1000, 0.15],
    allow_vary=vary
)

#16.0B component
profile.add_component(
    lorentzian,
    ['x02', 'amp2', 'gamma2'],
    initial_values=[tabvals_16p0b[0], comp_16p0b[0], comp_16p0b[1]],
    min_values=[16.0, 10, 0],
    max_values=[16.05, 1000, 0.06],
    allow_vary=vary
)

#16.4A component
profile.add_component(
    lorentzian,
    ['x03', 'amp3', 'gamma3'],
    initial_values=[tabvals_16p4a[0], comp_16p4a[0], comp_16p4a[1]],
    min_values=[16.40, 10, 0],
    max_values=[16.41, 1000, 0.03],
    allow_vary=vary
)

#16.4B component
profile.add_component(
    lorentzian,
    ['x04', 'amp4', 'gamma4'],
    initial_values=[tabvals_16p4b[0], comp_16p4b[0], comp_16p4b[1]],
    min_values=[16.43, 10, 0],
    max_values=[16.44, 1000, 0.10],
    allow_vary=vary
)

#16.7 component
profile.add_component(
    lorentzian,
    ['x05', 'amp5', 'gamma5'],
    initial_values=[tabvals_16p7[0], comp_16p7[0], comp_16p7[1]],
    min_values=[16.7, 10, 0],
    max_values=[16.75, 1000, 0.18],
    allow_vary=vary
)

#17.0 component
profile.add_component(
    lorentzian,
    ['x06', 'amp6', 'gamma6'],
    initial_values=[tabvals_17p0[0], comp_17p0[0], comp_17p0[1]],
    min_values=[16.9, 10, 0],
    max_values=[17.1, 1000, 0.24],
    allow_vary=vary
)

#17.2 component
profile.add_component(
    lorentzian,
    ['x07', 'amp7', 'gamma7'],
    initial_values=[tabvals_17p2[0], comp_17p2[0], comp_17p2[1]],
    min_values=[17.1, 10, 0],
    max_values=[17.2, 1000, 0.11],
    allow_vary=vary
)

#17.4 component
profile.add_component(
    lorentzian,
    ['x08', 'amp8', 'gamma8'],
    initial_values=[tabvals_17p4[0], comp_17p4[0], comp_17p4[1]],
    min_values=[17.3, 10, 0],
    max_values=[17.4, 1000, 0.12],
    allow_vary=vary
)

#17.7 component
profile.add_component(
    lorentzian,
    ['x09', 'amp9', 'gamma9'],
    initial_values=[tabvals_17p7[0], comp_17p7[0], comp_17p7[1]],
    min_values=[17.7, 10, 0],
    max_values=[17.8, 1000, 0.19],
    allow_vary=vary
)

wave_range = waves_7023_atm[bounds[0]:bounds[1]]
flux_range = fluxes_7023_atm_gsub[bounds[0]:bounds[1]]
global_range = spline_7023_atm[bounds[0]:bounds[1]]
profile_fit = profile.fit(wave_range, flux_range)

col_16p0a = '#cd6578'
col_16p0b = '#332288'
col_16p4a = '#016d25'
col_16p4b = '#88cbf0'
col_16p7 = '#82144a'
col_17p0 = '#43aa97'
col_17p2 = '#999932'
col_17p4 = '#aa4498'
col_17p7 = '#ca6678'

fig = plt.figure(figsize=(5, 6))
ax = plt.gca()
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range), color='black')
plt.plot(wave_range, flux_range + global_range, color='grey')

plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[True, False, False, False, False, False, False, False, False]), color=col_16p0a, label='16.0A')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, True, False, False, False, False, False, False, False]), color=col_16p0b, label='16.0B')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, True, False, False, False, False, False, False]), color=col_16p4a, label='16.4A')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, True, False, False, False, False, False]), color=col_16p4b, label='16.4B')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, True, False, False, False, False]), color=col_16p7, label='16.7')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, True, False, False, False]), color=col_17p0, label='17.0')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, False, True, False, False]), color=col_17p2, label='17.2')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, False, False, True, False]), color=col_17p4, label='17.4')
plt.plot(wave_range, global_range + profile_fit.produce_best_fit_curve(wave_range, component_mask=[False, False, False, False, False, False, False, False, True]), color=col_17p7, label='17.7')

ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 6))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=7, width=1, direction='in')
ax.tick_params(axis='x', which='minor', labelbottom=True, top=True, length=5, width=1, direction='in')
ax.tick_params(axis='y', which='major', labelleft=True, right=True, length=7, width=1, direction='in')
ax.tick_params(axis='y', which='minor', labelleft=True, right=True, length=5, width=1, direction='in')

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)

plt.title(r'16-18 $\mu$m decomposition - Lorentzian')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$F_\nu$ (MJy sr$^{-1}$)')

plt.xlim(15.5, 18.3)
plt.ylim(125, 1150)
plt.legend()
# plt.show()
plt.savefig(f'fig_4_reproduction_lorentzianprofile', dpi=300, bbox_inches='tight', pad_inches=0.1)


