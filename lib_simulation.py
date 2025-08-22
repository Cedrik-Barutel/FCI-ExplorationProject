#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cedrik

Updated on 22 08 2025
"""

#%%
# def multiplie_nombres(nombre1, nombre2):
#     """Multiplication de deux nombres entiers.

#     Cette fonction ne sert pas à grand chose.

#     Parameters
#     ----------
#     nombre1 : int
#         Le premier nombre entier.
#     nombre2 : int
#         Le second nombre entier,
#         très important pour cette fonction.

#     Returns
#     -------
#     int
#         Le produit des deux nombres.
#     """
#     return nombre1 * nombre2
#%%



import numpy as np
from scipy.interpolate import UnivariateSpline


def function_filament(F,xl,xr,x,li):
    """ calculating the phase field filaments between xl and xr with and interface li.

    Parameters
    ----------
    F : array(float)
        array of the filament 
    xl : float
        left boundary
    xr : float
        right boundary
    x : array(float)
        array of the space
    li : float
        length of phase field interface       
        
    Returns
    -------
    array(float)
        Phase field function of the filament
    """
    F = 1/(1+np.exp(-6/li*(x-xl))) * 1/(1+np.exp(6/li*(x-xr)))
    return F


def function_step(v,delta_t,t_i,delta_l,t):
    """ adding a smooth step to a function

    Parameters
    ----------
    f : array(float)
        input function array
    delta_t : float
        duration of the step
    t_i : float
        time of the step
    delta_l : array(float)
        amplitude of the jump
    t : array(float)
        input time array      
        
    Returns
    -------
    array(float)
        return function array with a step at t_i
    """
    v = v + delta_l/delta_t*np.exp((t-t_i)/delta_t)/(1+np.exp((t-t_i)/delta_t))**2
    return v



def function_convertion_array(old_array, new_length):
    """ convert an array by interpolation 

    Parameters
    ----------
    old_array : array
        input array
    new_length : int
        desired length for input arary    
        
    Returns
    -------
    array
        return the input array with the new desired length
    """
    old_indices = np.arange(0,len(old_array))
    new_indices = np.linspace(0,len(old_array)-1,new_length)
    spl = UnivariateSpline(old_indices,old_array,k=3,s=0)
    return spl(new_indices)


class Coefficient:
    def __init__(self, coefficient_name, coefficient_value):
        self.n = coefficient_name
        self.v = coefficient_value

def function_save_parameters(writer,fieldnames,save_param):
    writer.writerow({fieldnames[0]: save_param.n, fieldnames[1]: save_param.v})




