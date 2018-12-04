#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:51:42 2018

@author: marcorax

This file is a modified version of HOTS_Sparse_Libs 
with some differencies to work with channeled data, as building time surfaces,
with reference timestamps only from a referenced channel, 
and looking at past events rather than upcoming ones 
 
The original repo can be found on :
https://github.com/neuromorphic-paris/Sparse-HOTS

This file contains the HOTS_Sparse_Net class complementary functions that are used
inside the network with have no pratical use outside the context of the class
therefore is advised to import only the class itself and to use its method to perform
learning and classification

To better understand the math behind classification an optimization please take a look
to the original Sparse Hots paper and the new one by myself

The links:
 
"""
import numpy as np 
from scipy import optimize
from scipy.spatial import distance 
import matplotlib.pyplot as plt
import seaborn as sns
from Libs.Time_Surface_generators import Time_Surface_event

# Error functions and update basis (for online algorithms)
# =============================================================================
# The notation is the same as rapresented in the paper:
# a_j : activations/coefficients aj of the net
# Phi_j: the basis of each layer
# lam_aj: Lambda coefficient for aj sparsification
# lam_phi: Lambda coefficient for Phi_j regularization
# S: Input time surface
# S_tilde: reconstructed time surface
# =============================================================================
def error_func(a_j, S, Phi_j, lam_aj):
    S_tilde = sum([a*b for a,b in zip(Phi_j,a_j)])
    return np.sum((S-S_tilde)**2) + lam_aj*np.sum(np.abs(a_j))

def error_func_deriv_a_j(a_j, S, Phi_j, lam_aj):
    S_tilde = sum([a*b for a,b in zip(Phi_j,a_j)])
    residue = S-S_tilde
    return [-2*np.sum(np.multiply(residue,a)) for a in Phi_j] + lam_aj*np.sign(a_j)

# Online basis updating but with normalization on the values instead of hard 
# Thresholding    
def update_basis_online(Phi_j, a_j, eta, S, lam_phi, max_steps, precision):
    n_steps = 0 
    previous_err =  error_func(a_j, S, Phi_j, 0)
    err = previous_err + precision + 1
    while (abs(err-previous_err)>precision) & (n_steps<max_steps):
        residue = S-Phi_j
        for i in range(len(Phi_j)):
            Phi_j[i] = Phi_j[i] + residue[i]*eta*a_j[i] -eta*lam_phi*Phi_j[i]
        previous_err = err
        err =  error_func(a_j, S, Phi_j, 0)
        n_steps += 1

# Impose basis values between[1 0] rather than regularize, useful to reduce 
# parameters and improve stability at expenses of precision
def update_basis_online_hard_treshold(Phi_j, a_j, eta, S, max_steps, precision):
    n_steps = 0 
    previous_err =  error_func(a_j, S, Phi_j, 0)
    err = previous_err + precision + 1
    while (abs(err-previous_err)>precision) & (n_steps<max_steps):
        residue = S-Phi_j
        for i in range(len(Phi_j)):
            newbase = Phi_j[i] + residue[i]*eta*a_j[i]
            newbase[newbase>=1] = 1
            newbase[newbase<=0] = 0
            Phi_j[i] = newbase
        previous_err = err
        err =  error_func(a_j, S, Phi_j, 0)
        n_steps += 1        

# Error functions (for offline algorithbatch_surfacesms)
# These functions are developed to work on multiple data per time (a whole dataset)
# Notice that Error functions for the basis (here implemented as matrices) are 
# generally meant to work with scipy.optimize.minimize therefore reshaping
# has to be performed
# =============================================================================
# The notation is the same as rapresented in the paper:
# a_j : activations/coefficients aj of the net
# Phi_j: the basis of each layer
# Phi_j_dim: 
# lam_phi: Lambda coefficient for Phi_j regularization
# S_list: Input time surfaces used for learning 
# S_tilde: reconstructed time surface
# =============================================================================
def error_func_phi_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi):
    #scipy optimize minimze will input a monodimensional array
    #for the Phi_j, thus i need to reconstruct the matrixes 
    Phi_j = np.reshape(Phi_j, (Phi_j_num, Phi_j_dim_x, Phi_j_dim_y))    
    err = 0
    for k in range(len(S_list)):
        S_tilde = sum([a*b for a,b in zip(Phi_j,a_j[k])])
        err += np.sum((S_list[k]-S_tilde)**2) 
    result = err + lam_phi*np.sum(Phi_j**2)
    #as scipy performs operations on monodimensional arrays i need to flatten
    #the result
    return result.flatten()

def error_func_phi_grad_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi):
    #scipy optimize minimze will input a monodimensional array
    #for the Phi_j, thus i need to reconstruct the matrixes  
    Phi_j = np.reshape(Phi_j, (Phi_j_num, Phi_j_dim_x, Phi_j_dim_y))    
    #initialize the gradient result
    grad = []
    for i in range(len(Phi_j)):
        grad.append(np.zeros(Phi_j[0].shape))
    grad = np.array(grad)
    for k in range(len(S_list)):
        S_tilde = sum([a*b for a,b in zip(Phi_j,a_j[k])])
        residue = S_list[k]-S_tilde
        grad += [a*residue for a in a_j[k]]
    result = -2*grad + 2*lam_phi*Phi_j
    #as scipy performs operations on monodimensional arrays i need to flatten
    #the result
    return result.flatten()

# Function that update basis with conjugate gradient method
def update_basis_offline_CG(Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, a_j, S_list, lam_phi):
    # Error functions and derivatives for gradient descent
    Err = lambda Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, a_j, S_list, lam_phi: error_func_phi_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi)
    dErrdPhi = lambda Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, a_j, S_list, lam_phi: error_func_phi_grad_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi)
    res = optimize.minimize(fun=Err, x0=Phi_j.flatten(), args=(Phi_j_dim_x, Phi_j_dim_y,
                             Phi_j_num, a_j, S_list, lam_phi),
                             method="CG", jac=dErrdPhi)
    #Phi_j = res.x.reshape(..) won't work because the returned Phi_j is not the 
    #reference to the original one, therefore it cannot update implicitly self.basis[0]
    return res

# Function for computing events out of net activations, the linear relation
# between the activity value a_j and the obtained timestamp is mediated through 
# a delay_coefficient (namely alpha in the paper)
# =============================================================================
# activations : An array containing all the activations of a sublayer given a
#               batch 
# events : the reference events that generated the input timesurfaces
# delay_coeff : the delay coefficient called alpha in the original paper
#    
# It returns on and off events with polarities defined as the index of the base 
# generating them, do not confuse them with the on off label. 
# The two matrices are : on_events and off_events. They are composed of 3 columns:
# The first column is the timestamp array, the second column is a [2*N] array,
# with the spatial positions of each events. Lastly the third column stores
# the previously mentioned polarities 
# =============================================================================    
def events_from_activations(activations, events, delay_coeff):
    out_timestamps_on = []
    out_positions_on = []
    out_polarities_on = []
    out_timestamps_off = []
    out_positions_off = []
    out_polarities_off = []
    for i in range(len(events)):
        t = events[i][0]
        for j,a_j in enumerate(activations[i]):
            tout = t + delay_coeff*(1-abs(a_j))
            if (a_j > 0):
                out_timestamps_on.append(tout)
                out_positions_on.append(events[i][1])
                out_polarities_on.append(j)
            if (a_j < 0):
                out_timestamps_off.append(tout)
                out_positions_off.append(events[i][1])
                out_polarities_off.append(j)

    # The output data need to be sorted like the original dataset in respect to
    # the timestamp, because the time surface functions are expecting it to be sorted.
    out_timestamps_on = np.array(out_timestamps_on)
    out_positions_on = np.array(out_positions_on)
    out_polarities_on = np.array(out_polarities_on)
    out_timestamps_off = np.array(out_timestamps_off)
    out_positions_off = np.array(out_positions_off)
    out_polarities_off = np.array(out_polarities_off)

    sort_ind = np.argsort(out_timestamps_on)
    out_timestamps_on = out_timestamps_on[sort_ind]
    out_positions_on = out_positions_on[sort_ind]
    out_polarities_on = out_polarities_on[sort_ind]
    on_events = [out_timestamps_on, out_positions_on, out_polarities_on]

    sort_ind = np.argsort(out_timestamps_off)
    out_timestamps_off = out_timestamps_off[sort_ind]
    out_positions_off = out_positions_off[sort_ind]
    out_polarities_off = out_polarities_off[sort_ind]
    off_events = [out_timestamps_off, out_positions_off, out_polarities_off]

    return on_events, off_events



# Handy functions used to plot online graphs at each base update if verbose=True             
def create_figures(surfaces, num_of_plots, fig_names="Base N : "):
    figures = []
    axes = []
    # I am plotting several network basis with generated figure names
    if fig_names=="Base N : ":
        for i in range(num_of_plots):
            fig = plt.figure(fig_names+str(i))
            ax = sns.heatmap(surfaces[i], annot=False, cbar=False, vmin=0, vmax=1)
            figures.append(fig)
            axes.append(ax)
    # I am plotting a single general tsurface 
    else:
        fig = plt.figure(fig_names)
        ax = sns.heatmap(surfaces, annot=False, cbar=False, vmin=0, vmax=1)
        figures.append(fig)
        axes.append(ax)
    return figures, axes

def update_figures(figures, axes, surfaces):
    if len(figures) is 1:
        axes[0].clear()
        sns.heatmap(data=surfaces, ax=axes[0], annot=False, cbar=False, vmin=0, vmax=1)
    else:
        for i in range(len(figures)):
            axes[i].clear()
            sns.heatmap(data=surfaces[i], ax=axes[i], annot=False, cbar=False, vmin=0, vmax=1)

# Function to compute and exponential decay
# =============================================================================
# initial_value : the value of the function at t0
# final_value : the result at steady state
# time_decay : the time coefficient for exp decay functions, how quickly
#              the function will head for steady state
# time_array : the "x" coordinate of the function with the time values
# result : the "y" resulting coordinates of the function
# =============================================================================
def exp_decay(intial_value, final_value, time_decay, time_array):
    result = ((intial_value-final_value)*np.exp(-(time_array)/time_decay))+final_value
    return result
    