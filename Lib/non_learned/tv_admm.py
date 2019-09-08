import sys,os
sys.path.insert(0,'..')

"""Reference TV reconstruction using ADMM."""

import numpy as np
import odl
import odl.contrib.solvers.spdhg as spdhg #for total variation

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


np.random.seed(0)

#Define problem size and disretized geometry for CT scan
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=60)

#Create Radon Transform Function
radon_transform_operator = odl.tomo.RayTransform(space, geometry)

# Initalise Filtered Back Projection Function to use as starting point
fbp = odl.tomo.fbp_op(radon_transform_operator)


# Create Shepp Longon phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create Forward projections of phantom with white gaussian noise
sinogram_data = radon_transform_operator(phantom)
sinogram_data += odl.phantom.white_noise(radon_transform_operator.range) * np.mean(np.abs(sinogram_data)) * 0.05


#Create L2 data fidelity functional  
g = odl.solvers.L2NormSquared(radon_transform_operator.range).translated(sinogram_data)

#Create total variation regularisation functional 
f  = spdhg.TotalVariationNonNegative(space, alpha=10)

# Estimate of Radon Transform norm and add 10% to ensure tau  < sigma /||K||_2^2 
op_norm =  1.1*odl.power_method_opnorm(radon_transform_operator)

niter = 100

# Problem specific Parameters

sigma = 0.8  # Step size for the dual variable
tau = sigma  / (op_norm ** 2) # Step size for the primal variable
gamma = 0.99

# Choose the Filtered Back Projection as a starting point to improve reconstruction
x = fbp(sinogram_data)

with odl.util.Timer('runtime of iterative algorithm'):
    # Run the algorithm
    odl.solvers.admm_linearized(
        x, f, g, radon_transform_operator, tau=tau, sigma=sigma, niter=niter,
         gamma=gamma)

print('ssim = {}'.format(ssim(phantom.asarray(), x.asarray())))
print('psnr = {}'.format(psnr(phantom.asarray(), x.asarray(), data_range=np.max(phantom) - np.min(phantom))))

#Plot figure and save to results folder 
figure_folder = 'Results/'
x.show('', clim=[0, 1], saveto=figure_folder + 'shepp_logan_tv_ADMM')
x.show('', clim=[0.1, 0.4], saveto=figure_folder + 'shepp_logan_tv_windowed_ADMM')



