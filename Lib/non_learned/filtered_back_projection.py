import sys,os
sys.path.insert(0,'..')
"""Filtered back projection reference"""

import numpy as np
import odl

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


with odl.util.Timer('runtime of FBP reconstruction'):
    x = fbp(sinogram_data)

print('ssim = {}'.format(ssim(phantom.asarray(), x.asarray())))
print('psnr = {}'.format(psnr(phantom.asarray(), x.asarray(), data_range=np.max(phantom) - np.min(phantom))))


figure_folder = 'Results/'
x.show('', clim=[0, 1], saveto=figure_folder + 'shepp_logan_phantom_fbp')
