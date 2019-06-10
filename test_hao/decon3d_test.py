"""Test the deconvolution in git@github.com:haoxusci/flowdec.git for 3d image stack"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from flowdec.psf import GibsonLanni
from skimage import io


PSF_PARAMETERS_PATH = '/home/haoxu/.repos_py3/deconvolution/from_flowdec/microscope_psf.json'
PSFSTACK_PATH = '/home/haoxu/Documents/deconvolution_test/psfstack.tif'
IMAGESTACK_PATH = '/home/haoxu/Documents/deconvolution_test/stack.tif'

Gibs = GibsonLanni()
Gibs.load(PSF_PARAMETERS_PATH)
kernel = Gibs.generate()
kernel = np.asarray(kernel, dtype=np.float32)
io.imsave(PSFSTACK_PATH, kernel)

data = io.imread(IMAGESTACK_PATH)

algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()
res = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=30).data

io.imsave('/home/haoxu/Documents/deconvolution_test/stack_deconv.tif', np.asarray(res, dtype=np.uint16))
