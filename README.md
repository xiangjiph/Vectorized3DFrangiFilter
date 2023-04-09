# Vectorized3DFrangiFilter
This is an efficient implementation of the 3D Frangi filter in MATLAB, capable of running on both CPU and GPU. The frangi_filter function calculates the Frangi vesselness response in accordance with the method proposed by Frangi et al. (1998) for 3D images. This involves the following steps:

1. Calculation of the gamma-normalized gradient of the image after Gaussian filtering on multiple scales.
2. Computation of the Hessian matrix for each voxel in the image at each scale, followed by the computation of the three eigenvalues of the 3 x 3 Hessian matrix. These eigenvalues are then combined to obtain the local vesselness measure.
3. Calculation of the Frangi vesselness response for each voxel as the maximum of the responses obtained in step 2 over all scales.

For 3D vascular images, this process can be computationally intensive. In this implementation, we have leveraged the analytical formula for computing the eigenvalues of a 3 x 3 real symmetric matrix (Smith 1961) and implemented a vectorized version of the Frangi filter on both CPU and GPU in MATLAB.

In the `Demo.m` file, we demonstrate the computation of the Frangi vesselness for a 240 x 240 x 240 vascular image at 1 x 1 x 1 µm³ resolution. On our laptop workstation, this process takes 6.1 seconds and 1.1 seconds on the CPU and GPU, respectively.

Note that for optimal performance, the parameter `c` described in `Equation 13` in Frangi et al. (1998) must be manually adjusted in `frangi_filter_single_scale_gpu`. This value can be determined by running the computation on the CPU for some test datasets and printing `frangi_c` for multiple scales by uncommenting `disp(frangi_c);` in Line 71 of `frangi_filter`.

This implementation is part of the image segmentation algorithm we use to segment whole mouse brain vascular images, which contain about one trillion voxels. If you use this implementation in your research, please reference the following paper:

## Reference

    @article{Ji2021,
       title={Brain microvasculature has a common topology with local differences in geometry that match metabolic load},
       author={Ji, Xiang and Ferreira, Tiago and Friedman, Beth and Liu, Rui and Liechty, Hannah and Bas, Erhan and Chandrasheka, Jayaram and Kleinfeld, David}, 
       journal={Neuron},
       volume={109},
       number={7},
       pages={P1168-1187.E13},
       year={2021},
       publisher={Elsevier},
       url={https://doi.org/10.1016/j.neuron.2021.02.006}}