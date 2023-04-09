clc;clear;close all;
vsl_im = load('vessel_3d_image.mat');
% 240 x 240 x 240 3D vascular image at 1 x 1 x 1 um3 voxel resolution
vsl_im = vsl_im.vsl_im;
%% 
vessel_parameters = struct;
vessel_parameters.DoG_scale_list = [1, 2, 4, 8, 16];
vessel_parameters.alpha = 0.5;
vessel_parameters.beta = 0.5;
% On CPU: 6.1 seconds 
t_tic = tic;
frangi_cpu = frangi_filter(vsl_im, vessel_parameters);
fprintf("This implementation on CPU takes %.2f seconds\n", toc(t_tic))
% On GPU: 1.1 second
im_gpu = gpuArray(vsl_im);
t_tic = tic;
frangi_gpu = frangi_filter(im_gpu, vessel_parameters);
fprintf("This implementation on GPU takes %.2f seconds\n", toc(t_tic));
frangi_gpu = gather(frangi_gpu);