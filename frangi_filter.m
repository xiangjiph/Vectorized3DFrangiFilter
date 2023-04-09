function [vesselness] = frangi_filter(inputArray, opts)
% fun_multiscale_Frangi_vesselness computes the Frangi vesselness of each
% voxel on multiple scales, according to the method described in Frangi et
% al 1998
% 
% Input:
%     inputArray: 3D array of vasculature image, with isotropic voxel size 
%     vessel_parameters: struct, with field: DoG_scale_list, alpha, beta,
%     gamma, c
%           Dog_scale_list: float vector, standard deviations of the
%           gaussian filters used in the multiscale frangi filter 
%           alpha: float scalar, the alpha parameter used in the Frangi
%           filter
%           beta: float scalar, the alpha parameter used in the Frangi
%           filter
%
% Reference: 
%   Frangi et al 1998 Multiscale Vessel Enhancement Filtering
%
% Implemented by Xiang Ji, UC San Diego

% Set default values
if nargin < 2
    opts = struct;
end
if ~isfield(opts, 'DoG_scale_list')
    opts.DoG_scale_list = [4, 8, 16];
end
if ~isfield(opts, 'alpha')
    opts.alpha = 0.5;
end
if ~isfield(opts, 'beta')
    opts.beta = 0.5;
end

alpha = 2 * (opts.alpha) ^ 2;
beta = 2 * (opts.beta) ^ 2;

% Single precision is faster and accurate enough
input_gpuArrayQ = isa(inputArray, 'gpuArray');
if input_gpuArrayQ 
    if ~isaUnderlying(inputArray, 'float')
        inputArray = single(inputArray);
    end
elseif ~isa(inputArray, 'float')
    inputArray = single(inputArray);
end

image_size = size(inputArray);
vesselness = zeros(image_size, 'like', inputArray);

inputArray = rescale(inputArray);

for sigma_idx = 1 : numel(opts.DoG_scale_list)
    if opts.DoG_scale_list(sigma_idx) > 0
        if input_gpuArrayQ
            % Gaussian filter is faster in spatial domain when run on GPU 
            I_smoothed = imgaussfilt3(inputArray, opts.DoG_scale_list(sigma_idx), 'FilterDomain', 'spatial');
        else
            I_smoothed = imgaussfilt3(inputArray, opts.DoG_scale_list(sigma_idx));
        end
    else
        I_smoothed = inputArray;
    end    
    %% Run on CPU
    if ~input_gpuArrayQ
        [tmpA11, tmpA12, tmpA13, tmpA22, tmpA23, tmpA33] = gamma_normalized_hessian_3D(I_smoothed, opts.DoG_scale_list(sigma_idx), 1);
        [tmpEig1, tmpEig2, tmpEig3] = eig_3x3_real_sym(tmpA11, tmpA12, tmpA13, tmpA22, tmpA23, tmpA33);
        tmp_signal_measure = tmpEig1.^2 + tmpEig2.^2 + tmpEig3.^2;
        frangi_c = 0.5 * max(tmp_signal_measure(:));
        % disp(frangi_c);
        tmp_signal_measure = (1 - exp(- (tmp_signal_measure) ./ frangi_c));
        tmp_vesselness =  (1 - exp(- ((tmpEig2./tmpEig3).^2 ) ./ alpha)) ... % deviation from plane: 1 means plane
            .* exp(-( tmpEig1.^2 ./abs(tmpEig2 .* tmpEig3) ) ./ beta) ... % deviation from blob: 1 means eig1 << eig2 * eig3 -> plane or rod
            .* tmp_signal_measure;
        tmp_vesselness( isnan(tmp_vesselness) ) = 0;
        tmp_vesselness = tmp_vesselness - tmp_vesselness .* (tmpEig2>=0 | tmpEig3 >=0);
    else
    %% Completely run on GPU
        [tmpA11, tmpA12, tmpA13, tmpA22, tmpA23, tmpA33] = gamma_normalized_hessian_3D(I_smoothed, opts.DoG_scale_list(sigma_idx),1);
        tmp_vesselness = arrayfun(@frangi_filter_single_scale_gpu, tmpA11, tmpA12, tmpA13, tmpA22, tmpA23, tmpA33);
    end
%%
    vesselness = max(vesselness, tmp_vesselness);
end
end