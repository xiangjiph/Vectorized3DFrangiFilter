function [A11, A12, A13, A22, A23, A33] = gamma_normalized_hessian_3D(A11, gf_sigma, gamma)
% gamma_normalized_hessian_3D computes the second order derivative of the
% gaussian-smothed version of data use central difference. 
% Input: 
%   A11: input 3D array, merge variable name with A11 for performance. 
%   gf_size: standard deviation of the gaussian filter. default value is 0
%   gamma: parameter for gamma-normalization
% Output: 
%   hessian_matrix: structure with fields:
%       gaussian_std: size of the gaussian filter
%       A11, A12,A13, A22, A23, A33: six matrix element of the hessian
%       matrix(which is symmetric), correspond to Ixx, Ixy, Ixz, Iyy, Iyz,
%       Izz
% Note:
%   The data transfer from the GPU to the CPU is the main performance
%   bottleneck for this function. However, computing graident on CPU is
%   much more time comsuming than on GPU. The overall performance is better
%   if the computation is done on GPU and then transfer back to CPU. 
%
% Implemented by Xiang Ji, UC San Diego
if nargin < 2
    gf_sigma = 1;
    gamma = 1;
elseif nargin < 3
    gamma = 1;
end
gamma_normalization_factor = gf_sigma ^ gamma;

if gamma_normalization_factor > 1 + eps
    [A11, A22, A33] = fun_gradient3D(A11.* gamma_normalization_factor,[1,2,3]);
    [A11, A12, A13] = fun_gradient3D(A11.* gamma_normalization_factor,[1,2,3]);
    [A22, A23] = fun_gradient3D(A22.* gamma_normalization_factor,[2,3]);
    A33 = fun_gradient3D(A33.* gamma_normalization_factor,3);
else
    [A11, A22, A33] = fun_gradient3D(A11,[1,2,3]);
    [A11, A12, A13] = fun_gradient3D(A11,[1,2,3]);
    [A22, A23] = fun_gradient3D(A22,[2,3]);
    A33 = fun_gradient3D(A33,3);
end

end

function varargout = fun_gradient3D(inputArray, dim, method, save_to_ramQ )
% fun_gradient3D compute the gradient along dim. Forward difference is used
% at the boundary, central difference is used for the internal points
% Input: 
%   inputArray: 3D numerical array
%   dim: scalar or vector specifying the dimension(s) along which the
%   gradient is computed. 
%   save_to_ramQ: if the inputArray is a gpuArray and save_to_ramQ is true,
%   gather all the result to RAM.
% Output: 
%   varargout: computed gradient(s) 
if nargin < 3
    save_to_ramQ = false;
    method = 'central';
end

if nargin < 4
    save_to_ramQ = false;
end

ndim = length(dim);
varargout = cell(1,ndim);
input_size = size(inputArray);


for dir_idx = 1 : ndim
    grad = zeros(input_size, 'like', inputArray);
    grad_dir = dim(dir_idx);
    switch grad_dir
        case 1
            switch method
                case 'central'
                    grad(1,:,:) = inputArray(2,:,:) - inputArray(1,:,:);
                    grad(input_size(1),:,:) = inputArray(input_size(1),:,:) - inputArray(input_size(1) - 1, :, :);
                    grad(2:input_size(1)-1, :,:) = (inputArray(3:input_size(1),:,:) - inputArray(1:input_size(1) - 2, :, :))./2;
                case 'intermediate'
                    grad(1:end-1,:,:) = diff(inputArray, 1, 1);
            end
        case 2
            switch method
                case 'central'
                    grad(:,1,:) = inputArray(:,2,:) - inputArray(:,1,:);
                    grad(:,input_size(2),:) = inputArray(:,input_size(2),:) - inputArray(:,input_size(2) - 1, :);
                    grad(:, 2:input_size(2) - 1,:) = (inputArray(:,3:input_size(2),:) - inputArray( :,1:input_size(2) - 2, :))./2;
                case 'intermediate'
                    grad(:, 1:end-1,:) = diff(inputArray, 1, 2);
                    
            end
        case 3
            switch method
                case 'central'
                    grad(:,:,1) = inputArray(:,:,2) - inputArray(:,:,1);
                    grad(:,:,input_size(3)) = inputArray(:,:,input_size(3)) - inputArray(:,:,input_size(3) - 1);
                    grad(:,:,2:input_size(3)-1) = (inputArray(:,:,3:input_size(3)) - inputArray(:,:,1:input_size(3) - 2))./2;
                case 'intermediate'
                    grad(:, :, 1:end-1) = diff(inputArray, 1, 3);
            end
    end
    
    if save_to_ramQ && isa(inputArray, 'gpuArray')
        varargout{dir_idx} = gather(grad);
    else
        varargout{dir_idx} = grad;
    end
    
end

end