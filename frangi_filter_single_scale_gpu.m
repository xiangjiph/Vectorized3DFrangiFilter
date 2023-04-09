function vesselness = frangi_filter_single_scale_gpu(A11,A12,A13,A22,A23,A33)
% frangi_filter_single_scale_gpu implements Frangi vesselness on a single
% scale on GPU. The function is used with @arrayfun and gpuArray. Using
% this function and @arrayfun without casting inputs to gpuArray has very
% poor performance. 
% Input:
%   A11: gpuArray, the (1,1) elements of the hessian matrix of all the pixels 
%   A12: gpuArray, the (1,2) elements of the hessian matrix of all the pixels 
%   A13: gpuArray, the (1,3) elements of the hessian matrix of all the pixels 
%   A22: gpuArray, the (2,2) elements of the hessian matrix of all the pixels 
%   A23: gpuArray, the (2,3) elements of the hessian matrix of all the pixels 
%   A33: gpuArray, the (3,3) elements of the hessian matrix of all the pixels 
% Output: 
%   vesselness: gpuArray, the Frangi vesselness response
% Reference: 
%   Smith, O.K. (1961). Eigenvalues of a symmetric 3 x 3 matrix. Commun.
%   ACM 4, 168
%   Frangi et al 1998 Multiscale Vessel Enhancement Filtering
%
% Implemented by Xiang Ji, UC San Diego
%% Frangi parameters
frangi_alpha = 0.5;
frangi_beta = 0.5;
frangi_c = 0.075; % This value should be adjusted accroding to dataset 
%% Compute eigenvalues
c = 2 * pi/3;
precision_error = eps('double');
p = A12 * A12 + A13 * A13 + A23 * A23;
eig1 = A11*0;
eig2 = A11*0;
eig3 = A11*0;
eig2_t = (A11 + A22 + A33)/3;
p = (A11 - eig2_t)^2 + (A22 - eig2_t)^2 + (A33 - eig2_t)^2 + 2 * p + precision_error;
p = sqrt(p/6);
A11 = (A11 - eig2_t)/p;
A22 = (A22 - eig2_t)/p;
A33 = (A33 - eig2_t)/p;
A12 = A12 / p;
A13 = A13 / p;
A23 = A23 / p;
eig3_t = 0.5 * ( A11 * A22 * A33 + A12 * A23 * A13 * 2 - ...
    (A13*A13) * A22  - (A12*A12) * A33 - (A23*A23) * A11);
eig3_t = acos( min(1, max(-1, eig3_t)) )/3;
eig1_t = eig2_t + 2 * p * cos(eig3_t);
eig3_t = eig2_t + 2 * p * cos(eig3_t + c);
eig2_t = 3 * eig2_t - eig1_t - eig3_t;
% Sort eigenvalues from small to large according to their absolute
abs_e1 = abs(eig1_t);
abs_e2 = abs(eig2_t);
abs_e3 = abs(eig3_t);

if abs_e1 < abs_e2
    if abs_e2 < abs_e3 % 1,2,3
        eig1 = eig1_t;
        eig2 = eig2_t;
        eig3 = eig3_t;
    elseif abs_e1 < abs_e3 % 1,3,2
        eig1 = eig1_t;
        eig2 = eig3_t;
        eig3 = eig2_t;
    else
        eig1 = eig3_t;% 3,1,2
        eig2 = eig1_t;
        eig3 = eig2_t;
    end
else% abs_e2 < abs_e1
    if abs_e1 < abs_e3
        eig1 = eig2_t;
        eig2 = eig1_t;
        eig3 = eig3_t;
    elseif abs_e2 < abs_e3   % abs_e2 < abs_e1 & abs_e3 < abs_e1
        eig1 = eig2_t;
        eig2 = eig3_t;
        eig3 = eig1_t;
    else
        eig1 = eig3_t;
        eig2 = eig2_t;
        eig3 = eig1_t;
    end
end
%% Compute Frangi vesselness
vesselness = eig1*0;
if eig2 >= 0 
    return;
elseif eig3 >= 0
    return;
else
    signal_level = eig1 * eig1 + eig2 * eig2 + eig3 * eig3;
    signal_level = 1 - exp(- signal_level/ (2 * frangi_c ^ 2));
    vesselness =  (1 - exp(- ((eig2/eig3)^2 ) / (2 * frangi_alpha ^2))) * ...
        exp(-( eig1^2 /abs(eig2 * eig3) ) / (2 * frangi_beta ^ 2)) * signal_level;
    if isnan(vesselness)
        vesselness = eig1*0;
    end
end
end
