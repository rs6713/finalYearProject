function descriptors = MACH(images, options)
%% function Descriptors = MACH(images, options)
% Function for the machine learning feature extraction
%
% Input:
%   <images>: a set of n RGB color images. Size: [h, w, 3, n]

% Output:
%   descriptors: the extracted LOMO descriptors. Size: [d, n]
% 
% Example:
%     I = imread('../images/000_45_a.bmp');
%     descriptor = LOMO(I);

%% set parameters, check system

t0 = tic;
% Get GPU device information
deviceInfo = gpuDevice;

% Check the GPU compute capability
computeCapability = str2double(deviceInfo.ComputeCapability);
assert(computeCapability >= 3.0, ...
    'This example requires a GPU device with compute capability 3.0 or higher.')

%% extract Features: descriptors


%% finishing, clear temp vars, create descriptors
descriptors='hi';

feaTime = toc(t0);
meanTime = feaTime / size(images, 4);
fprintf('LOMO feature extraction finished. Running time: %.3f seconds in total, %.3f seconds per image.\n', feaTime, meanTime);
end



