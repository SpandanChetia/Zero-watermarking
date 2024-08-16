%% Feature Extraction Helper Functions
% 
% addpath('desc_GLTP.m');
% addpath('desc_LTeP.m');
% addpath('ct_gridHist.m');
% addpath('desc_LDN.m');
% addpath('descriptor_LDN.m')
% addpath('ECC_Encryption.m');
% addpath('ECC_Decryption.m');
% addpath('ECC_Add.m');
% addpath('ECC_modfrac.m');
% addpath('ECC_NP.m');
% addpath('pso.m');
% addpath('psnr.m');
% addpath('Pso_Fitness.m');

%% Step : Read the Image

img = imread('attackImage.jpg');
img = imresize(img,[512 512]);
if size(img, 3) == 3
    img = im2gray(img);
end

%% Step : Feature Extraction from Host Image

[LDN_hist, imgDesc] = desc_LDN(img);

%% Display Results
disp('LDN Feature Histogram:');
disp(LDN_hist);

figure;
imshow(imgDesc, []);
title('Descriptor Image');

%% Perceptual Hash Generation

mean_val = mean(LDN_hist);
perceptual_hash = zeros(size(LDN_hist));

for i = 1:length(LDN_hist)
    if LDN_hist(i) >= mean_val
        perceptual_hash(i) = 1;
    else
        perceptual_hash(i) = 0;
    end
end


%% Convert the perceptual hash to a string (optional)
hash_str = num2str(perceptual_hash);
hash_str = hash_str(~isspace(hash_str));

disp('Perceptual Hash (String):');
disp(hash_str);


%% Step : Load Zerowatermark and decrypt it

load('ZeroWatermark.mat','ZeroWatermark', 'C2', 'r', 'R_pub');

%% Step : Load Original Watermark
watermarkImage = imread('E:\Programs Solved\SummerInternship\watermarks\watermark_5.jpeg');
watermarkImage = imresize(watermarkImage,[64 64]);
watermarkImage = im2gray(watermarkImage);

figure;
imshow(watermarkImage);
title('Original Binary Watermark Image');



%% Step : Perform XOR Operation Between the ZeroWatermark and Perceptual Hash

perceptual_hash_repeated = repmat(perceptual_hash, size(ZeroWatermark, 1), 1);
encyExtractedWatermark = (ZeroWatermark-perceptual_hash_repeated);

%% Display Extracted Watermark

figure;
imshow(encyExtractedWatermark,[]);
title('Encrypted Extracted Watermark');

%% Step : Decrypt watermark

a = -7;
b = 10;
p = 487;
G = [13, 46];

decryptedImage = ECC_Decryption(encyExtractedWatermark, r, C2, [a, b, p], G);
imshow((decryptedImage),[]);
title('Decrypted Watermark');

ExtractedWatermark = decryptedImage;

%% Experimental Analysis

% Normalized Correlation 
NC = normalized_correlation(double(watermarkImage),double(ExtractedWatermark));
disp(['Normalized Correlation: ',num2str(NC)]);

% Calculate PSNR
psnr_value = psnr(double(watermarkImage),double(ExtractedWatermark));
fprintf('PSNR: %.4f dB\n', psnr_value);

% Calculate SSIM
ssim_value = ssim(double(watermarkImage),double(ExtractedWatermark));
fprintf('SSIM: %.4f\n', ssim_value);

% Calculate BER
[number,ratio] = biterr(double(watermarkImage),double(ExtractedWatermark));
fprintf('BER: %.4f%%\n', ratio);


%% Helper Functions

% Normalized Correlation function
function nc = normalized_correlation(W, W_prime)
    if size(W) ~= size(W_prime)
        error('The dimensions of the original and extracted watermarks must be the same.');
    end

    W = double(W);
    W_prime = double(W_prime);
    numerator = sum(sum(W .* W_prime));
    denominator = sqrt(sum(sum(W .^ 2)) * sum(sum(W_prime .^ 2)));
    nc = numerator / denominator;    
end

% function decryptedImage = ECCP_Decryption(cipherImage, k, R_pub, E, G)
%     [row, col] = size(cipherImage);
%     decryptedImage = zeros(row, col);
% 
%     a = E(1);
%     b = E(2);
%     p = E(3);
% 
%     C2X = R_pub(1);
%     C2Y = R_pub(2);
% 
%     [kC2X, kC2Y] = ECCP_NP(a, b, p, k, C2X, C2Y);
%     kC2Y = mod(-1 * kC2Y, p);
% 
%     for i = 1:col
%         for j = 1:2:row
%             [decryptedImage(i, j), decryptedImage(i, j + 1)] = ECCP_Add(a, b, p, cipherImage(i, j), cipherImage(i, j + 1), kC2X, kC2Y);
%         end
%     end
% end
% 
% function [resx, resy] = ECCP_NP(a, b, p, n, x, y)
%     if n == 1
%         resx = x;
%         resy = y;
%         return;
%     end
% 
%     if n >= 2
%         [xsub, ysub] = ECCP_NP(a, b, p, n - 1, x, y);
%         if xsub == 255 && ysub == 255
%             resx = 255;
%             resy = 255;
%         else
%             [resx, resy] = ECCP_Add(a, b, p, x, y, xsub, ysub);
%         end
%     end
% end
% 
% function [ resx,resy ] = ECCP_Add( a,b,p,x1,y1,x2,y2 )
%     if x1==x2 && y1==y2
%         n= 3*x1^2+a;
%         d= 2*y1;
%         if  ~isinf( n)|| ~isinf( d )  
%         m=ECCP_modfrac(n,d,p);
%         resx = mod(m^2-x1-x2,p);
%         resy = mod(m*(x1-resx)-y1,p);
%         else 
%             resx=255;
%             resy=255;
%         end
%     end
% 
%     if x1==x2 && y1~=y2
%         resx = 255;
%         resy = 255;
%     end
% 
%     if x1 ~= x2
%         n=y2-y1;
%         d=x2-x1;
% 
%         if  ~isinf( n)|| ~isinf( d )  
%         m=ECCP_modfrac(n,d,p);
%         resx = mod(m^2-x1-x2,p);
%         resy = mod(m*(x1-resx)-y1,p); 
%         else 
%             resx=255;
%             resy=255;
%         end
%     end
% end
% 
% 
% function y = ECCP_modfrac(n, d, m)
%     n = mod(n, m);
%     d = mod(d, m);
% 
%     i = 1;
%     while mod(d * i, m) ~= 1
%         i = i + 1;
%     end
% 
%     y = mod(n * i, m);
% end
