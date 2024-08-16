%% Helper Functions

addpath('desc_GLTP.m');
addpath('desc_LTeP.m');
addpath('ct_gridHist.m');
addpath('desc_LDN.m');
addpath('descriptor_LDN.m')
addpath('ECC_Encryption.m');
addpath('ECC_Decryption.m');
addpath('ECC_Add.m');
addpath('ECC_modfrac.m');
addpath('ECC_NP.m');
addpath('pso.m');
addpath('psnr.m');
addpath('Pso_Fitness.m');

%% Step : Read the Image

img = imread('image.jpg');
img = imresize(img,[512 512]);
if size(img, 3) == 3
    img = rgb2gray(img);
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

%% Step : Watermark Processing

watermarkImage = imread('E:\Programs Solved\SummerInternship\watermarks\watermark_5.jpeg');
watermarkImage = imresize(watermarkImage,[64 64]);
watermarkImage = im2gray(watermarkImage);


figure;
imshow(watermarkImage);
title('Original Binary Watermark Image');

%% Step : Encrypting Watermark

inputImage = watermarkImage;
if size(inputImage, 3) == 3
    inputImage = rgb2gray(inputImage);
end
inputImage = double(inputImage);

r = pso(inputImage);

a = -7;
b = 10;
p = 487;
G = [13, 46];



[R_pubX, R_pubY] = ECC_NP(a, b, p, r, G(1), G(2));
R_pub = [R_pubX, R_pubY];

[cipherImage, C2] = ECC_Encryption(inputImage, r, R_pub, [a, b, p], G);
imshow(cipherImage,[]);
title('Encrypted Watermark using ECC');



%% Step : Perform XOR Operation Between the Binary Watermark and Perceptual Hash

perceptual_hash_repeated = repmat(perceptual_hash, size(cipherImage, 1), 1);
ZeroWatermark = (cipherImage+perceptual_hash_repeated);

%% Display Zerowatermark
figure;
imshow(ZeroWatermark, []);
title('Zero Watermark Image After XOR Operation');
save('Zerowatermark.mat','ZeroWatermark', 'C2', 'r', 'R_pub');

