% Add the DataHash function to the MATLAB path

% Read the images
host_img = imread('img.jpg');
watermark1 = imread('watermark1.jpeg');

% Convert the watermark image to grayscale
watermark1_gray = rgb2gray(watermark1);

% Display the single watermark
imshow(watermark1_gray); 
title('Single Watermark');

% Hash the watermark using SHA-256 directly
hash_value = DataHash(watermark1_gray, 'SHA-256');
disp(['SHA-256 Hash: ', hash_value]);

% Convert hash to chaotic sequence initial conditions
key_stream = zeros(1, 8);
for i = 1:8
    % Extract each block of 8 characters from hash_value
    block = hash_value((i-1)*8+1:i*8);
    
    % Convert each character in the block to ASCII value
    ascii_values = double(block);
    
    % Compute the mean of ASCII values
    key_stream(i) = (10^(-15))*mean(ascii_values);
end

% Initial conditions for chaotic system
x0 = 1 + key_stream(1) + key_stream(5);
y0 = 1 + key_stream(2) + key_stream(6);
z0 = 1 + key_stream(3) + key_stream(7);
w0 = 1 + key_stream(4) + key_stream(8);
u0 = 1 + key_stream(5) + key_stream(8);

% Generate chaotic sequences
m = 5000;
len = numel(watermark1_gray);  % Ensure length matches the watermark
[x_seq, y_seq, z_seq, w_seq, u_seq] = chaotic_sequence(x0, y0, z0, w0, u0, len, m);

% Sorting the chaotic sequence x_seq in descending order and recording the index
[sorted_x, index] = sort(x_seq(1:len), 'descend'); % Limit index to length of watermark

% Ensure index matches the size of the watermark
scrambled_watermark = watermark1_gray;
scrambled_watermark(:) = watermark1_gray(index);

% Reshape chaotic sequence y_seq to match the watermark size
y_seq_reshaped = reshape(y_seq(1:len), size(watermark1_gray));

% Diffuse and encrypt the scrambled watermark using chaotic sequence y_seq
ciphertext = bitxor(uint8(scrambled_watermark), uint8(y_seq_reshaped));

% Reshape the ciphertext to the original watermark size
[wm_height, wm_width] = size(watermark1_gray);
encrypted_watermark = reshape(ciphertext, [wm_height, wm_width]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% LWT   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decompose the host image into its color components
R = double(host_img(:,:,1));
G = double(host_img(:,:,2));
B = double(host_img(:,:,3));

wavelet_type = 'db2';
level = 1; 

% Apply LWT to each component
[LL_R, ~] = lwt2(R,'Wavelet', wavelet_type, Level=level,Int2Int=true);
[LL_G, ~] = lwt2(G,'Wavelet', wavelet_type, Level=level,Int2Int=true);
[LL_B, ~] = lwt2(B,'Wavelet', wavelet_type, Level=level,Int2Int=true);

% Display low-frequency components
figure;
subplot(1,3,1); imshow(LL_R, []); title('Low-frequency R');
subplot(1,3,2); imshow(LL_G, []); title('Low-frequency G');
subplot(1,3,3); imshow(LL_B, []); title('Low-frequency B');
%%%%%%%%%%%%%%%%%%%%%%%%%%%% SVD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block size for SVD
block_size = 4;

% Function to divide matrix into blocks and perform SVD
[Tk_R, Tk_G, Tk_B] = apply_svd(LL_R, LL_G, LL_B, block_size, z_seq);

% Combine singular values into a feature matrix
feature_matrix = cat(3, Tk_R, Tk_G, Tk_B);

% Display feature matrix
figure;
imshow(feature_matrix, []); title('Feature Matrix');

% Binarize the feature matrix
binary_feature_matrix = binarize_feature_matrix(feature_matrix);

% Display binary feature matrix
figure;
imshow(binary_feature_matrix, []); title('Binary Feature Matrix');

size_binary = size(binary_feature_matrix);
size_encrypted = size(encrypted_watermark);

% Ensure dimensions match
if ~isequal(size_binary, size_encrypted)
    % Reshape binary_feature_matrix if sizes don't match
    binary_feature_matrix = imresize(binary_feature_matrix, size_encrypted);
end

% Perform XOR operation
final_zero_watermark = xor(binary_feature_matrix, encrypted_watermark);
final_zero_watermark = logical(final_zero_watermark); % Convert to logical if binary

% Ensure final_zero_watermark is two-dimensional
[m, n, p] = size(final_zero_watermark);

% Reshape the 3D matrix to a 2D matrix
% The resulting matrix will have dimensions [m, n*p]
final_zero_watermark_2D = reshape(final_zero_watermark, m, n * p);

% Display using imshow
figure;
imshow(final_zero_watermark_2D); % Display grayscale image with automatic intensity scaling
title('Final Zero-Watermark');

% Scramble and diffuse using chaotic sequences

% Obtain current timestamp
timestamp = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');

% Register in Intellectual Property Database
registration_data = struct('Key', hash_value, 'Timestamp', timestamp, 'ZeroWatermark', final_zero_watermark);
disp('Registration Data:');
disp(registration_data);

% Save registration data
save('registration_data.mat', 'registration_data');

function [x_seq, y_seq, z_seq, w_seq, u_seq] = chaotic_sequence(x0, y0, z0, w0, u0, len, m)
    % Chaotic sequence generation using given initial conditions and length
    x_seq = zeros(1, len);
    y_seq = zeros(1, len);
    z_seq = zeros(1, len);
    w_seq = zeros(1, len);
    u_seq = zeros(1, len);
    
    x_seq(1) = x0;
    y_seq(1) = y0;
    z_seq(1) = z0;
    w_seq(1) = w0;
    u_seq(1) = u0;
    
    for i = 2:(m + len)
        x_seq(i) = mod(3.99 * x_seq(i-1) * (1 - x_seq(i-1)), 1);
        y_seq(i) = mod(3.99 * y_seq(i-1) * (1 - y_seq(i-1)), 1);
        z_seq(i) = mod(3.99 * z_seq(i-1) * (1 - z_seq(i-1)), 1);
        w_seq(i) = mod(3.99 * w_seq(i-1) * (1 - w_seq(i-1)), 1);
        u_seq(i) = mod(3.99 * u_seq(i-1) * (1 - u_seq(i-1)), 1);
    end
    
    x_seq = x_seq(m+1:m+len);  % Ensure sequence length matches len
    y_seq = y_seq(m+1:m+len);  % Ensure sequence length matches len
    z_seq = z_seq(m+1:m+len);  % Ensure sequence length matches len
    w_seq = w_seq(m+1:m+len);  % Ensure sequence length matches len
    u_seq = u_seq(m+1:m+len);  % Ensure sequence length matches len
end

function [Tk_R, Tk_G, Tk_B] = apply_svd(LL_R, LL_G, LL_B, block_size, z_seq)
    [rows, cols] = size(LL_R);
    Tk_R = zeros(rows, cols);
    Tk_G = zeros(rows, cols);
    Tk_B = zeros(rows, cols);

    % Iterate over blocks
    for i = 1:block_size:rows
        for j = 1:block_size:cols
            % Ensure the block indices stay within bounds
            if i+block_size-1 <= rows && j+block_size-1 <= cols
                % Extract blocks
                block_R = LL_R(i:i+block_size-1, j:j+block_size-1);
                block_G = LL_G(i:i+block_size-1, j:j+block_size-1);
                block_B = LL_B(i:i+block_size-1, j:j+block_size-1);

                % Apply SVD
                [~, S_R, ~] = svd(block_R);
                [~, S_G, ~] = svd(block_G);
                [~, S_B, ~] = svd(block_B);

                % Correct with chaotic sequence
                S_R(1,1) = S_R(1,1) + z_seq(i+j);
                S_G(1,1) = S_G(1,1) + z_seq(i+j);
                S_B(1,1) = S_B(1,1) + z_seq(i+j);

                % Store maximum singular values
                Tk_R(i:i+block_size-1, j:j+block_size-1) = S_R(1,1);
                Tk_G(i:i+block_size-1, j:j+block_size-1) = S_G(1,1);
                Tk_B(i:i+block_size-1, j:j+block_size-1) = S_B(1,1);
            end
        end
    end
end

function binary_feature_matrix = binarize_feature_matrix(feature_matrix)
    [M, N, C] = size(feature_matrix);
    binary_feature_matrix = zeros(M, N, C);

    for k = 1:C
        feature_vec = feature_matrix(:,:,k);
        binary_vec = zeros(size(feature_vec));

        % Binarize each column of the feature matrix
        for j = 1:N
            column = feature_vec(:, j);
            binary_vec(:, j) = column > mean(column); % Example binarization condition
        end

        binary_feature_matrix(:,:,k) = binary_vec;
    end
end

function final_zero_watermark = scramble_diffuse(zero_watermark, x_seq, y_seq)
    % Scramble zero-watermark
    [~, scramble_indices] = sort(x_seq, 'descend');
    scrambled_watermark = zero_watermark(scramble_indices);

    % Diffuse zero-watermark
    final_zero_watermark = bitxor(scrambled_watermark, uint8(y_seq));
end