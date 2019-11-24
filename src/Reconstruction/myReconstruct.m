function[] = myReconstruct(path, h, w, num_ids, num_train, num_test, img_idx)
% path = '../../../att_faces/s'; h= 112; w = 92; num_ids = 32; num_train =  6; num_test = 4;
d = h*w;
trainSet = zeros(h, w, num_train, num_ids);
testSet = zeros(h, w, num_test, num_ids);
meanArr = zeros(d, 1);

for i = 1:num_ids
    newpath = strcat(path, int2str(i));
    S = dir(fullfile(newpath, '*.pgm'));
    for j = 1:num_train
        trainSet(:, :, j, i) = double(imread(strcat(newpath, '/', S(j).name)));
    end
    for j = (num_train + 1):size(S, 1)
        testSet(:, :, j-num_train, i) = double(imread(strcat(newpath, '/', S(j).name)));
    end
end

meanArr(:, :) = reshape(double(sum(sum((trainSet(:, :, :, :)), 4), 3)./(num_train*num_ids)), [d, 1]);
temp = reshape(trainSet(:, :, :, :), [d, num_train*num_ids]);
A = temp - meanArr;
[U, ~, ~] = svd(A);
normU = vecnorm(U, 2, 1);
U = U./normU;

ks = [2, 10, 20, 50, 75, 100, 125, 150, 175]';
figure();

for i = 1:size(ks, 1)
    k = ks(i);
    eig_vec = U(:, 1:k);
    alpha_ = eig_vec' * A(:, img_idx);
    face = reshape((eig_vec * alpha_), [112, 92]);
    subplot(3, 3, i)
    imshow(face, [], 'InitialMagnification', 'fit');
    title(strcat('k = ', int2str(ks(i))));
end

figure();
for i = 1:25
    k = 25;
    eig_vec = U(:, 1:k);
    eig_vec_1 = reshape((eig_vec(:, 26-i)), [112, 92]);
    subplot(5, 5, i)
    imshow(eig_vec_1, [], 'InitialMagnification', 'fit');
    title(strcat('eigenvector = ', int2str(26-i)));
end
end








