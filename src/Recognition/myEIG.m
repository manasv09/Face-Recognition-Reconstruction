function[] = myEIG(path, h, w, num_ids, num_train, num_test, k)
% path = '../../../att_faces/s'; h= 112; w = 92; num_ids = 32; num_train =  6; num_test = 4; k = 192;
d = h*w;
trainSet = zeros(w, h, num_train, num_ids);
testSet = zeros(w, h, num_test, num_ids);
meanArr = zeros(d, 1);

for i = 1:num_ids
    newpath = strcat(path, int2str(i));
    S = dir(fullfile(newpath, '*.pgm'));
    for j = 1:num_train
        trainSet(:, :, j, i) = double(imread(strcat(newpath, '/', S(j).name))');
    end
    for j = (num_train + 1):size(S, 1)
        testSet(:, :, j-num_train, i) = double(imread(strcat(newpath, '/', S(j).name))');
    end
end

meanArr(:, :) = reshape(double(sum(sum((trainSet(:, :, :, :)), 4), 3)./(num_train*num_ids)), [d, 1]);
temp = reshape(trainSet(:, :, :, :), [d, num_train*num_ids]);
A = temp - meanArr;
L = A' * A;
[W, D] = eig(L);
[~, idx] = sort(D);
W_1 = zeros(192);
for i = 1:192
    W_1(:, i) = W(:, idx(193-i));
end
V = A * W_1;
normV = vecnorm(V, 2, 1);
V = V./normV;
ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 175, 192]';


if k == 0
    myScore = zeros(size(ks, 1), 1);
    for m = 1:size(ks, 1)
        eig_vec = V(:, 1:ks(m));
        alphas = eig_vec' * A;

        args = zeros(num_test*num_ids, 1);

        testTemp = reshape(testSet(:, :, :, :), [d, num_test*num_ids]);
        testA = testTemp - meanArr;
        alpha_ps = eig_vec' * testA;

        for i = 1:(num_ids*num_test)
            [~, argmin] = min(vecnorm((alpha_ps(:, i) - alphas), 2, 1).^2);
            args(i) = argmin;
        end

        id = 1;
        false = 0;
        true = 0;
        for i = 1:num_ids
            for j = 1:num_test
                if ceil(args(id)/num_train) == i
                    true = true+1;
                    id  = id+1;
                else
                    false = false + 1;
                    id = id+1;
                end
            end
        end

        myScore(m) = true/(true+false);
    end
    figure();
    plot(ks, myScore);
    title('Face Recognition Rates of ORL using eig function');
else
    eig_vec = V(:, 1:k);
    alphas = eig_vec' * A;

    args = zeros(num_test*num_ids, 1);

    testTemp = reshape(testSet(:, :, :, :), [d, num_test*num_ids]);
    testA = testTemp - meanArr;
    alpha_ps = eig_vec' * testA;

    for i = 1:(num_ids*num_test)
        [~, argmin] = min(vecnorm((alpha_ps(:, i) - alphas), 2, 1).^2);
        args(i) = argmin;
    end

    id = 1;
    false = 0;
    true = 0;
    for i = 1:num_ids
        for j = 1:num_test
            if ceil(args(id)/num_train) == i
                true = true+1;
                id  = id+1;
            else
                false = false + 1;
                id = id+1;
            end
        end
    end
    disp(strcat('Face Recognition Rate of ORL using eig for k = ', int2str(k)));
   disp(true/(true+false)); 
end
end
