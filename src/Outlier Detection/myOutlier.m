function[] = myOutlier(path, h, w, num_ids, num_train, num_test, k)
% path = '../../../att_faces/s'; h= 112; w = 92; num_ids = 32; num_train =  6; num_test = 4; k = 0;
d = h*w;
trainSet = zeros(h, w, num_train, num_ids);
testSet = zeros(h, w, num_test, num_ids);
outliers = zeros(h, w, num_test, (40-num_ids));
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

for i = 1:(40-num_ids)
    newpath = strcat(path, int2str(num_ids+i));
    S = dir(fullfile(newpath, '*.pgm'));
    for j = (num_train + 1):size(S, 1)
        outliers(:, :, j-num_train, i) = double(imread(strcat(newpath, '/', S(j).name)));
    end
end

meanArr(:, :) = reshape(double(sum(sum((trainSet(:, :, :, :)), 4), 3)./(num_train*num_ids)), [d, 1]);
temp = reshape(trainSet(:, :, :, :), [d, num_train*num_ids]);
A = temp - meanArr;
[U, ~, ~] = svd(A);
normU = vecnorm(U, 2, 1);
U = U./normU;

ks = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 175]';


if k == 0
    myTestScore = zeros(size(ks, 1), 1);
    myOutScore1 = zeros(size(ks, 1), 1);
    myOutScore2 = zeros(size(ks, 1), 1);
    for m = 1:size(ks, 1)
        eig_vec = U(:, 1:ks(m));
        alphas = eig_vec' * A;

        args = zeros(num_test*num_ids, 1);
        vals = zeros(num_test*num_ids, 1);

        testTemp = reshape(testSet(:, :, :, :), [d, num_test*num_ids]);
        testA = testTemp - meanArr;
        alpha_ps = eig_vec' * testA;

        for i = 1:(num_ids*num_test)
            [argval, argmin] = min(vecnorm((alpha_ps(:, i) - alphas), 2, 1).^2);
            args(i) = argmin;
            vals(i) = argval;
        end

        matched = [];

        id = 1;
        false = 0;
        true = 0;
        for i = 1:num_ids
            for j = 1:num_test
                if ceil(args(id)/num_train) == i
                    matched = [matched vals(id)]; %#ok<AGROW>
                    true = true+1;
                    id  = id+1;
                else
                    false = false + 1;
                    id = id+1;
                end
            end
        end

        matched = sort(matched);
        idx = ceil(0.75*size(matched, 2));
        th = matched(idx);
        myTestScore(m) = true/(true+false);

        valsO = zeros(num_test*num_ids, 1);

        outTemp = reshape(outliers(:, :, :, :), [d, num_test*(40-num_ids)]);
        outA = outTemp - meanArr;
        alpha_out = eig_vec' * outA;

        for i = 1:((40-num_ids)*num_test)
            [argval, ~] = min(vecnorm((alpha_out(:, i) - alphas), 2, 1).^2);
            valsO(i) = argval;
        end

        id = 1;
        falsep = 0;
        falsen = 0;
        for i = 1:(40-num_ids)*num_test
            if valsO(i) > th
                falsep = falsep+1;
            end
        end

        vals1 = zeros(10*num_ids, 1);

        for i =1:num_ids*10
            if i < 193
                [argval, ~] = min(vecnorm((alphas(:, i) - alphas), 2, 1).^2);
                vals1(i) = argval;
            else
                [argval, ~] = min(vecnorm((alpha_ps(:, i-192) - alphas), 2, 1).^2);
                vals1(i) = argval;
            end
        end

        for i = 1:num_ids*10
            if vals1(i) > th
                falsen = falsen + 1;
            end
        end


        myOutScore1(m) = (falsep)/32; %% should be more
        myOutScore2(m) = falsen/320; %% should be less
    end
    figure();
    hold on
    plot(ks, myTestScore);
    plot(ks, myOutScore1);
    plot(ks, myOutScore2);
    legend('Test Recognition Rate (w/o thresholding)', 'False Positive Recognition Rate (with thresholding)', 'False Negative Recognition Rate (with thresholding)', 'Location', 'east');
    hold off
else
    eig_vec = U(:, 1:k);
    alphas = eig_vec' * A;

    args = zeros(num_test*num_ids, 1);
    vals = zeros(num_test*num_ids, 1);

    testTemp = reshape(testSet(:, :, :, :), [d, num_test*num_ids]);
    testA = testTemp - meanArr;
    alpha_ps = eig_vec' * testA;

    for i = 1:(num_ids*num_test)
        [argval, argmin] = min(vecnorm((alpha_ps(:, i) - alphas), 2, 1).^2);
        args(i) = argmin;
        vals(i) = argval;
    end

    matched = [];

    id = 1;
    false = 0;
    true = 0;
    for i = 1:num_ids
        for j = 1:num_test
            if ceil(args(id)/num_train) == i
                matched = [matched vals(id)]; %#ok<AGROW>
                true = true+1;
                id  = id+1;
            else
                false = false + 1;
                id = id+1;
            end
        end
    end

    matched = sort(matched);
    idx = ceil(0.75*size(matched, 2));
    th = matched(idx);
    disp(strcat('Test Recognition Rate (w/o thresholding) for k = ', int2str(k)))
    disp(true/(true+false));

    valsO = zeros(num_test*(40-num_ids), 1);

    outTemp = reshape(outliers(:, :, :, :), [d, num_test*(40-num_ids)]);
    outA = outTemp - meanArr;
    alpha_out = eig_vec' * outA;

    for i = 1:((40-num_ids)*num_test)
        [argval, ~] = min(vecnorm((alpha_out(:, i) - alphas), 2, 1).^2);
        valsO(i) = argval;
    end

    falsep = 0;
    for i = 1:32
        if valsO(i) > th
            falsep = falsep + 1;
        end
    end

    vals1 = zeros(10*num_ids, 1);

    for i =1:num_ids*10
        if i < 193
            [argval, ~] = min(vecnorm((alphas(:, i) - alphas), 2, 1).^2);
            vals1(i) = argval;
        else
            [argval, ~] = min(vecnorm((alpha_ps(:, i-192) - alphas), 2, 1).^2);
            vals1(i) = argval;
        end
    end

    falsen = 0;
    for i = 1:num_ids*10
        if vals1(i) > th
            falsen = falsen + 1;
        end
    end

    disp('Overall Stats (with threshold):')
    disp(strcat('False Positive Recognition Rate for k = ', int2str(k)))
    disp((32-falsep)/32);
    disp(strcat('False Negative Recognition Rate for k = ', int2str(k)))
    disp(falsen/320);

end

end
