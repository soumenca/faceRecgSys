clear;

imgSets = imageSet('/home/soumen/Desktop/PhDCourseWork/miniproject/DS/LFWCropped20/', 'recursive');
feature = [];
for k = 1:length(imgSets)
    for l = 1:imgSets(k).Count
        Input_image = read(imgSets(k), l);
        if size(Input_image, 3) == 3
            Input_image = rgb2gray(Input_image);
        end
        [m, n] = size(Input_image);
        F = [];
        m1 = floor(m/8) * 8;
        n1 = floor(n/8) * 8;
        for i = 1 : floor(m/8) : m1
            for j = 1 : floor(n/8) : n1
                lbp_feature = extractLBPFeatures(Input_image(i:floor(i+m/8-1), j:floor(j+n/8-1)));
                F = horzcat(F, lbp_feature);
            end
        end
        F = horzcat(F, k);
        feature = [feature; F];
    end
end
%csvwrite('faceFeature_lfw.csv',feature);

for x = 10:10:300
    fprintf(".......Output.......\n");
    reducedData = funPCA(feature, x);
    funKNN(reducedData);
    funNB(reducedData);
    %funSVM(reducedData); 
end

%% PCA Implementation .............

function reducedData = funPCA(feature, noFeature)
    [~, fc] = size(feature);
    feature1 = feature(:,1:(fc-1));
    %noFeature = 80;

    coeff= pca(feature1);
    reducedDimension = coeff(:, 1:noFeature);
    reducedData = feature1 * reducedDimension;
    reducedData = [reducedData, feature(:,fc)];
    fprintf("The feature of the original data is: %d\n",  fc-1);
    fprintf("The feature of the reduced data is: %d\n",  noFeature);
end


%% Classification .............KNN.......

function [] = funKNN(reducedData)
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .80, 0, .20);
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);
    valData = reducedData(valIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    valClass = reducedData(valIndex, noFeature);

    Mdl = fitcknn(trainData, trainClass,'NumNeighbors',1);
    predictClass = predict(Mdl, testData);

    results = testClass == predictClass;
    true = sum(results == 1);
    accuracy = (true / length(testClass)) * 100;
    fprintf("The Accuracy of prediction using KNN is %0.4f\n", accuracy);
    nb = fitcknn(reducedData(:,1:noFeature-1), reducedData(:,noFeature), 'NumNeighbors',1);
    nb = crossval(nb);
    fprintf("The loss obtained by cross-validated classification using KNN is %0.4f\n",  kfoldLoss(nb));

end



%% Classification ................SVM........

function [] = funSVM(reducedData)
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .80, 0, .20);
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);
    valData = reducedData(valIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    valClass = reducedData(valIndex, noFeature);

    results = single(multisvm(trainData, trainClass, testData));
    
    true = sum(results == 1);
    accuracy = (true / length(testClass)) * 100;
    fprintf("The Accuracy of prediction using SVM is %0.4f\n", accuracy);

end

%% Classification .............NaiveBayes......

function [] = funNB(reducedData)
    [fr, noFeature] = size(reducedData);
    [trainIndex,valIndex,testIndex] = dividerand(fr, .80, 0, .20);
    trainData = reducedData(trainIndex, 1:noFeature-1);
    testData = reducedData(testIndex, 1:noFeature-1);
    valData = reducedData(valIndex, 1:noFeature-1);

    trainClass = reducedData(trainIndex, noFeature);
    testClass = reducedData(testIndex, noFeature);
    valClass = reducedData(valIndex, noFeature);

    Mdl = fitcnb(trainData, trainClass, 'DistributionNames', 'normal');
    predictClass = predict(Mdl, testData);
    results = testClass == predictClass;
    
    true = sum(results == 1);
    accuracy = (true / length(testClass)) * 100;
    fprintf("The Accuracy of prediction using NB is %0.4f\n", accuracy);
    nb = fitcnb(reducedData(:,1:noFeature-1), reducedData(:,noFeature));
    nb = crossval(nb);
    fprintf("The loss obtained by cross-validated classification using NB is %0.4f\n",  kfoldLoss(nb));

end