function [imdsTrain, imdsTest, trainingLabels, testLabels,testIdx] = partitionlocalizationData(imds,label,num,subject_datanum)

% reset
clear imdsTrain
clear imdsTest
clear trainingLabels
clear testLabels
clear testIdx
clear testIndex
clear trainingIdx
clear testIndex1
numFiles = size(subject_datanum,2);

rng(0);

% Use 20% of the images for testing.
testIdx = num:5:numFiles;
testIdx1 = zeros(numFiles,1);
testIdx1(testIdx) = 1;

testIndex =  zeros(sum(subject_datanum),1);
cnt = 1;
cnt1 = 1;
for i = 1 : numFiles
    if testIdx1(i)
        for j = 1 : subject_datanum(i)
            testIndex(cnt) = 1;
            testIndex1(cnt1) = cnt;
            cnt = cnt + 1;
            cnt1 = cnt1 +1;
        end
    else
        for j = 1 : subject_datanum(i)
            testIndex(cnt) = 0;
            cnt = cnt + 1;
        end
    end
end

% Use the rest for training.
trainingIdx = setdiff(1:length(imds.Files),testIndex1);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIndex1);

imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);

% Create pixel label datastores for training and test.
trainingLabels = label(trainingIdx);
testLabels = label(testIndex1);

imdsTrain.Labels = categorical(trainingLabels);
imdsTest.Labels = categorical(testLabels);

end