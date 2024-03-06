clc ;clear; close all

%% load to data

datasetpath = 'D:\yachae_sw\CTImages\localization_100_detection\';
imageDir = fullfile(datasetpath);
load("label.mat")
load("AP.mat")
imds = imageDatastore(imageDir);
Classes = [0 1];

% each dicom count
for i = 1 : size(AP,2)
    subject_datanum(i) = size(AP(i).Label,1);
end

%% design optimization variable

num_folds = 5;
datanum = 100;

l3rate = 1/(length(label) / sum(label));
classweightmin = 1/((length(label)- sum(label))/sum(label));

for fold_idx = 1  % : num_folds

  optimVars = [
                optimizableVariable('InitialLearnRate',[0.00001 0.001],'Type','real')
                optimizableVariable('miniBatchSize',[4 64],'Type','integer')
                optimizableVariable('augmentation', [l3rate 0.5], 'Type', 'real')
                optimizableVariable('classweightfactor', [classweightmin 1], 'Type', 'real')];

  ObjFcn = @(optimVars) VGG16CVOpti(optimVars, imds, label, fold_idx,subject_datanum, l3rate);


  BayesObjectR = bayesopt(ObjFcn,optimVars,...
      'AcquisitionFunctionName','expected-improvement-plus','PlotFcn',[],...
      'IsObjectiveDeterministic',false,...
      'MaxObjectiveEvaluations',30,...
      'UseParallel',false);

  bayesianop(fold_idx).bestop{1,1} = bestPoint(BayesObjectR);

  save('bestParameterR.mat', 'bayesianop','BayesObjectR');
end

%%
for fold_idx = 1 % : num_folds

[imdsTrain, imdsTest, trainingLabels, testLabels,testIdx] = partitionlocalizationData(imds,label,fold_idx,subject_datanum);
Classes = [0 1];

% data augmentation
cnt1 = 1;
for i = 1 : size(trainingLabels,2)
    if trainingLabels(i) == 1
        L3trindex(cnt1) = i;
        cnt1 = cnt1 + 1;
    end
end

cnt1 = 1;
clear temp1
if bayesianop(fold_idx).bestop{1,1}.augmentation > l3rate
    augmentnum = round((bayesianop(fold_idx).bestop{1,1}.augmentation * size(trainingLabels,2) - sum(trainingLabels)));
    temp = readall(imdsTrain);
    if augnumQ == 0
    else
        for augnum = 1 : augnumQ


            angle = [-45 45];
            xTrans = [-10 10];
            yTrans = [-10 10];


            tform = randomAffine2d(...
                'Rotation',angle,...
                'XReflection',true,...
                'YReflection',true,...
                'XTranslation', xTrans, ...
                'YTranslation', yTrans);


            if trainingLabels(L3trindex(augnum)) == 1
                rout = affineOutputView(size(temp{L3trindex(augnum)}), tform, 'BoundsStyle', 'centerOutput');
                temp1{cnt1} = imwarp(temp{L3trindex(augnum)}, tform, 'OutputView', rout);
                trainingLabels1(cnt1) = 1;
                cnt1 = cnt1 + 1;
            end
        end
    end
    augindex = randperm(sum(trainingLabels),augnumR);
    for augnum = augindex
        angle = [-45 45];
        xTrans = [-10 10];
        yTrans = [-10 10];


        tform = randomAffine2d(...
            'Rotation',angle,...
            'XReflection',true,...
            'YReflection',true,...
            'XTranslation', xTrans, ...
            'YTranslation', yTrans);
        rout = affineOutputView(size(temp{L3trindex(augnum)}), tform, 'BoundsStyle', 'centerOutput');
        temp1{cnt1} = imwarp(temp{L3trindex(augnum)}, tform, 'OutputView', rout);
        trainingLabels1(cnt1) = 1;
        cnt1 = cnt1 + 1;
    end
else

end

TrX = [temp ;temp1.'];
TrY = [trainingLabels,trainingLabels1];

clear temp temp1

for i = 1 : size(TrX)
    for j = 1 : 3
        TrX1(:,:,j,i) = imresize(TrX{i},[224 224]);
    end
end

TstX = readall(imdsTest);

for i = 1 : size(TstX)
    for j = 1 : 3
        TstX1(:,:,j,i) = imresize(TstX{i},[224 224]);
    end
end

    options = trainingOptions('adam', ...
        'InitialLearnRate', bayesianop(fold_idx).bestop{1,1}.InitialLearnRate, ...
        'MaxEpochs', 20, ...
        'ValidationData',{TstX1,categorical(testLabels)}, ...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,5), ...
        'MiniBatchSize', bayesianop(fold_idx).bestop{1,1}.miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 1, ...
        'ExecutionEnvironment','gpu');


net = vgg16("Weights","imagenet");
layersTransfer = net.Layers(1:end-3);
numClasses = numel(Classes);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer('Classes',categorical([0 1]) ,'ClassWeights',[bayesianop(fold_idx).bestop{1,1}.classweightfactor;1])];


nettrained{fold_idx}  = trainNetwork(TrX1,categorical(TrY.'),layers,options);
preLabel{fold_idx} =classify(nettrained{fold_idx},TstX1);

save('VGG16_trained_2.mat', 'nettrained');

CWeighted{fold_idx} = confusionchart(categorical(testLabels),preLabel{fold_idx}, Title="With Class Weighting",RowSummary="row-normalized");

for i = 1:numClasses
    PrecisionWeighted(i) = CWeighted{fold_idx}.NormalizedValues(i,i) / sum(CWeighted{fold_idx}.NormalizedValues(i,:));
    RecallWeighted(i) = CWeighted{fold_idx}.NormalizedValues(i,i) / sum(CWeighted{fold_idx}.NormalizedValues(:,i));
    f1Weighted(i) = max(0,(2*PrecisionWeighted(i)*RecallWeighted(i)) / (PrecisionWeighted(i)+RecallWeighted(i)));
end

end







