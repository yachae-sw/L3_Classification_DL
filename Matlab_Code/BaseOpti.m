function [Final_Objective] = BaseOpti(optimVars, imds, label, fold_idx, subject_datanum)
miniBatchSize = optimVars.miniBatchSize;
MaxEpochs = 10;

% load to data
[imdsTrain, imdsTest, trainingLabels, testLabels,testIdx] = partitionlocalizationData(imds,label,fold_idx,subject_datanum);
Classes = [0 1];

% find L3 location
cnt = 1;
for i = 1 : size(trainingLabels,2)
    if trainingLabels(i) == 1
        L3trindex(cnt) = i;
        cnt = cnt + 1;
    end
end

% reset data
clear temp TrX TrY TrX1 sf_TrX sf_TrY TstX TstX1

% all data augmentation
temp = readall(imdsTrain);

cnt2 = 1;
for i = 1 : size(trainingLabels,2)
    angle = [-10 10];
    Trans = [-10 10];
    Shears = [-10 10];

    tform = randomAffine2d(...
        'Rotation',angle,...
        'XReflection',true,...
        'YReflection',true,...
        'XTranslation', Trans, ...
        'YTranslation', Trans, ...
        'XShear', Shears, ...
        'YShear',Shears);
    rout = affineOutputView(size(temp{i}), tform, 'BoundsStyle', 'centerOutput');
    temp_c{cnt2} = imwarp(temp{i}, tform, 'OutputView', rout);
    trainingLabels_c(cnt2) = trainingLabels(i);
    cnt2 = cnt2 + 1;
end

TrX = [temp ;temp_c.'];
TrY = [trainingLabels,trainingLabels_c];

% shuffle train dataset
num_samples = size(TrX, 1);
shuffle_indices = randperm(num_samples);
cnt5 = 1;
for j = shuffle_indices
    sf_TrX{cnt5,:} = TrX{j, :};
    cnt5 = cnt5 + 1;
end
sf_TrY = TrY(shuffle_indices);

clear num_samples shuffle_indices

% train data 4D
for i = 1 : size(sf_TrX)
    for j = 1 : 3
        TrX1(:,:,j,i) = imresize(sf_TrX{i},[224 224]);
    end
end

% test data
TstX = readall(imdsTest);

for i = 1 : size(TstX)
    for j = 1 : 3
        TstX1(:,:,j,i) = imresize(TstX{i},[224 224]);
    end
end

% model option setting
options = trainingOptions(char(optimVars.optimizer), ...
    'InitialLearnRate', optimVars.InitialLearnRate, ...
    'MaxEpochs', MaxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'ExecutionEnvironment','gpu');

    % 'ValidationData',{TstX1,categorical(testLabels)}, ...
    % 'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10), ...

% model vgg16 setting
% net = vgg16("Weights","imagenet");
% layersTransfer = net.Layers(1:end-3);
% numClasses = numel(Classes);
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer('Classes',categorical([0 1]) ,'ClassWeights',[optimVars.classweightfactor;1])];

% model layer ResNet50
net = resnet50;
numClasses = numel(Classes);
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name', 'new_conv');
lgraph = replaceLayer(lgraph, 'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput','Classes',categorical([0 1]));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000',newClassLayer);

% vgg16 train model and result
% nettrained  = trainNetwork(TrX1,categorical(sf_TrY.'),layers,options);

% ResNet50 train model and result
nettrained  = trainNetwork(TrX1,categorical(sf_TrY.'),lgraph,options);

preLabel =classify(nettrained,TstX1);
percentLabel = predict(nettrained,TstX1);

CWeighted = confusionchart(categorical(testLabels),preLabel, Title="With Class Weighting",RowSummary="row-normalized");

PrecisionWeighted = CWeighted.NormalizedValues(2,2) / sum(CWeighted.NormalizedValues(2,:));
RecallWeighted = CWeighted.NormalizedValues(2,2) / sum(CWeighted.NormalizedValues(:,2));
f1Weighted = max(0.00001,(2*PrecisionWeighted*RecallWeighted) / (PrecisionWeighted+RecallWeighted));

testsection = subject_datanum(testIdx);

cnt6 = 1;
for i = 1 : size(testsection,2)
    for j = 1 : testsection(i)
        subjectLabel{i}(j) = percentLabel(cnt6,2); % each pred label percent
        eachLabel{i}(j) = preLabel(cnt6); % each pred label
        eachtestLabel{i}(j) = testLabels(cnt6); % each test label
        cnt6 = cnt6 + 1;
    end
end

% each f1 score
for i = 1 : size(testIdx,2)
    eachCWeighted = confusionchart(categorical(eachtestLabel{:,i}),eachLabel{:,i}, Title="With Class Weighting",RowSummary="row-normalized",ColumnSummary="column-normalized");
    
    PrecisionWeighted = eachCWeighted.NormalizedValues(2,2) / sum(eachCWeighted.NormalizedValues(2,:));
    RecallWeighted = eachCWeighted.NormalizedValues(2,2) / sum(eachCWeighted.NormalizedValues(:,2));
    eachf1Weighted{i} = max(0,(2*PrecisionWeighted*RecallWeighted) / (PrecisionWeighted+RecallWeighted));
end

eachf1mean = mean(cell2mat(eachf1Weighted));

fprintf('f1 score = %d, each f1 score = %d \n',f1Weighted, eachf1mean)

Final_Objective = 1-f1Weighted;

close all

end

