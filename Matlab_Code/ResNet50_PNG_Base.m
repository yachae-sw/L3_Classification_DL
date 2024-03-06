clc ;clear; close all
%% load to data

Base = 'D:\yachae_sw\CTImages\';
imageDir = fullfile(Base,'localization_100_detection','*.*');
load("D:\yachae_sw\code\classification\data\label3.mat")
load("D:\yachae_sw\code\classification\data\AP3.mat")
imds = imageDatastore(imageDir);
Classes = [0 1];

% load to excel data
[AP_Data_Label] = readcell(fullfile(Base, 'CT_Mask_nii_100/CT_100_label_20230807.xlsx'));

% each dicom count
for i = 1 : size(AP,2)
    subject_datanum(i) = size(AP(i).Label,1);
end

%% base design optimization variable(5-fold cross validation)

num_folds = 5;
datanum = 100;
alll3rate = 1/(length(label) / sum(label));
classweightsqmin = (1/sqrt(length(label)- sum(label)))/(1/sqrt(sum(label)));
classweightmin = (1/(length(label)- sum(label)))/(1/(sum(label)));

for fold_idx = 1 : num_folds

    fprintf('Processing %d among %d folds \n', fold_idx,5); % 5-fold cross validation

    optimVars = [
        optimizableVariable('optimizer',{'sgdm','adam'},'Type','categorical')
        optimizableVariable('InitialLearnRate',[0.00001 0.01],'Type','real')
        optimizableVariable('miniBatchSize',[8 128],'Type','integer')];

    ObjFcn = @(optimVars) BaseOpti(optimVars, imds, label, fold_idx,subject_datanum);

    BayesObjectR = bayesopt(ObjFcn,optimVars, ...
        'AcquisitionFunctionName','expected-improvement-plus','PlotFcn',[],...
        'IsObjectiveDeterministic',false,...
        'MaxObjectiveEvaluations',30,...
        'UseParallel',false);
    
    bayesianbaseop{fold_idx,:} = bestPoint(BayesObjectR);

end

save('bestParameterR_base.mat', 'bayesianbaseop');

%% base Model Training with Bayesian Optimization

cv_base_result = struct('nettrained', cell(1, num_folds), 'testIdx', cell(1, num_folds), 'testLabels', cell(1, num_folds), 'preLabel', cell(1, num_folds), 'percentLabel', cell(1, num_folds), 'f1Weighted', cell(1, num_folds));

for fold_idx = 1 : num_folds

    [imdsTrain, imdsTest, trainingLabels, testLabels,testIdx] = partitionlocalizationData(imds,label,fold_idx,subject_datanum);
    Classes = [0 1];
    
    cv_base_result(fold_idx).testIdx = testIdx;

    % find L3 location
    cnt = 1;
    for i = 1 : size(trainingLabels,2)
        if trainingLabels(i) == 1
            L3trindex(cnt) = i;
            cnt = cnt + 1;
        end
    end
    
    L3rate = 1/(length(trainingLabels) / sum(trainingLabels));

    % data augmentation
    clear temp TrX TrY TrX1 sf_TrX sf_TrY TstX TstX1
   
    TrX = readall(imdsTrain);
    sf_TrY = trainingLabels;
    
    % train data 4D
    for i = 1 : size(TrX)
        for j = 1 : 3
            TrX1(:,:,j,i) = imresize(TrX{i},[224 224]);
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
    options = trainingOptions(char(bayesianbaseop{1}.optimizer), ...
        'InitialLearnRate', bayesianbaseop{1}.InitialLearnRate, ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', bayesianbaseop{1}.miniBatchSize, ...
        'Plots','training-progress',...
        'Shuffle', 'every-epoch', ...
        'Verbose', 1);
    
        % 'ValidationData',{TstX1,categorical(testLabels)}, ...
        % 'OutputFcn',@(info)stopIfAccuracyNotImproving(info,5), ...
    
    % model vgg 16
    % net = vgg16("Weights","imagenet");
    % layersTransfer = net.Layers(1:end-3);
    % numClasses = numel(Classes);
    % layers = [
    %     layersTransfer
    %     fullyConnectedLayer(numClasses)
    %     softmaxLayer
    %     classificationLayer('Classes',categorical([0 1]) ,'ClassWeights',[bayesianop(fold_idx).bestop{1,1}.classweightfactor;1])];
    
    % model layer ResNet50
    net = resnet50;
    numClasses = numel(Classes);
    lgraph = layerGraph(net);
    newFCLayer = fullyConnectedLayer(numClasses,'Name', 'new_conv');
    lgraph = replaceLayer(lgraph, 'fc1000',newFCLayer);
    newClassLayer = classificationLayer('Name','new_classoutput','Classes',categorical([0 1]));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000',newClassLayer);
    
    % vgg16 train model and result
    % nettrained{fold_idx}  = trainNetwork(TrX1,categorical(sf_TrY.'),layers,options);

    % ResNet50 train model and result
   [nettrained2 , info2] = trainNetwork(TrX1,categorical(sf_TrY.'),lgraph,options);

    cv_base_result(fold_idx).testLabels = categorical(testLabels);
    cv_base_result(fold_idx).preLabel =classify(cv_base_result(fold_idx).nettrained,TstX1);
    cv_base_result(fold_idx).percentLabel = predict(cv_base_result(fold_idx).nettrained,TstX1);

    CWeighted{fold_idx} = confusionchart(categorical(testLabels),cv_base_result(fold_idx).preLabel, Title="With Class Weighting",RowSummary="row-normalized",ColumnSummary="column-normalized");
    
    PrecisionWeighted = CWeighted{fold_idx}.NormalizedValues(2,2) / sum(CWeighted{fold_idx}.NormalizedValues(2,:));
    RecallWeighted = CWeighted{fold_idx}.NormalizedValues(2,2) / sum(CWeighted{fold_idx}.NormalizedValues(:,2));
    cv_base_result(fold_idx).f1Weighted = max(0,(2*PrecisionWeighted*RecallWeighted) / (PrecisionWeighted+RecallWeighted));
    
    % test section CT count
    testsection = subject_datanum(cv_base_result(fold_idx).testIdx);

    % load to value
    cnt6 = 1;
    for i = 1 : size(testsection,2)
        for j = 1 : testsection(i)
            % subjectLabel{k}{i}(j) = percentLabel{fold_idx}(cnt,2);
            cv_base_result(fold_idx).eachpredLabel{i}(j) = cv_base_result(fold_idx).preLabel(cnt6);
            cv_base_result(fold_idx).eachtestLabel{i}(j) = cv_base_result(fold_idx).testLabels(cnt6);
            cnt6 = cnt6 + 1;
        end
    end

    % each f1 score by person
    for m = 1 : size(cv_base_result(fold_idx).testIdx,2)
        eachCWeighted = confusionchart(categorical(cv_base_result(fold_idx).eachtestLabel{m}),cv_base_result(fold_idx).eachpredLabel{m}, Title="With Class Weighting",RowSummary="row-normalized",ColumnSummary="column-normalized");
        
        PrecisionWeighted = eachCWeighted.NormalizedValues(2,2) / sum(eachCWeighted.NormalizedValues(2,:));
        RecallWeighted = eachCWeighted.NormalizedValues(2,2) / sum(eachCWeighted.NormalizedValues(:,2));
        cv_base_result(fold_idx).eachf1score{m} = max(0,(2*PrecisionWeighted*RecallWeighted) / (PrecisionWeighted+RecallWeighted));
    end

    % mean f1 score
    cv_base_result(fold_idx).meanf1score = mean(cell2mat(cv_base_result(fold_idx).eachf1score));

end

% save to result
save('D:\yachae_sw\code\classification\data\ResNet50_base_Result.mat', 'cv_base_result');
