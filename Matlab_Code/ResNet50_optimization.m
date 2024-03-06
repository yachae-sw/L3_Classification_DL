clc ;clear; close all
%% load to data

Base = 'D:\yachae_sw\CTImages\';
datasetpath = 'D:\yachae_sw\CTImages\classification_150_detection\';
imageDir = fullfile(datasetpath);
load("D:\yachae_sw\code\classification\data\label7.mat")
load("D:\yachae_sw\code\classification\data\AP7.mat")
imds = imageDatastore(imageDir);
Classes = [0 1];

% load to excel data
[AP_Data_Label] = readcell(fullfile(Base, 'CT_Mask_nii_150/CT_L3_label_20230910.xlsx'));

% each dicom count
for i = 1 : size(AP,2)
    subject_datanum(i) = size(AP(i).Label,1);
end

%% design optimization variable(5-fold cross validation)

num_folds = 5;
datanum = 100;

for fold_idx =  1 : num_folds

    fprintf('Processing %d among %d folds \n', fold_idx,5); % 5-fold cross validation

    [imdsTrain, imdsTest, trainingLabels, testLabels,testIdx] = partitionlocalizationData(imds,label,fold_idx,subject_datanum);

    alll3rate = 1/(length(trainingLabels) / sum(trainingLabels));
    classweightsqmin = (1/sqrt(length(trainingLabels)- sum(trainingLabels)))/(1/sqrt(sum(trainingLabels)));
    classweightmin = (1/(length(trainingLabels)- sum(trainingLabels)))/(1/(sum(trainingLabels)));

    optimVars2 = [
        optimizableVariable('L2Regularization',[0.00001 0.01],'Type','real')
        optimizableVariable('InitialLearnRate',[0.00001 0.01],'Type','real')
        optimizableVariable('miniBatchSize',[16 64],'Type','integer')
        optimizableVariable('augmentation', [alll3rate 0.25], 'Type', 'real')
        optimizableVariable('classweightfactor', [classweightsqmin 1], 'Type', 'real')
        optimizableVariable('GradientThreshold',[1 6],'Type','integer')
        optimizableVariable('GradientThresholdMethod',{'global-l2norm','l2norm'},'Type','categorical')];

    ObjFcn2 = @(optimVars2) ResNefact50CVOpti3(optimVars2, imds, label, fold_idx,subject_datanum);

    BayesObjectR = bayesopt(ObjFcn2,optimVars2, ...
        'AcquisitionFunctionName','expected-improvement-plus','PlotFcn',[],...
        'IsObjectiveDeterministic',false,...
        'MaxObjectiveEvaluations',50,...
        'UseParallel',false);
    
    bayesianop{fold_idx,1} = bestPoint(BayesObjectR);
    bayesianop{fold_idx,2} = BayesObjectR;
end

save('bestParameterR150_result.mat', 'bayesianop');

%% Model Training with Bayesian Optimization

cv_result = struct('nettrained', cell(1, num_folds), 'testIdx', cell(1, num_folds), 'testLabels', cell(1, num_folds), 'preLabel', cell(1, num_folds), 'percentLabel', cell(1, num_folds), 'f1Weighted', cell(1, num_folds));

for fold_idx = 4% 1 : num_folds

    [imdsTrain, imdsTest, trainingLabels, testLabels,testIdx] = partitionlocalizationData(imds,label,fold_idx,subject_datanum);
    Classes = [0 1];
    
    cv_result(fold_idx).testIdx = testIdx;

    clear L3trindex
    % find L3 location
    cnt = 1;
    for i = 1 : size(trainingLabels,2)
        if trainingLabels(i) == 1
            L3trindex(cnt) = i;
            cnt = cnt + 1;
        end
    end
    
    % reset data
    clear temp temp_c TrX TrY TrX1 sf_TrX sf_TrY TstX TstX1
    clear trainingLabels_c tempL3_all tempL3_all2 trainingLabelsL3_all trainingLabelsL3_all2
    clear temp_re temp_re2 trainingLabels_re trainingLabels_re2
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

    % data augmentation

    augmentnum = round((bayesianop{fold_idx}.augmentation * size(trainingLabels,2)) - sum(trainingLabels));
    augnumR = rem(augmentnum,sum(trainingLabels)); % calculate Remainder
    augnumQ= fix(augmentnum/sum(trainingLabels)); % calculate Quotient
        
    if augnumQ ~= 0
        % augnumQ running 2 times
        [tempL3_all, trainingLabelsL3_all] = augquotient(augnumQ,L3trindex,temp,trainingLabels);
        [tempL3_all2, trainingLabelsL3_all2] = augquotient(augnumQ,L3trindex,temp,trainingLabels);
    
        % augnumR running 2 times
        if augnumR ~=0
            [temp_re, trainingLabels_re] = augremainder(augnumR,L3trindex,temp,trainingLabels);
            [temp_re2, trainingLabels_re2] = augremainder(augnumR,L3trindex,temp,trainingLabels);
    
            TrX = [temp ;tempL3_all.';tempL3_all2.';temp_re.';temp_re2.';temp_c.'];
            TrY = [trainingLabels,trainingLabelsL3_all,trainingLabelsL3_all2,trainingLabels_re,trainingLabels_re2,trainingLabels_c];
        else
            TrX = [temp ;tempL3_all.';tempL3_all2.';temp_c.'];
            TrY = [trainingLabels,trainingLabelsL3_all,trainingLabelsL3_all2,trainingLabels_c];
        end
    else
        % augnumR running 2 times
        if augnumR ~=0
            [temp_re, trainingLabels_re] = augremainder(augnumR,L3trindex,temp,trainingLabels);
            [temp_re2, trainingLabels_re2] = augremainder(augnumR,L3trindex,temp,trainingLabels);
    
            TrX = [temp ;temp_re.';temp_re2.';temp_c.'];
            TrY = [trainingLabels,trainingLabels_re,trainingLabels_re2,trainingLabels_c];
        else
            TrX = [temp ;temp_c.'];
            TrY = [trainingLabels,trainingLabels_c];
        end
    end
    
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
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', bayesianop{fold_idx}.InitialLearnRate, ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', bayesianop{fold_idx}.miniBatchSize, ...
        'L2Regularization',bayesianop{fold_idx}.L2Regularization,...
        'GradientThresholdMethod',char(bayesianop{fold_idx}.GradientThresholdMethod),...
        'GradientThreshold',bayesianop{fold_idx}.GradientThreshold, ...
        'Plots','training-progress',...
        'SequenceLength','longest', ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',10, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose', 1, ...
        'ExecutionEnvironment','gpu');
    
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
    newClassLayer = classificationLayer('Name','new_classoutput','Classes',categorical([0 1]) ,'ClassWeights',[bayesianop{fold_idx}.classweightfactor;1]);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000',newClassLayer);
    
    % vgg16 train model and result
    [nettrained,info]  = trainNetwork(TrX1,categorical(sf_TrY.'),lgraph,options);

    % ResNet50 train model and result
    cv_result(fold_idx).nettrained = trainNetwork(TrX1,categorical(sf_TrY.'),lgraph,options);

    cv_result(fold_idx).testLabels = categorical(testLabels);
    cv_result(fold_idx).preLabel =classify(cv_result(fold_idx).nettrained,TstX1);
    cv_result(fold_idx).percentLabel = predict(cv_result(fold_idx).nettrained,TstX1);

    CWeighted{fold_idx} = confusionchart(categorical(testLabels),cv_result(fold_idx).preLabel, Title="With Class Weighting",RowSummary="row-normalized",ColumnSummary="column-normalized");
    
    PrecisionWeighted = CWeighted{fold_idx}.NormalizedValues(2,2) / sum(CWeighted{fold_idx}.NormalizedValues(2,:));
    RecallWeighted = CWeighted{fold_idx}.NormalizedValues(2,2) / sum(CWeighted{fold_idx}.NormalizedValues(:,2));
    cv_result(fold_idx).f1Weighted = max(0.00001,(2*PrecisionWeighted*RecallWeighted) / (PrecisionWeighted+RecallWeighted));
    
    % test section CT count
    testsection = subject_datanum(cv_result(fold_idx).testIdx);

    % load to value
    cnt6 = 1;
    for i = 1 : size(testsection,2)
        for j = 1 : testsection(i)
            cv_result(fold_idx).eachpercentLabel{i}(j) = cv_result(fold_idx).percentLabel(cnt6,2);
            cv_result(fold_idx).eachpredLabel{i}(j) = cv_result(fold_idx).preLabel(cnt6);
            cv_result(fold_idx).eachtestLabel{i}(j) = cv_result(fold_idx).testLabels(cnt6);
            cnt6 = cnt6 + 1;
        end
    end

    % each f1 score by person
    for m = 1 : size(cv_result(fold_idx).testIdx,2)
        eachCWeighted = confusionchart(categorical(cv_result(fold_idx).eachtestLabel{m}),cv_result(fold_idx).eachpredLabel{m}, Title="With Class Weighting",RowSummary="row-normalized",ColumnSummary="column-normalized");
        
        PrecisionWeighted = eachCWeighted.NormalizedValues(2,2) / sum(eachCWeighted.NormalizedValues(2,:));
        RecallWeighted = eachCWeighted.NormalizedValues(2,2) / sum(eachCWeighted.NormalizedValues(:,2));
        cv_result(fold_idx).eachf1score{m} = max(0,(2*PrecisionWeighted*RecallWeighted) / (PrecisionWeighted+RecallWeighted));
    end

    % mean f1 score
    cv_result(fold_idx).meanf1score = mean(cell2mat(cv_result(fold_idx).eachf1score));
    
    % section error
    for n = 1 : size(cv_result(fold_idx).testIdx,2)
        cnt2 = 1;
        cnt3 = 1;
        cnt4 = 1;
        if all(categorical(cv_result(fold_idx).eachpredLabel{n}) == '0') % no pred L3
            maxValue = max(cv_result(fold_idx).eachpercentLabel{n});
            cv_result(fold_idx).noprednumber{n} = 5 * (n - 1) + fold_idx;
            for u = 1 : size(cv_result(fold_idx).eachpercentLabel{n},2)  
                if cv_result(fold_idx).eachpercentLabel{n}(u) == maxValue
                    cv_result(fold_idx).L3predindex{1,n} = u;
                end
                if cv_result(fold_idx).eachtestLabel{n}(u) == categorical(1)
                    cv_result(fold_idx).L3testindex{n}(cnt4) = u;
                    cnt4 = cnt4 + 1;
                end
            end
        else
            for u = 1 : size(cv_result(fold_idx).eachpercentLabel{n},2)            
                if cv_result(fold_idx).eachtestLabel{n}(u) == categorical(1)
                    cv_result(fold_idx).L3testindex{n}(cnt2) = u;
                    cnt2 = cnt2 + 1;
                end
                if cv_result(fold_idx).eachpredLabel{n}(u) == categorical(1)
                    cv_result(fold_idx).L3predindex{1,n}(cnt3) = u;
                    cnt3 = cnt3 + 1;
                end
            end
        end
        % calculate median
        cv_result(fold_idx).mediantest{n} = median(cv_result(fold_idx).L3testindex{1,n});
        cv_result(fold_idx).medianpred{n} = median(cv_result(fold_idx).L3predindex{n});
        % ct number standard
        result_num = 5 * (n - 1) + fold_idx;
        medianresult{result_num,1} = cv_result(fold_idx).mediantest{n};
        medianresult{result_num,2} = cv_result(fold_idx).medianpred{n};
        % fold standard
        result_num2 = (fold_idx - 1) * 30 + n;
        medianresult{result_num2,3} = cv_result(fold_idx).mediantest{n};
        medianresult{result_num2,4} = cv_result(fold_idx).medianpred{n};
        % calculate median error
        cv_result(fold_idx).errorpred{n} = abs(cv_result(fold_idx).mediantest{n} - cv_result(fold_idx).medianpred{n});
        
    end
    cv_result(fold_idx).meanerrorpred = mean(cell2mat(cv_result(fold_idx).errorpred));
end

% save to result
% save('D:\yachae_sw\code\classification\result\ResNet50_Result_150_cv12.mat', 'cv_result');
% save('D:\yachae_sw\code\classification\result\medianresult.mat', 'medianresult');
