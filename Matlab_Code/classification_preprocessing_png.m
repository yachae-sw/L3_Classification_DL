clc;close all;clear;
%% Double Dicom data

% load dicom file path
Base = 'D:\yachae_sw\CTImages\';
List = dir(fullfile(Base, 'CT_DCM_150', '*.*'));
List = List([List.isdir]);
SubFolder = {List.name};
SubFolder(ismember(SubFolder, {'.', '..'})) = [];
Source_dir1 = cellfun(@(c)[Base 'CT_DCM_150\' c '\'],SubFolder,'uni',false);

% make each folder
% for i = 1 : length(Source_dir1)
%     A = SubFolder{:,i};
%     B = 'DICOM';
%     mkdir (A,B)
% end

% load excel file
[AP_Data_Label] = readcell(fullfile(Base, 'CT_Mask_nii_150/CT_L3_label_20230909.xlsx'));

% load to L3 location
AP_L3_Label = cell(size(AP_Data_Label, 1) - 1, 1);
for i = 2 : size(AP_Data_Label, 1)
    L3_Answer_Start = AP_Data_Label{i, 6};
    L3_Answer_End = AP_Data_Label{i, 7};
    L3_End_Reverse=fliplr(AP_Data_Label{i,5}-L3_Answer_Start)+1;
    L3_Start_Reverse=fliplr(AP_Data_Label{i,5}-L3_Answer_End)+1;

    AP_L3_Label{i-1} = zeros(AP_Data_Label{i, 5}, 1);
    for j = 1 : AP_Data_Label{i, 5}
        if L3_Start_Reverse <= L3_End_Reverse
            AP_L3_Label{i-1}(L3_Start_Reverse:L3_End_Reverse) = 1;
        end
    end
end

%% Control HU value and Save data in struct form

% create struct
cell_max = size(SubFolder,2);
Dicom2 = struct('Dicomname', cell(1, cell_max), 'Subject', cell(1, cell_max));

count = 0;
for j = 1 : size(SubFolder, 2)
    str1 = dir(fullfile(Source_dir1{j}, '*.dcm*')); % load to dicom file
    for k = AP_Data_Label{j+1, 3} : AP_Data_Label{j+1, 10}
        if AP_Data_Label{j+1, 3} == 1
            try
                Temp = [Source_dir1{j}, str1(k).name];
                Dicom2(j).Dicomname{k,:} = [str1(k).name];
                dicomimage = dicomread(Temp);
                dicomimage32 = int32(dicomimage);
                info = dicominfo(Temp);
                % image resize
                reimage = imresize(dicomimage32, [256 256]);

                for b = 1 : size(reimage,1)
                    for c = 1 :size(reimage,2)
                        hounsfieldImage(b,c) = int32(reimage(b,c))*info.RescaleSlope + int32(info.RescaleIntercept); % control HU
                        % Hu max value setting
                        if hounsfieldImage(b,c) > 1000
                            hounsfieldImage(b,c) = 1000;
                        % Hu min value setting
                        elseif hounsfieldImage(b,c) < -1023
                            hounsfieldImage(b,c) = -1023;
                        end
                    end
                end

                Dicom2(j).Subject{k,:} = hounsfieldImage;
            catch err
                break;
            end
        else
            % start file number is not 1
            minus = AP_Data_Label{j+1, 3}-1;
            try
                Temp = [Source_dir1{j}, str1(k).name];
                Dicom2(j).Dicomname{k-minus,:} = [str1(k).name];
                dicomimage = dicomread(Temp);
                dicomimage32 = int32(dicomimage);
                info = dicominfo(Temp);
                reimage = imresize(dicomimage32, [256 256]);
                for b = 1 : size(reimage,1)
                    for c = 1 :size(reimage,2)
                        hounsfieldImage(b,c) = int32(reimage(b,c))*info.RescaleSlope + int32(info.RescaleIntercept);
                        if hounsfieldImage(b,c) > 1000
                            hounsfieldImage(b,c) = 1000;
                        elseif hounsfieldImage(b,c) < -1023
                            hounsfieldImage(b,c) = -1023;
                        end
                    end
                end
                Dicom2(j).Subject{k-minus,:} = hounsfieldImage;
            catch err
                break;
            end
        end
    end
    % Visualize running numbers
    count = count + 1;
    disp(count)
end

% save to dicom
save("D:\yachae_sw\code\classification\data\Dicom6.mat",'Dicom2','-mat','-v7.3')

%% Save name in struct form

cnt = 1;
for i = 1 : size(Dicom2,2)
    cnt = 1;
    for j = 1 : AP_Data_Label{i+1,5}
        if AP_L3_Label{i,:}(j,1) == 0 % L3 label
            Dicom2(i).Label{cnt,:} = AP_L3_Label{i,:}(j,1);

            str = Dicom2(i).Dicomname{j,:};
            str_trim = str(1:strfind(str,'.')-1);
            new_name = replace(str_trim,"'",""); % remove special characters
            Dicom2(i).JPGImageName{cnt,:} = [new_name,'.png'];
        else
            Dicom2(i).Label{cnt,:} = AP_L3_Label{i,:}(j,1);
            
            str = Dicom2(i).Dicomname{j,:};
            str_trim = str(1:strfind(str,'.')-1);
            new_name = replace(str_trim,"'","");
            Dicom2(i).JPGImageName{cnt,:} = [new_name,'.png'];
        end
        cnt = cnt + 1;
    end
end

%% Change data type

cnt1 = 1;
for i = 1 : size(Dicom2,2)
    for j = 1 : size(Dicom2(i).Dicomname,1)
        Dicom2(i).RawImageRev{cnt1,:} = int16(Dicom2(i).Subject{j,:}); % uint to int
        cnt1 = cnt1 + 1;
    end
    cnt1 = 1;
end
% Dicom2 = rmfield(Dicom2, 'Subject');

%% control image contrast

for i = 1 : size(Dicom2,2)
    cnt = 1;
    for j = 1 : size(Dicom2(i).RawImageRev,1)
            % calculate stretchlimit section
            cnt1 = 1;
            cnt2 = 1;
            loadimage = Dicom2(i).RawImageRev{j,:};
            for b = 1 : 256
                for c = 1 : 256
                    if loadimage(b,c) <= -1023
                        cnt1 = cnt1 + 1;
                    elseif loadimage(b,c) >= 1000
                        cnt2 = cnt2 + 1;
                    end
                end            
            end
            stretchlimst = round(cnt1 / (256 * 256),4);
            stretchlimend = 1 - max(0.02, round(cnt2 / (256 * 256),4));
            Dicom2(i).stretchlim{j,1} = stretchlimst;
            Dicom2(i).stretchlim{j,2} = stretchlimend;

            imadjustimage = imadjust(Dicom2(i).RawImageRev{j,:},stretchlim(Dicom2(i).RawImageRev{j,:},[stretchlimst stretchlimend]),[]);
            Dicom2(i).ImageAdjust{cnt,:} = imadjustimage;
            Dicom2(i).JPGImageName{cnt,:} = Dicom2(i).JPGImageName{j,:};
            cnt = cnt + 1;
    end
end

%% image and label combine to png

AP = struct;
WriteOrNot = 1;

cnt = 1;
for i = 1 : size(Dicom2,2)
    AP(i).Label = Dicom2(i).Label ;
    for j = 1 : size(Dicom2(i).ImageAdjust,1)
        if WriteOrNot == 1
            Temp = [Base,'localization_150_detection\',Dicom2(i).JPGImageName{j,:}];
            
            if ~exist([Base,'localization_150_detection'], 'dir')
                mkdir([Base,'localization_150_detection'])
            end    
            
            AP(i).JPGWriteImage{j,:} = imadjust(uint16(Dicom2(i).ImageAdjust{j,:})); % change data type
            imwrite(AP(i).JPGWriteImage{j,:}, Temp,BitDepth=16);
            label(cnt) = Dicom2(i).Label{j}; % save label
            cnt = cnt + 1; 
        else
            continue;
        end
    end
end

% save to AP and Label
save('D:\yachae_sw\code\classification\data\AP6.mat','AP')
save('D:\yachae_sw\code\classification\data\label6.mat','label')
