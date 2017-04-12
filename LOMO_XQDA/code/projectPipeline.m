%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Final Year Project Pipeline to handle all available classification,
%%feature extraction methods automatically


%%Feature extraction works with auto detect file pre-existence and forcing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Notes
%Matlab uses 1-indexing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Setup- set directories, number of experiment repeats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imgDir = '../images/';
featuresDir='../data/';
featureExts='*.mat';
resultsDir='../results/';
addpath(imgDir);
%%Experiment parameters
numFolds=10; %Number of times to repeat experiment
numRanks = 100;

%%Features
LOMO_F=1;
MACH_F=2;
%%Classifiers
XQDA_F=1;
%%Which feature extractors to run
%%Which classifiers to run
featureExtractors= [{LOMO_F, @LOMO};{MACH_F, @MACH}];%%,{MACH, @MACH}
featureName={'LOMO.mat', 'MACH.mat'};
featureForce=false; 
featureExtractorsRun=[MACH_F];%LOMO_F
classifiers= [{XQDA_F, @XQDA}];
classifiersRun=[XQDA_F];
classifierName={'XQDA'};

features=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Import all images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgList = dir([imgDir, '*.png']);%[imgDir, '*.png']
n = length(imgList);

%% Allocate memory
info = imfinfo([imgDir, imgList(1).name]);
images = zeros(info.Height, info.Width, 3, n, 'uint8');

%% read images
for i = 1 : n
    images(:,:,:,i) = imread([imgDir, imgList(i).name]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Extract features using all feature extraction methods
%%Store in data directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%if feature set doesn't already exist

%%Perform feature extraction
disp('Checking if features to Extract already exist or forced \n')
for i=1:length(featureExtractorsRun)
    %Check if features already exist 
    featureList=dir([featuresDir, '*.mat']);
    featuresAvail=[featureList.name]
    currFeatureName=cell2mat(featureName(featureExtractorsRun(i)))
    %Run feature extraction function
    %If being forced or features Available doesnt already exist
    if(featureForce || any(strcmp(currFeatureName,featuresAvail))==0)
        idx=find(cell2mat(featureExtractors(:,1))==featureExtractorsRun(i),1);
        %Could do error checking here to test match exists: 1x0
        %featureID= cell2mat(featureExtractors(u,1));
        featureFunct= cell2mat(featureExtractors(idx,2))
        fprintf('Extracting current feature %s, place in data directory \n',currFeatureName)
        features=featureFunct(images);
        %{
        for u=1:size(featureExtractors,1)
            featureID= cell2mat(featureExtractors(u,1));
            featureFunct= cell2mat(featureExtractors(u,2));       
            if(featureExtractorsRun(i)==featureID)
                fprintf('Extracting current feature %s',currFeatureName)
                %features(i)=featureFunct(images);
            end
        end
        %}
    else
      fprintf('Already exists. Not extracting current feature %s \n',currFeatureName)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Perform classification on all extracted features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Get features
%%gets data on all files in directory, store in array of structs
%% probFea and galFea
numImages=0;
featuresList = dir(strcat(featuresDir,featureExts));%data/*.mat
%[featuresDir, '*.mat']

%%Loads extracted features into arrays
%for i= 1: size(featuresList,1)
for i=1:length(featureExtractorsRun)
    currFeatureName=cell2mat(featureName(featureExtractorsRun(i)))
    %%loads variables called descriptors from file
    %%featureList(1,i) as 1x1 struct
    %fileName=strcat(featuresDir,featuresList(i).name);
    if(any(currFeatureName==featuresAvail)~=0)
        fprintf('Currently loading features %s into matrices \n',currFeatureName);
        load(currFeatureName, 'descriptors');
        numImages= size(descriptors,1)/2;
        galFea(i,:,:) = descriptors(1 : numImages, :);
        probFea(i,:,:) = descriptors(numImages + 1 : end, :);
        clear descriptors
    else
        fprintf('Could not load features %s into matrices as folder didnt exist \n',currFeatureName);
    end
end
%%For all extracted features 
%%For all classification techniques
%%Get results
noTests=length(classifiersRun)*size(galFea,1);%%galfea i is number rows?
cms = zeros(noTests, numFolds, numRanks);

%%Select classifiers want to run
for i=1:length(classifiersRun)
   % for u=1:size(classifiers,1)
        %potentClassifier=cell2mat(classifiers(u));
        idx=find(cell2mat(classifiers(:,1))==classifiersRun(i),1);
        currClassifierId=cell2mat(classifiers(idx,1));
        currClassifierFunct=cell2mat(classifiers(idx,2));
        currClassifierName=cell2mat(classifierName(currClassifierId));
        fprintf('Currently running classifier %s \n',currClassifierName)
       % if(classifiersRun(i)==potentClassifier(1)) 
            %method=currClassifier(2);
            %method = potentClassifier(2);
            %%For every set of features
            for ft=1:size(galFea,1)
                %Repeat classification process numFolds times
                for iter=1:numFolds
                    p = randperm(numImages);
                    galFea1 = squeeze(galFea( ft,p(1:numImages/2), : ));
                    probFea1 = squeeze(probFea(ft, p(1:numImages/2), : ));

                    t0 = tic;
                    [W, M] = XQDA(galFea1, probFea1, (1:numImages/2)', (1:numImages/2)');

                    %{
                    %% if you need to set different parameters other than the defaults, set them accordingly
                    options.lambda = 0.001;
                    options.qdaDims = -1;
                    options.verbose = true;
                    [W, M] = XQDA(galFea1, probFea1, (1:numImages/2)', (1:numImages/2)', options);
                    %}

                    clear galFea1 probFea1
                    trainTime = toc(t0);
                    %Squeeze removes singleton dimensions
                    galFea2 = squeeze(galFea(ft, p(numImages/2+1 : end), : ));
                    probFea2 = squeeze(probFea(ft, p(numImages/2+1 : end), : ));

                    t0 = tic;
                    dist = MahDist(M, galFea2 * W, probFea2 * W);
                    clear galFea2 probFea2 M W
                    matchTime = toc(t0);      

                    fprintf('Fold %d: ', iter);
                    fprintf('Training time: %.3g seconds. ', trainTime);    
                    fprintf('Matching time: %.3g seconds.\n', matchTime); 
                    %CMS is for every feature set, repeated 10 times, 100 ranks.
                    %Different cms for every classifier
                    cms(ft, iter,:) = EvalCMC( -dist, 1 : numImages / 2, 1 : numImages / 2, numRanks );
                    clear dist           

                    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(ft, iter,[1,5,10,15,20]) * 100);

                end
                %Mean for every feature set, classifier combination
                meanCms = mean(squeeze(cms(ft,:,:)));
                figure
                plot(1 : numRanks, meanCms)
                title(sprintf('CMS Curve for Classifier %s and feature set %s', currClassifierName, cell2mat(featureName(1,ft))))
                xlabel('No. Ranks of ordered Gallery Images') % x-axis label
                ylabel('% Gallery Images that contain match within that rank') % y-axis label

                csvFileName=strcat(resultsDir,currClassifierName,'_',int2str(ft));
                csvwrite(csvFileName,meanCms)
                %%type csvlist.dat

                fprintf('The average performance:\n');
                fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
            end
       % end
   % end
end









