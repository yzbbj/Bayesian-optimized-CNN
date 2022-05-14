%%This code uses Bayesian optimazation algorithm to optimze the arguments 
%%of the CNN network

%%Load initial data
digitDatasetPath = fullfile('.');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds)

%%Load orininal training and validation sets
number_of_each_train_test = 150;
[imds_of_train_set,imds_of_test_set] = splitEachLabel(imds, ...
    number_of_each_train_test,'randomize');
Xtrain=imds_of_train_set.Files;
YTrain=imds_of_train_set.Labels;
Xtest=imds_of_test_set.Files;
YTest=imds_of_test_set.Labels;

%%Specify training and validation sets,which are the input of the Bayesian 
% optimazation algorithm

%Each set is a 4-dimensional matrix,which is used to storage 3-dimensional
%matrixs,each of which represents a picture,at the size of 64*86*3
XTrain = zeros(64,86,3,600);
for i = 1:600   %The size of train sets is 150*4    
    XTrain(:,:,:,i) = imread(Xtrain{i});
end
size(XTrain)
XTrain=uint8(XTrain);

XTest = zeros(64,86,3,200);
for i = 1:200   %The size of validation sets is 50*4
    XTest(:,:,:,i) = imread(Xtest{i});
end
size(XTest)
XTest=uint8(XTest);

idx = randperm(numel(YTest),30);
XValidation = XTest(:,:,:,idx);
XTest(:,:,:,idx) = [];
YValidation = YTest(idx);
YTest(idx) = [];
figure;


%%Determine which variables are going to be optimized
OptimizableVariables = [
    optimizableVariable('SectionDepth',[1 3],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-1 1],'Transform','log')
    optimizableVariable('Momentum',[0.9 0.98])
    optimizableVariable('L2Regularization',[1e-10 1e-2], ...
    'Transform','log')];

%%Create objective function of Bayesian optimization
ObjectiveFunction = createObjectiveFunction(XTrain,YTrain,XValidation, ...
    YValidation);

%%Perform Bayesian Optimization
%bayesopt is used to find the global minimum of a function using Bayesian
BayesObject = bayesopt(ObjectiveFunction,OptimizableVariables, ...
    'MaxObjectiveEvaluations',2,...
    'MaxTime',0.5*60*60, ...    
    'UseParallel',false);

%%Evaluate the best network
BestIdx = BayesObject.IndexOfMinimumTrace(end);
FileName = BayesObject.UserDataTrace{BestIdx};
BestNetwork = load(FileName);
valError = BestNetwork.valError;
BestOptions = BestNetwork.options;

%Use the network to predict the labels and calculate test error
[YPredicted,probabilities] = classify(BestNetwork.trainedNet,XValidation);
%label = classify(net,I)
testError = 1 - mean(YPredicted == YValidation);
numYValidation = numel(YValidation);
testErrorSE = sqrt(testError*(1-testError)/numYValidation);
testError95CI = [testError - 1.96*testErrorSE, testError + 1.96*testErrorSE];

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
Confusion_Matrix = confusionchart(YValidation,YPredicted);
Confusion_Matrix.Title = 'Confusion Matrix for Test Data';
Confusion_Matrix.ColumnSummary = 'column-normalized';
Confusion_Matrix.RowSummary = 'row-normalized';

%%Display some of the test images with the predictions and the
%%possibilities of the preditcions
figure
idx = randperm(numel(YValidation),4);
for i = 1:numel(idx)
    subplot(2,2,i)
    imshow(XValidation(:,:,:,idx(i)));
    probability = num2str(100*max(probabilities(idx(i),:)),3);
    label = [YPredicted(idx(i)),', ',probability,'%'];
    title(label)
end


%%Define objective Function
function ObjectiveFunction = createObjectiveFunction(XTrain,YTrain, ...
    XValidation,YValidation)
ObjectiveFunction = @valErrorFunction;
    function [valError, variableConstriants,fileName] = valErrorFunction( ...
            OptimizableVariables)
        
        %Define CNN network
        imageSize = [64 86 3];
        numClasses = numel(unique(YTrain));
        numFilters = round(32/sqrt( ...
            OptimizableVariables.SectionDepth));
        layers = [
            imageInputLayer(imageSize)
          
            convBlock(3,numFilters, ...
            OptimizableVariables.SectionDepth)  
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            
            convBlock(3,2*numFilters, ...
            OptimizableVariables.SectionDepth)
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            
            convBlock(3,4*numFilters, ...
            OptimizableVariables.SectionDepth)
            averagePooling2dLayer(8)
 
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
        miniBatchSize = 128;
        validationFrequency = floor(numel(YTrain)/miniBatchSize);
        options = trainingOptions('sgdm', ...
            'InitialLearnRate',OptimizableVariables.InitialLearnRate, ...
            'Momentum',OptimizableVariables.Momentum, ...
            'MaxEpochs',60, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',40, ...
            'LearnRateDropFactor',0.1, ...
            'MiniBatchSize',miniBatchSize, ...
            'L2Regularization',OptimizableVariables.L2Regularization, ...
            'Shuffle','every-epoch', ...
            'Verbose',false, ...         
            'ValidationData',{XValidation,YValidation}, ...
            'ValidationFrequency',validationFrequency);
        %Data augmentation
        pixelRange = [-4 4];
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange);
        datasource = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
            'DataAugmentation',imageAugmenter);
        trainedNet = trainNetwork(datasource,layers,options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))
        YPredicted = classify(trainedNet,XValidation);
        valError = 1 - mean(YPredicted == YValidation);
        fileName = num2str(valError) + ".mat";
        save(fileName,'trainedNet','valError','options')
        variableConstriants = [];
    end
end

%%Define 
function layers = convBlock(FilterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(FilterSize,numFilters,'Padding','same')
    batchNormalizationLayer
   reluLayer];
layers = repmat(layers,numConvLayers,1);
end

        
        
        
        