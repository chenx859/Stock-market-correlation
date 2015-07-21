%% Initialization
clear ; close all; clc
format longG;
%% download data from Yahoo
c =yahoo;
SP = fetch(c,'^GSPC','close','4/10/2006','4/10/2015');
NAS = fetch(c,'^IXIC','close','4/10/2006','4/10/2015');
DAX = fetch(c,'^GDAXI','close','4/10/2006','4/10/2015');
FTSE = fetch(c,'^FTSE','close','4/10/2006','4/10/2015');
NIkkei= fetch(c,'^N225','close','4/10/2006','4/10/2015');
HS= fetch(c,'^HSI','close','4/10/2006','4/10/2015');
DJIA = fetch(c,'DIA','close','4/10/2006','4/10/2015');
GLD= fetch(c,'GLD','close','4/10/2006','4/10/2015');
USO= fetch(c,'USO','close','4/10/2006','4/10/2015');
EUR=csvread('exchange.csv');

close(c)

%% Interplate daily price based on NASDAQ
DAX=interp1(DAX(:,1),DAX(:,2),NAS(:,1));
FTSE=interp1(FTSE(:,1),FTSE(:,2),NAS(:,1));
NIkkei=interp1(NIkkei(:,1),NIkkei(:,2),NAS(:,1));
HS=interp1(HS(:,1),HS(:,2),NAS(:,1));
SP=interp1(SP(:,1),SP(:,2),NAS(:,1));
DJIA=interp1(DJIA(:,1),DJIA(:,2),NAS(:,1));
GLD=interp1(GLD(:,1),GLD(:,2),NAS(:,1));
USO=interp1(USO(:,1),USO(:,2),USO(:,1));
EUR=interp1(EUR(:,1),EUR(:,2),NAS(:,1));
NAS=interp1(NAS(:,1),NAS(:,2),NAS(:,1));

index=[NAS,SP,DJIA,DAX,FTSE,NIkkei,HS,USO,EUR,GLD];
% Change daily price to daily returns
indexRet=price2ret(index);

%Calculate correlation with repect to NASDAQ accross different lags
lag=ones(size(indexRet,2),11);
for i=1:size(indexRet,2);
    for j=-5:5;
        if j<=0;
        cor=corrcoef(indexRet(1:length(indexRet)+j,i),indexRet(-j+1:length(indexRet),1));
        lag(i,j+6)=cor(1,2);
        end
        if j>=1;
        cor=corrcoef(indexRet(j+1:length(indexRet),i),index(1:length(indexRet)-j,1));
        lag(i,j+6)=cor(1,2);
        end
    end
end

x=-5:5;
plot(x,lag','LineWidth', 2)
legend('NASDAQ','S&P500','DJIA','DAX','FTSE','NIKKEI','HANG SENG','OIL','EUR/USD','GOLD')
set(gca,'FontSize',18,'fontWeight','bold')
xlabel('Lags') % x-axis label
ylabel('Correlation with respect to NASDAQ') % y-axis label
title('fig.1 Correlation with respect to NASDAQ accross lags')
%% Install libsvm tool box
mex -setup
make
%% Classiy NASDAQ daily returns into 2 groups, -1 and 1.
indexRet(indexRet)=0;
indexRet(indexRet)=1;
% Using svm to predict the trend of NASDAQ index
% Split data into train dataset, cross-validation dataset, test dataset 
trainIdx=1:1510; % train dataset index
crIdx=1511:1887; % Cross validation dataset index
testIdx=1888:2265; % test dataset index
Accuracy=ones(1,size(index,2)-3);


for i=4:size(indexRet,2)
    traindata=indexRet(trainIdx,i);
    crdata=indexRet(crIdx,i);
    testdata=indexRet(testIdx,i);
    trainlabel=indexRet(trainIdx,1);
    crlabel=indexRet(crIdx,1);
    testlabel=indexRet(testIdx,1);
    
    % Use cross validation to find the best parameters
    [bestc, bestg] = SVMcr(traindata,trainlabel, crdata, crlabel);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
    model = svmtrain(trainlabel,traindata,cmd);
    plotData(traindata,trainlabel);
    % Test and predict the trend of NASDAQ index
    [predict_label, accuracy, prob_estimates] = svmpredict(testlabel, testdata, model);
    Accuracy(1,i-3)=accuracy(1);
end
bar(Accuracy);
set(gca,'XTickLabel',{'DAX','FTSE','NIKKEI','HANG SENG','OIL','EUR/USD','GOLD'})
ylabel('Accuracy') % y-axis label
ylim([40 75])
title('Fig.2 Prediction accuracy of NASDAQ by single feature')
%% Test multiple features
traindata=indexRet(trainIdx,4:7);
crdata=indexRet(crIdx,4:7);
testdata=indexRet(testIdx,4:7);
trainlabel=indexRet(trainIdx,1);
crlabel=indexRet(crIdx,1);
testlabel=indexRet(testIdx,1);
% Use cross validation to find the best parameters
[bestc, bestg] = SVMcr(traindata,trainlabel, crdata, crlabel);
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(trainlabel,traindata,cmd);
% Test and predict the trend of NASDAQ index
[predict_label, accuracy, prob_estimates] = svmpredict(testlabel, testdata, model);
%% Using svm to predict the trend of NASDAQ index with lag -1
% Train the data
trainIdx=1:1510; % train dataset index
crIdx=1511:1887; % Cross validation dataset index
testIdx=1888:2265; % test dataset index
Accuracy=ones(1,size(index,2)-3);
for i=4:size(index,2)
    traindata=indexRet(trainIdx(1:end-1),i);
    crdata=indexRet(crIdx(1:end-1),i);
    testdata=indexRet(testIdx(1:end-1),i);
    trainlabel=indexRet(trainIdx(2:end),1);
    crlabel=indexRet(crIdx(2:end),1);
    testlabel=indexRet(testIdx(2:end),1);
    
    % Use cross validation to find the best parameters
    [bestc, bestg] = SVMcr(traindata,trainlabel, crdata, crlabel);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
    model = svmtrain(trainlabel,traindata,cmd);
    % Test and predict the trend of NASDAQ index
    [predict_label, accuracy, prob_estimates] = svmpredict(testlabel, testdata, model);
    Accuracy(1,i-3)=accuracy(1);
end
bar(Accuracy);
set(gca,'XTickLabel',{'DAX','FTSE','NIKKEI','HANG SENG','OIL','EUR/USD','GOLD'})
ylabel('Accuracy') % y-axis label
ylim([40 75])
title('Fig.3 Prediction accuracy of NASDAQ by single feature with lag -1')
