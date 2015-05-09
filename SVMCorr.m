format longG;
c =yahoo;
%download data from Yahoo
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

%Interplate daily price based on NASDAQ
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

NAS2=ones(length(NAS)-1,1);
DAX2=ones(length(NAS)-1,1);
FTSE2=ones(length(NAS)-1,1);
NIkkei2=ones(length(NAS)-1,1);
HS2=ones(length(NAS)-1,1);
DJIA2=ones(length(NAS)-1,1);
SP2=ones(length(NAS)-1,1);
USO2=ones(length(NAS)-1,1);
EUR2=ones(length(NAS)-1,1);
GLD2=ones(length(NAS)-1,1);

%Calculate daily returns of indices
for i=1:length(NAS)-1;
    NAS2(i) = (NAS(i+1)-NAS(i))/NAS(i);
    DAX2(i) = (DAX(i+1)-DAX(i))/DAX(i);
    FTSE2(i) = (FTSE(i+1)-FTSE(i))/FTSE(i);
    NIkkei2(i) = (NIkkei(i+1)-NIkkei(i))/NIkkei(i);
    HS2(i) = (HS(i+1)-HS(i))/HS(i);
    SP2(i) = (SP(i+1)-SP(i))/SP(i);
    DJIA2(i) = (DJIA(i+1)-DJIA(i))/DJIA(i);
    USO2(i) = (USO(i+1)-USO(i))/USO(i);
    EUR2(i) = (EUR(i+1)-EUR(i))/EUR(i);
    GLD2(i) = (GLD(i+1)-GLD(i))/GLD(i);
end

index=[NAS2,SP2,DJIA2,DAX2,FTSE2,NIkkei2,HS2,USO2,EUR2,GLD2];

%Calculate correlation with repect to NASDAQ accross different lags
lag=ones(size(index,2),11);
for i=1:size(index,2);
    for j=-5:5;
        if j<=0;
        cor=corrcoef(index(1:length(index)+j,i),index(-j+1:length(index),1));
        lag(i,j+6)=cor(1,2);
        end
        if j>=1;
        cor=corrcoef(index(j+1:length(index),i),index(1:length(index)-j,1));
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
for i=1:length(index);
    for j=1:size(index,2)
       if index(i,j)<0;
           index(i,j)=-1;
       else index(i,j)=1;
       end
    end
end
% Using svm to predict the trend of NASDAQ index
% Train the data
ind=fix(2*length(index)/3);
trainlabel=index(1:ind,1);
testlabel=index(ind:end,1);
Accuracy=ones(1,size(index,2)-3);
for i=4:size(index,2)
    traindata=index(1:ind,i);
    % Use cross validation to find the best parameters
    [bestacc,bestc,bestg] = SVMcg(trainlabel,traindata,-2,2,-2,2,3,0.5,0.5,0.5);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
    model = svmtrain(trainlabel,traindata,cmd);
    % Test and predict the trend of NASDAQ index
    testdata=index(ind:end,i);
    [predict_label, accuracy, prob_estimates] = svmpredict(testlabel, testdata, model);
    Accuracy(1,i-3)=accuracy(1);
end
bar(Accuracy);
set(gca,'XTickLabel',{'DAX','FTSE','NIKKEI','HANG SENG','OIL','EUR/USD','GOLD'})
ylabel('Accuracy') % y-axis label
ylim([40 75])
title('Fig.2 Prediction accuracy of NASDAQ by single feature')
%% Test multiple features
ind=fix(2*length(index)/3);
trainlabel=index(1:ind,3);
traindata=index(1:ind,4:5);
% Use cross validation to find the best parameters
[bestacc,bestc,bestg] = SVMcg(trainlabel,traindata,-2,2,-2,2,5,0.5,0.5,2);
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(trainlabel,traindata,cmd);
% Test and predict the trend of NASDAQ index
testdata=index(ind+1:end,4:5);
testlabel=index(ind+1:end,3);
[predict_label, accuracy, prob_estimates] = svmpredict(testlabel, testdata, model);
%% Using svm to predict the trend of NASDAQ index with lag -1
% Train the data
ind=fix(2*length(index)/3);
trainlabel=index(2:ind,1);
testlabel=index(ind+1:end,1);
Accuracy=ones(1,size(index,2)-3);
for i=4:size(index,2)
    traindata=index(1:ind-1,i);
    % Use cross validation to find the best parameters
    [bestacc,bestc,bestg] = SVMcg(trainlabel,traindata,-2,2,-2,2,3,0.5,0.5,0.5);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
    model = svmtrain(trainlabel,traindata,cmd);
    % Test and predict the trend of NASDAQ index
    testdata=index(ind:end-1,i);
    [predict_label, accuracy, prob_estimates] = svmpredict(testlabel, testdata, model);
    Accuracy(1,i-3)=accuracy(1);
end
bar(Accuracy);
set(gca,'XTickLabel',{'DAX','FTSE','NIKKEI','HANG SENG','OIL','EUR/USD','GOLD'})
ylabel('Accuracy') % y-axis label
ylim([40 75])
title('Fig.3 Prediction accuracy of NASDAQ by single feature with lag -1')
