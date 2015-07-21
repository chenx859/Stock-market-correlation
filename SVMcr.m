function [bestc,bestg] = SVMcr(y, X, yval, Xval)
Crange=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
gRange=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
bestc = 1;
bestg = 1;
accuracy=0;

for i=1:length(Crange)
    for j=1:length(gRange)
        model = svmtrain(y, X,['-c ',num2str(Crange(i)),' -g ',num2str(gRange(j))]);
        [predict_label, accuracyTemp, prob_estimates] = svmpredict(yval, Xval, model);
        if accuracyTemp>accuracy
            bestc=Crange(i);
            bestg=gRange(j);
        end
    end
end