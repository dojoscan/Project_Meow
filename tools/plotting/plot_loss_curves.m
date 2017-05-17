close all
clear all
clc

filename = 'C:\Master Chalmers\2 year\volvo thesis\excel\acc.xlsx';
mark_ind = 1:20:1000;
data = xlsread(filename);
figure
%semilogy(data(:,1),data(:,2),'r--',data(:,1),data(:,3),'g--',data(:,1),data(:,4),'b--')
%hold on
%leg{e} = [group ', \eta = ' num2str(10^(-(e)))];
semilogy(data(:,1),smooth(data(:,2)),'m-o','MarkerIndices',mark_ind)
hold on
semilogy(data(:,1),smooth(data(:,3)),'c-*','MarkerIndices',mark_ind)
hold on
semilogy(data(:,1),smooth(data(:,4)),'k-+','MarkerIndices', mark_ind)
%set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold off
xlabel('Iteration')
ylabel('Training Loss')
legend('Bounding Box','Classification','Object Confidence')