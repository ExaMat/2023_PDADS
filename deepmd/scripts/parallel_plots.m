function parallel_plots(filename)
%filename is a string for a .csv file that is in your matlab path
R=readtable(filename);
%edit below for different combinations, this creates parallel plot in paper
T=removevars(R,{'job','generation','uuid','birth_id','start_eval_time','stop_eval_time','eval_time'});

energy=table2array(T(:, 8));force=table2array(T(:,9));

a=energy<0.004;b=force<0.04;both_l=a&b;
accuracy = categorical(both_l,[0 1],{'inaccurate' 'accurate'});
Both_T=addvars(T,accuracy);

start=table2array(R(:, 12));stop=table2array(R(:, 13));
runtime=stop-start;runtime=runtime/60;
time_T=addvars(Both_T,runtime);
 
figure;p_time=parallelplot(time_T,'GroupVariable', 'accuracy','Color', {'[0.7 0.7 0.7]','[0.3 0.6 1]'},'FontSize',16);p_time.LineAlpha = [0.3 0.3];
