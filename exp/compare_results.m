clear;clc;
tin_result_path = 'output/results/eval_result_def_tin.mat';
my_result_path = 'output/results/eval_result_def_spamap_0.6.mat';
load(tin_result_path);
tin_ap = AP;
tin_rec = REC;
load(my_result_path);
my_ap = AP;
my_rec = REC;
clear AP;
clear REC;

com_ap = cat(2, my_ap, tin_ap);
com_rec = cat(2, my_rec, tin_rec);