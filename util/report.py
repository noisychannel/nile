#!/usr/bin/env python

import sys
import os
from tabulate import tabulate
import numpy as np
import pprint as pp

log_dir = sys.argv[1]

def analysis(all_res, total_iterations):
    mdata = []
    for exp_res, exp_id in all_res:
        if sum([len(l) for l in exp_res]) != 0:
            if len(exp_res) != 0 and sum([len(l) for l in exp_res]) != 0:
                best_run = sorted(range(len(exp_res)), key=lambda i: exp_res[i][2], reverse=True)[0]
                dev_res = [x[2] for x in exp_res]
                it_string = str(total_iterations[exp_id][1]) + "-" + str(total_iterations[exp_id][0])
                mdata.append([exp_id, best_run + 1, exp_res[best_run][0], it_string, exp_res[best_run][1], exp_res[best_run][2], '-', exp_res[best_run][3], round(np.mean(dev_res), 4), round(np.std(dev_res), 4)])
            else:
                mdata.append(['-' for _ in range(9)])
    print tabulate(mdata, headers=['Exp ID', 'Best Run', 'Best iter', 'Iterations completed', 'Best tune', 'Best dev', 'Best test', 'Gain (dev)', 'Mean (dev)', 'SD (dev)'])

def process_err_file(err_file, baseline_dev):
    best_dev = []
    iter_index = 0
    iter_score = 0
    e_file = open(err_file)
    for line in e_file:
        if "EBLEU" in line:
            line_iter = line.strip().split(" ")[1:]
            iter_index = int(line_iter[0])
            iter_score = round(float(line_iter[len(line_iter) -1][:-1]), 4)
        if "best" in line and "Dev" in line:
            score = float(line.strip().split(" ")[2])
            best_dev = [iter_index, iter_score, round(score, 4), round((score - baseline_dev), 4)]
    return iter_index, best_dev

all_results = []
total_iterations = {}
for exp in range(1, 25):
    for postfix in ["", "_1", "_2"]:
        exp_results = []
        exp_total_iterations = []
        for run in range(1, 11):
            if exp < 9:
                err_file = log_dir + "/medium." + str(exp) + postfix + "." + str(run) + ".err"
                baseline_dev = 0.246188852765
            elif exp >=9 and exp < 17:
                err_file = log_dir + "/full." + str(exp) + postfix + "." + str(run) + ".err"
                baseline_dev = 0.231397522122
            else:
                err_file = log_dir + "/zh." + str(exp) + postfix + "." + str(run) + ".err"
                baseline_dev = 0.186694463872
            if (not os.path.isfile(err_file)):
                break
            total_iter, best_dev = process_err_file(err_file, baseline_dev)
            exp_total_iterations.append(total_iter)
            exp_results.append(best_dev)
        if len(exp_results) > 0:
            all_results.append((exp_results, str(exp) + postfix))
            total_iterations[str(exp) + postfix] = (np.max(exp_total_iterations), np.min(exp_total_iterations))

analysis(all_results, total_iterations)
