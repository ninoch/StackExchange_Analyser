import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dateutil import parser
import pickle
import operator
from datetime import datetime, timedelta
import scipy 
from scipy import stats
import random
from copy import copy

# ************************
# **** MAIN VARIABLES ****
# ************************

files_prefix = "Combined_charts/"
max_session_length = 10
appropriate_session_length = 5 

markers = ['h', 'x', 'v', 'o', '+', 'p', 'D', '2', '8', '*', 'h']
colors = ['blue', 'green', 'red']
# ********************************************
# **** READ DATA FROM FILE TO DRAW CHARTS ****
# ********************************************

def extract_the_data_from_files(path_to_all_files, files_name):
    tmp = [[] for x in range(len(path_to_all_files))]
    for ind in range(len(path_to_all_files)):
        with open(path_to_all_files[ind], 'rb') as file_name:
            tmp[ind] = pickle.load(file_name)

    names = ["acceptance_probability", "words", "urls", "codes"]
    for ind in range(4):
        ys = []
        errs = []
        for the_file in range(len(path_to_all_files)):
            ys.append(tmp[the_file][4 * ind])
            errs.append(tmp[the_file][4 * ind + 1])
        draw_overall_chart(names[ind], files_name, ys, errs)

        ys = []
        errs = []
        for the_file in range(len(path_to_all_files)):
            ys.append(tmp[the_file][4 * ind + 2])
            errs.append(tmp[the_file][4 * ind + 3])
        draw_session_seprated_chart(names[ind], files_name, ys, errs)

# *********************************
# **** CHART DRAWING FUNCTIONS ****
# *********************************

def draw_overall_chart(chart_name, files_name, ys, errs=[]):
    plt.clf()

    for ind in range(len(ys)):
        if errs[ind] == []:
            plt.errorbar(np.arange(1, max_session_length + 1), ys[ind], marker=markers[ind], color=colors[ind], label = files_name[ind])
        else:
            plt.errorbar(np.arange(1, max_session_length + 1), ys[ind], yerr=errs[ind], marker=markers[ind], color=colors[ind], label = files_name[ind])

    plt.xlabel("Session index", fontsize = 16)
    plt.ylabel("Overall " + chart_name, fontsize = 16)
    plt.title("Overall " + chart_name)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize = 16)
    plt.xticks(range(max_session_length + 1))
    plt.savefig(files_prefix + "overall_" + chart_name + ".png") 

def draw_session_seprated_chart(chart_name, files_name, ys, errs=[]):
    plt.clf()

    for ind in range(len(ys)):
        for session_length in range(appropriate_session_length):
            if errs[ind] == []:
                if session_length == 0:
                    plt.errorbar(range(1, session_length + 2), ys[ind][session_length], marker=markers[ind], color=colors[ind], label = files_name[ind])
                else:
                    plt.errorbar(range(1, session_length + 2), ys[ind][session_length], marker=markers[ind], color=colors[ind])
            else:
                if session_length == 0:
                    plt.errorbar(range(1, session_length + 2), ys[ind][session_length], yerr=[errs[ind][session_length], errs[ind][session_length]], marker=markers[ind], color=colors[ind], label = files_name[ind])
                else:
                    plt.errorbar(range(1, session_length + 2), ys[ind][session_length], yerr=[errs[ind][session_length], errs[ind][session_length]], marker=markers[ind], color=colors[ind])

    
    plt.xlabel("Session index", fontsize = 16)
    plt.ylabel(chart_name, fontsize = 16)
    plt.title(chart_name + " in session")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize = 16)
    plt.xticks(range(appropriate_session_length + 1))
    plt.savefig(files_prefix + "in_session_" + chart_name + ".png") 


# *************************************
# **** STARTING OF THE CODE (MAIN) ****
# *************************************

extract_the_data_from_files(["Original_data/all_variables.obj", "Randomized_all_data/all_variables.obj", "Randomized_over_one_user/all_variables.obj"], ["Original", "Randomized", "User_randomized"])
