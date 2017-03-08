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

# ******************************
# **** INPUT LIKE VARIABLES ****
# ******************************

shuffle_or_not = True
number_of_iterations = 5
files_prefix = "Randomized_all_data/"


# ************************
# **** MAIN VARIABLES ****
# ************************

user_actions = dict()
signup_time = dict()
session_seprator = 100 * 60
max_session_length = 10

number_of_sessions = [0] * (max_session_length + 1)
session_accepted_number = [[0 for x in range(max_session_length + 1)] for y in range(max_session_length + 1)]
session_words_number = [[[] for x in range(max_session_length + 1)] for y in range(max_session_length + 1)]
session_links_number = [[[] for x in range(max_session_length + 1)] for y in range(max_session_length + 1)]
session_codes_number = [[[] for x in range(max_session_length + 1)] for y in range(max_session_length + 1)]

all_variables_for_pickle = []

def add_this_for_dump(array):
    global all_variables_for_pickle

    all_variables_for_pickle.append(array)

def save_dumped_variables():
    with open(files_prefix + 'all_variables.obj', 'wb') as file_name:
        pickle.dump(all_variables_for_pickle, file_name)

# *************************************
# **** READING AND PROCESSING DATA ****
# *************************************

class User_action:
    def __init__(self, usr, signup_dur, accepted, words, code_lines, links):
        self.usr = usr
        self.signup_dur = signup_dur
        self.accepted = accepted
        self.words = words
        self.codes = code_lines
        self.links = links 


def preProcess(df, accptFlag):
    OWNER_ID_COL = df.columns.get_loc("OwnerUserId") + 1
    TIME_COL = df.columns.get_loc("CreationDate") + 1
    ANSWER_ID = df.columns.get_loc("Id") + 1
    USER_SIGNUP_DUR = df.columns.get_loc("Seconds_from_SignUp") + 1
    WORDS_COL = df.columns.get_loc("Words") + 1
    LINKS_COL = df.columns.get_loc("Links") + 1
    CODE_COL = df.columns.get_loc("Code_lines") + 1

    if accptFlag:
        ACCPTD_ANSWER_ID = df.columns.get_loc("AcceptedAnswerId") + 1

    for row in df.itertuples():
        owner = row[OWNER_ID_COL]
        time = row[TIME_COL]
        ansId = row[ANSWER_ID]
        words = row[WORDS_COL]
        codes = row[CODE_COL]
        links = row[LINKS_COL]

        if accptFlag:
            accptdAnsId = row[ACCPTD_ANSWER_ID]
        else:
            accptdAnsId = -999 #For data set where ans is not accepted
        
        signup_dur = row[USER_SIGNUP_DUR]
        if not np.isnan(signup_dur) and signup_dur > 0:
            if owner not in signup_time:
                signup_time[owner] = parser.parse(time)-timedelta(seconds=signup_dur)
            
            if ansId == accptdAnsId:
                new_action = User_action(owner, signup_dur, True, words, codes, links)
            else:
                new_action = User_action(owner, signup_dur, False, words, codes, links)
        
            if owner not in user_actions:
                user_actions[owner] = list()

            user_actions[owner].append(new_action)


def read_input():
    # Reading Accepted Data
    print "Reading Accepted ..."
    df = pd.read_csv('Data/AcceptedAnswers_SO.csv').dropna(how="all")
    df = df.drop_duplicates(subset=["OwnerUserId","CreationDate"])
    preProcess(df, True)
    del(df)

    # Reading not Accepted Data
    print "Reading NoAccepted ...."
    df = pd.read_csv('Data/NoAcceptedAnswers_SO.csv').dropna(how="all")
    df = df.drop_duplicates(subset=["OwnerUserId","CreationDate"])
    preProcess(df, False)
    del(df)


# ********************************************
# **** READ DATA FROM FILE TO DRAW CHARTS ****
# ********************************************

def extract_the_data_from_file():
    with open(files_prefix + 'all_variables.obj', 'rb') as file_name:
        tmp = pickle.load(file_name)

    names = ["acceptance_probability_redraw", "words_redraw", "urls_redraw", "codes_redraw"]
    for ind in range(4):
        draw_overall_chart(names[ind], tmp[4 * ind], tmp[4 * ind + 1])
        draw_session_seprated_chart(names[ind], tmp[4 * ind + 2], tmp[4 * ind + 3])

# *********************************
# **** CHART DRAWING FUNCTIONS ****
# *********************************

def draw_overall_chart(chart_name, y, err=[]):
    plt.clf()
    if err == []:
        plt.errorbar(np.arange(1, max_session_length + 1), y, marker='o', label = "Overall " + chart_name)
    else:
        plt.errorbar(np.arange(1, max_session_length + 1), y, yerr=err, marker='o', label = "Overall " + chart_name)

    plt.xlabel("Session index", fontsize = 16)
    plt.ylabel("Overall " + chart_name, fontsize = 16)
    plt.title("Overall " + chart_name)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize = 16)
    plt.xticks(range(max_session_length + 1))
    plt.savefig(files_prefix + "overall_" + chart_name + ".png") 

def draw_session_seprated_chart(chart_name, y, err=[]):
    plt.clf()
    markers = ['h', 'o', 'v', 'x', '+', 'p', 'D', '2', '8', '*', 'h']

    for session_length in range(max_session_length):
        if err == []:
            plt.errorbar(range(1, session_length + 2), y[session_length], marker=markers[session_length], label = "Session length " + str(session_length + 1))
        else:
            plt.errorbar(range(1, session_length + 2), y[session_length], yerr=[err[session_length], err[session_length]], marker=markers[session_length], label = "Session length " + str(session_length + 1))
    
    plt.xlabel("Session index", fontsize = 16)
    plt.ylabel(chart_name, fontsize = 16)
    plt.title(chart_name + " in session")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize = 16)
    plt.xticks(range(max_session_length + 1))
    plt.savefig(files_prefix + "in_session_" + chart_name + ".png") 

# **********************************
# **** EXTRACT DATA FOR DRAWING ****
# **********************************

def extract_acceptance_charts():
    # Acceptance Overall
    overall_accepted_number = get_aggregated_session_index_array(session_accepted_number, False)
    overall_session_number = get_aggregated_session_index_array(np.array([number_of_sessions]*(max_session_length + 1)).transpose(), False)

    y = []
    for session_index in range(max_session_length):
        y.append(float(overall_accepted_number[session_index]) / float(overall_session_number[session_index]))
    
    add_this_for_dump(y)
    add_this_for_dump([])
    draw_overall_chart("acceptance_probability", y)

    # Acceptance for sessions 
    session_acceptance_prob = []
    for session_length in range(1, max_session_length + 1):
        y = []
        for session_index in range(session_length):
            if number_of_sessions[session_length] != 0:
                y.append(float(session_accepted_number[session_length][session_index]) / float(number_of_sessions[session_length]))
        session_acceptance_prob.append(y)
    
    add_this_for_dump(session_acceptance_prob)
    add_this_for_dump([])
    draw_session_seprated_chart("acceptance_probability", session_acceptance_prob)


def extract_overal_charts(the_overal_array, chart_name):
    y = []
    err = []

    for session_index in range(max_session_length):
        m, h = mean_confidence_interval(the_overal_array[session_index])
        y.append(m)
        err.append(h)

    add_this_for_dump(y)
    add_this_for_dump(err)
    draw_overall_chart(chart_name, y, err)

def extract_session_charts(array_2d, chart_name):
    markers = ['', 'o', 'v', 'x', '+', 'p', 'D', '2', '8', '*', 'h']

    session_y = []
    session_err = []

    for session_length in range(1, max_session_length + 1):
        y = []
        err = []

        for session_index in range(session_length):
            m, mf= mean_confidence_interval(array_2d[session_length][session_index])
            y.append(m)
            err.append(mf)

        session_y.append(y)
        session_err.append(err)

    add_this_for_dump(session_y)
    add_this_for_dump(session_err)
    draw_session_seprated_chart(chart_name, session_y, session_err)

# *****************************
# **** SECONDARY FUNCTIONS ****
# *****************************

def get_aggregated_session_index_array(array_2d, append=True):
    if append == True:
        tmp = [[] for x in range(max_session_length)]
        for session_length in range(1, max_session_length + 1):
            for session_index in range(session_length):
                tmp[session_index].extend(array_2d[session_length][session_index])
    else:
        tmp = [0] * max_session_length
        for session_length in range(1, max_session_length + 1):
            for session_index in range(session_length):
                tmp[session_index] += array_2d[session_length][session_index]
    return tmp

def randomize_all_the_data():
    global user_actions
    print "Randomizing ..."
    tmp = []
    for usr, actions in user_actions.items():
        for action in actions:
            tmp.append(copy(action))

    indexes = [x for x in range(len(tmp))]
    random.shuffle(indexes)

    ind = 0
    for usr, actions in user_actions.items():
        for t in range(len(actions)):
            user_actions[usr][t].accepted = tmp[indexes[ind]].accepted
            user_actions[usr][t].words = tmp[indexes[ind]].words
            user_actions[usr][t].codes = tmp[indexes[ind]].codes
            user_actions[usr][t].links = tmp[indexes[ind]].links
            ind += 1


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def compute_user_sessions(actions):
    global session_words_number, session_accepted_number, session_codes_number, session_links_number, number_of_sessions

    cnt = 1
    if len(actions) > 1:
        for ind in range(1, len(actions)):
            if actions[ind].signup_dur - actions[ind - 1].signup_dur > session_seprator:
                if cnt <= max_session_length:
                    number_of_sessions[cnt] += 1
                    for indy in range(cnt):
                        if actions[ind - cnt + indy].accepted == True:
                            session_accepted_number[cnt][indy] += 1

                        session_words_number[cnt][indy].append(actions[ind - cnt + indy].words)
                        session_codes_number[cnt][indy].append(actions[ind - cnt + indy].codes)
                        session_links_number[cnt][indy].append(actions[ind - cnt + indy].links)
                cnt = 1
            else:
                cnt += 1
    else:
        number_of_sessions[1] += 1
        if actions[0].accepted == True:
            session_accepted_number[1][0] += 1

        session_words_number[1][0].append(actions[0].words)
        session_codes_number[1][0].append(actions[0].codes)
        session_links_number[1][0].append(actions[0].links)

# ************************
# **** MAIN FUNCTIONS ****
# ************************

def iterate_through_users():
    for usr, actions in user_actions.items():
        actions.sort(key=operator.attrgetter('signup_dur'))

    for rtimes in range(number_of_iterations):
        if shuffle_or_not:
            randomize_all_the_data()

        for usr, actions in user_actions.items():
            compute_user_sessions(actions)

def draw_all_charts():
    extract_acceptance_charts()

    extract_overal_charts(get_aggregated_session_index_array(session_words_number), "words")
    extract_session_charts(session_words_number, "words")

    extract_overal_charts(get_aggregated_session_index_array(session_links_number), "urls")
    extract_session_charts(session_links_number, "urls")

    extract_overal_charts(get_aggregated_session_index_array(session_codes_number), "code_lines")
    extract_session_charts(session_codes_number, "codes")


# *************************************
# **** STARTING OF THE CODE (MAIN) ****
# *************************************

read_input()
iterate_through_users()
draw_all_charts()
save_dumped_variables()
#extract_the_data_from_file()
