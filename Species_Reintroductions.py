def CASE_TO_SOLVE_init(CASE_TO_SOLVE):
    global DATA_TYPE
    DATA_TYPE = CASE_TO_SOLVE

def get_CASE():
    print(DATA_TYPE)
#############################################################
"""
Importing the Libraries
"""
#############################################################
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.integrate import odeint
import operator
import itertools
import sys
import re
import pickle
import time
import datetime
import gzip, pickle, pickletools
import os
import glob

import joblib
from joblib import Parallel, delayed
from joblib import parallel_backend
from tqdm import tqdm

#############################################################
"""
Baseline Methods
"""
#############################################################
def getProjectedNet(M, flag = [True, True]):
    #M = M_df.copy(deep = True)
    n,m = M.shape
    M_Empty = {(r,c) for r in range(n) for c in range(m) if M.iloc[r,c] == 0}
    if flag[0]:
        A = np.zeros((M.shape[0],M.shape[0]))
        kn = np.sum(M, axis = 0)
        iter_List_1 = ((i,j,k) \
                     for i in range(m) \
                         for j in range(n-1) \
                             for k in range(j+1,n) \
                                 if M_Empty.isdisjoint({(j,i),(k,i)}))
        for (i,j,k) in iter_List_1:
            Term = M.iloc[j,i]/kn[i]+M.iloc[k,i]/kn[i]
            A[(j,k),(k,j)] += Term
        A_df = pd.DataFrame(A)
        A_df.columns = M.index
        A_df.index = M.index
        #A = np.array(A)
    else:
        A_df = None
        
    if flag[1]:
        B = np.zeros((M.shape[1],M.shape[1]))
        km = np.sum(M, axis = 1)
        iter_List_0 = ((i,j,k) \
                     for i in range(n) \
                         for j in range(m-1) \
                             for k in range(j+1,m) \
                                 if M_Empty.isdisjoint({(i,j),(i,k)}))
        for (i,j,k) in iter_List_0:
            Term = M.iloc[i,j]/km[i]+M.iloc[i,k]/km[i]
            B[(j,k),(k,j)] += Term
        B_df = pd.DataFrame(B)
        B_df.columns = M.columns
        B_df.index = M.columns
        #B = np.array(B)
    else:
        B_df = None
    
    return A_df, B_df

def getAllNetDefinitions(M_df,A_df,B_df, flag = [True, True, True]):
    n,m = M_df.shape
  
    if flag[1]:
        A = np.array(A_df)
        NetA = nx.Graph()
        #print("Net A")
        #NetA.add_nodes_from(np.array(range(self.n)))
        NetA.add_nodes_from(M_df.index)
        for i in range(n):
            for j in range(i+1,n):
                if A[i,j] != 0:
                    NetA.add_edge(M_df.index[i],M_df.index[j],weight = A[i,j])
                    #print('Edge:',M_df.index[i],M_df.index[j],";Weight:",A[i,j])
    else:
        NetA = None

    if flag[2]:
        B = np.array(B_df)
        NetB = nx.Graph()
        #print("Net B")
        #NetB.add_nodes_from(np.array(range(self.m)))
        NetB.add_nodes_from(M_df.columns)
        for i in range(m):
            for j in range(i+1,m):
                if B[i,j] != 0:
                    NetB.add_edge(M_df.columns[i],M_df.columns[j],weight = B[i,j])
                    #print('Edge:',M_df.columns[i],M_df.columns[j],";Weight:",B[i,j])
    else:
        NetB = None

    if flag[0]:
        M = np.array(M_df)
        NetM = nx.Graph()
        NetM.add_nodes_from(M_df.columns)
        NetM.add_nodes_from(M_df.index)
        for i in range(n):
            for j in range(m):
                if M[i,j] != 0:
                    NetM.add_edge(M_df.index[i],M_df.columns[j])
    else:
        NetM = None
    
    return NetM, NetA, NetB

def getNetDefinitions(M_df):
    n,m = M_df.shape
    M = np.array(M_df)
    NetM = nx.Graph()
    NetM.add_nodes_from(M_df.columns)
    NetM.add_nodes_from(M_df.index)
    for i in range(n):
        for j in range(m):
            if M[i,j] != 0:
                NetM.add_edge(M_df.index[i],M_df.columns[j])    
    return NetM

#############################################################
"""
EOM Methods
"""
#############################################################
def EOM1D(x, t, A0):
    K = 5
    B = 0.1
    C = 1
    D = 5
    E = 0.9
    H = 0.1
    term1 = B + x*(1 - (x/K))*((x/C) - 1)
    n = A0.shape[0]
    term2 = []
    for i in range(len(x)):
        temp = 0
        for j in range(n):
            if A0[i,j] != 0:
                temp+=A0[i,j]*x[i]*x[j]/(D+E*x[i]+H*x[j])
        term2.append(temp)
    return term1 + np.array(term2)

def EOM2D(x, t, M0):
    alpha = 0.3
    h = 0.7
    mu = 1e-4
    gamma0 = 1.0
    p = 0.5
    bii = 1
    #bij = 0.01
    bij = 0
    
    n,m = M0.shape
    D = np.concatenate((np.sum(np.array(M0), axis = 1),np.sum(np.array(M0), axis = 0)), axis = None)
    term1 = mu + x*alpha
    term2 = []
    term3 = []
    for i in range(n):
        temp1 = 0
        temp2 = 0
        for j in range(n):
            if i != j:
                temp1 += bij*x[j]
            elif i == j:
                temp1 += bii*x[j]
        term2.append((-1.0)*temp1*x[i])
        for k in range(m):
            if M0.iloc[i,k]:
                temp2 += (gamma0 / ((D[i])**p))*x[n+k]
        term3.append((temp2 / (1 + h*temp2))*x[i])
    for i in range(m):
        temp1 = 0
        temp2 = 0
        for j in range(m):
            if i != j:
                temp1 += bij*x[n+j]
            elif i == j:
                temp1 += bii*x[n+j]
        term2.append((-1.0)*temp1*x[n+i])
        for k in range(n):
            if M0.iloc[k,i]:
                temp2 += (gamma0 / ((D[i+n])**p))*x[k]
        term3.append((temp2 / (1 + h*temp2))*x[i+n])
    return term1 + np.array(term2) + np.array(term3)

def ODE_Solve_Low(A,t0=0,tf=100,num=50,x_low=10e-3, specify_model = None):
    t = np.linspace(t0,tf,num)
    n,m = A.shape
    if specify_model == None:
        if DATA_TYPE.split('_')[1] == '1D':
            x0 = np.ones(n)*x_low
            xl = odeint(EOM1D,x0,t,args=(A,))
            xl_ss = xl[-1,:]
            xl_ss_T = xl_ss.T
            beta_eff, x_eff = betaspace(xl_ss_T,A)
            return beta_eff, x_eff, xl
        elif DATA_TYPE.split('_')[1] == '2D':
            x0 = np.ones(n+m)*x_low
            xl = odeint(EOM2D,x0,t,args=(A,))
            return xl, xl[:,0:n], xl[:,-m:]
    else:
        if specify_model == '1D':
            x0 = np.ones(n)*x_low
            xl = odeint(EOM1D,x0,t,args=(A,))
            xl_ss = xl[-1,:]
            xl_ss_T = xl_ss.T
            beta_eff, x_eff = betaspace(xl_ss_T,A)
            return beta_eff, x_eff, xl
        elif specify_model == '2D':
            x0 = np.ones(n+m)*x_low
            xl = odeint(EOM2D,x0,t,args=(A,))
            return xl, xl[:,0:n], xl[:,-m:]

def betaspace(x_ss_T,A):
    sAout = sum(A)
    x_nss = np.dot(A,x_ss_T)
    if sum(sum(A)) == 0:
        beta_eff = 0
        x_eff = 0
    else:
        beta_eff = sum(sum(A*A))/sum(sum(A))
        x_eff = sum(x_nss)/sum(sAout)
    return beta_eff, x_eff

#############################################################
"""
Get Resilience and Persistence
"""
#############################################################
def getResilience(xl):
    tol = 10e-6
    patience = 0
    for i in range(1,xl.shape[0]):
        data_prev = xl[i-1,:]
        data_temp = xl[i,:]
        data_diff = data_temp - data_prev
        if np.mean(data_diff) < tol:
            patience += 1
            if patience == 5: return i
    return i

def getPersistence(Active_Species):
    Active_Species = Active_Species[Active_Species != 0]
    return len(Active_Species)

def isExtinct(value, threshold = 0.0, tol = 10e-6):
    if DATA_TYPE.split('_')[1] == '1D':
        threshold = 0.11577398
    elif DATA_TYPE.split('_')[1] == '2D':
        threshold = 0.3
    if abs(value) < threshold:
        return 0
    else:
        return value

#############################################################
"""
Perturbation Methods
"""
#############################################################
def RandomRemoval(M_df,f,axis=0,seed = 0):
    np.random.seed(seed)
    if axis == 0:
        Node_List = M_df.index
        n_del = int(len(Node_List)*f)
        Node_del = np.random.choice(Node_List, size = (n_del,), replace = False)
        M_df = M_df.drop(labels = Node_del, axis = axis)
        #M_df = M_df.loc[(M_df!=0).any(axis=1)] #Cleaning Row isoloates
    elif axis == 1:
        Node_List = M_df.columns
        n_del = int(len(Node_List)*f)
        Node_del = np.random.choice(Node_List, size = (n_del,), replace = False)
        M_df = M_df.drop(labels = Node_del, axis = axis)
        #M_df = M_df.loc[:, (M_df!=0).any(axis=0)] #Cleaning Column isoloates
    else:
        M_df = RandomRemoval(M_df,f/2.0,axis=0,seed = seed)
        M_df = RandomRemoval(M_df,f/2.0,axis=1,seed = seed)
        #M_df = M_df.loc[(M_df!=0).any(axis=1)] #Cleaning Row isoloates
        #M_df = M_df.loc[:, (M_df!=0).any(axis=0)] #Cleaning Column isoloates
    return M_df

def TargetedRemoval(M_df,f,axis=0, strategy = True, seed = 0):
    if axis == 0:
        np.random.seed(seed)
        Node_List = M_df.index
        n_del = int(len(Node_List)*f)
        if strategy == True:
            degree_list = np.sum(M_df, axis = 1) / np.sum((np.sum(M_df, axis = 1)))
        elif strategy == False:
            degree_list = max(np.sum(np.array(M_df), axis = 1)) - np.sum(np.array(M_df), axis = 1)
            degree_list = degree_list / np.sum(degree_list)
        Node_del = np.random.choice(Node_List, size = (n_del,), replace = False, p = degree_list)
        M_df = M_df.drop(labels = Node_del, axis = axis)
        #M_df = M_df.loc[(M_df!=0).any(axis=1)] #Cleaning Row isoloates
    elif axis == 1:
        np.random.seed(seed)
        Node_List = M_df.columns
        n_del = int(len(Node_List)*f)
        if strategy == True:
            degree_list = np.sum(M_df, axis = 0) / np.sum((np.sum(M_df, axis = 0)))
        elif strategy == False:
            degree_list = max(np.sum(np.array(M_df), axis = 0)) - np.sum(np.array(M_df), axis = 0)
            degree_list = degree_list / np.sum(degree_list)
        Node_del = np.random.choice(Node_List, size = (n_del,), replace = False, p = degree_list)
        M_df = M_df.drop(labels = Node_del, axis = axis)
        #M_df = M_df.loc[:, (M_df!=0).any(axis=0)] #Cleaning Column isoloates
    else:
        M_df = TargetedRemoval(M_df,f/2.0,axis=0,strategy = strategy, seed = seed)
        M_df = TargetedRemoval(M_df,f/2.0,axis=1,strategy = strategy, seed = seed)
        #M_df = M_df.loc[(M_df!=0).any(axis=1)] #Cleaning Row isoloates
        #M_df = M_df.loc[:, (M_df!=0).any(axis=0)] #Cleaning Column isoloates
    return M_df

def getProspectiveSpecies(M_per, M_ori,axis = 0):
    if axis == 0:
        return np.setdiff1d(M_ori.index, M_per.index)
    elif (axis == 1):
        return np.setdiff1d(M_ori.columns, M_per.columns)
    else:
        arr1 = np.setdiff1d(M_ori.index, M_per.index)
        arr2 = np.setdiff1d(M_ori.columns, M_per.columns)
        return np.union1d(arr1,arr2)

def Perturbation(M_df,flag,f,axis,seed):
    if flag == 0:
        return RandomRemoval(M_df,f=f,axis=axis,seed=seed)
    elif flag == 1:
        return TargetedRemoval(M_df,f=f,axis=axis,strategy = True,seed=seed)
    elif flag == 2:
        return TargetedRemoval(M_df,f=f,axis=axis,strategy = False,seed=seed)

def PerturbationEnsembling(M_df,i,j,k,N=100, Display_Output = True):
    collection = []
    cnt = 0
    iter_cnt = 0
    while len(collection) < N:
        try:
            M_per = Perturbation(M_df, flag = i, f = 0.1*(j+1), axis = k, seed = cnt)
            Node_del = set(getProspectiveSpecies(M_per, M_df,axis = k))
            if Node_del in collection:
                #print(cnt,": ***Repeat***")
                pass
            else:
                collection.append(Node_del)
                #print(cnt,": Unique")
            cnt += 1
        except:
            cnt += 1
        if iter_cnt >= 10*N:
            if Display_Output:print('Break: ', len(collection))
            break
        iter_cnt += 1
    if Display_Output:print('Success: ', len(collection))
    collection = [np.array(list(x)) for x in collection]
    return collection

def RemovalFromNodeList(Node_del,M_df,axis=0):
    if axis == 0:
        M_df = M_df.drop(labels = Node_del, axis = axis)
        #M_df = M_df.loc[(M_df!=0).any(axis=1)] #Cleaning Row isoloates
    elif axis == 1:
        M_df = M_df.drop(labels = Node_del, axis = axis)
        #M_df = M_df.loc[:, (M_df!=0).any(axis=0)] #Cleaning Column isoloates
    else:
        Node_del_row = []
        Node_del_col = []
        for i in Node_del:
            if i[0] == M_df.index[0][0]:
                Node_del_row.append(i)
            elif i[0] == M_df.columns[0][0]:
                Node_del_col.append(i)
        M_df = RemovalFromNodeList(Node_del_row,M_df,axis=0)
        M_df = RemovalFromNodeList(Node_del_col,M_df,axis=1)
        #M_df = M_df.loc[(M_df!=0).any(axis=1)] #Cleaning Row isoloates
        #M_df = M_df.loc[:, (M_df!=0).any(axis=0)] #Cleaning Column isoloates

    return M_df

#############################################################
"""
Data Pickling and Time
"""
#############################################################
def PickleObj(obj, filepath):
    with gzip.open(filepath, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)

def UnpickleObj(filepath):
    with gzip.open(filepath, 'rb') as f:
        p = pickle.Unpickler(f)
        obj = p.load()
    return obj

def tic(): 
    return time.time()

def tac(t_start):
    t_sec = round(time.time() - t_start)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    return 'Time passed: {} hour: {} min: {} sec'.format(t_hour,t_min,t_sec)

#############################################################
"""
File Reader Class
"""
#############################################################
class Write_Log():
    def __init__ (self, Filename, timestamp = False, Mode = 'a'):
        self.__Path = Filename
        #self.Mode = Mode
        self.__file_object = open(self.__Path, Mode)
        if timestamp:
            self.__file_object.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
            self.__file_object.flush()
        #self.close_Log()
            
    def close_Log(self):
        self.__file_object.close()
    
    def Open_Log(self, timestamp = False, Mode = 'a'):
        self.__file_object = open(self.__Path, Mode)
        if timestamp:
            self.__file_object.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
            self.__file_object.flush()
    
    def Log_Entry(self, data):
        #self.__file_object = open(self.__Path, self.Mode)
        self.__file_object.write(str(data) + '\n')
        self.__file_object.flush()
        #self.close_Log()

#############################################################
"""
Segment 01
"""
#############################################################
def getWebOfLifeData(NetworkName, IncludeSpecies = 'yes'):
    url = "http://www.web-of-life.es/download/"+NetworkName+"_"+IncludeSpecies+".csv"
    dataframe = pd.read_csv(url,index_col=0)
    if NetworkName == 'M_PL_063':
        p = re.compile('Number|Frequency')
        s = list(dataframe.columns)
        s.extend(list(dataframe.index))
        Node_del = [ x for x in s if p.match(x) ]
        for label in Node_del:
            for ax in [0,1]:
                try:dataframe = dataframe.drop(labels = label, axis = ax)
                except:pass
    dataframe[dataframe > 0] = 1
    return dataframe

def Segment_01(NetworkName,SpeciesName = True,File_Obj = None):
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_one.pkl")
    try:
        M_df = UnpickleObj(FileName)
        if File_Obj != None:
            File_Obj.Log_Entry("{}: Exists in Storage".format(NetworkName))
    except:
        M_df = getWebOfLifeData(NetworkName)
        n,m = M_df.shape
        
        m_Encoding = np.array(M_df.columns)
        M_df.columns = np.char.add('B',np.array(range(len(M_df.columns)), dtype = 'str'))
        n_Encoding = np.array(M_df.index)
        M_df.index = np.char.add('A',np.array(range(len(M_df.index)), dtype = 'str'))

        os.makedirs(os.path.dirname(FileName), exist_ok=True)
        PickleObj(M_df, FileName)

#############################################################
"""
Segment 02
"""
#############################################################
def Segment_02(NetworkName, Ensembles = 100, Fraction = range(9), Overwrite = True, Display_Output = True, Display_Count = False, File_Obj = None):
    t_seg02 = tic()
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_two.pkl")
    try:
        data_two = UnpickleObj(FileName)
        it_exists = True
    except:
        it_exists = False
    if Overwrite or (not it_exists):
        M_df = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_one.pkl"))
        data_two = {}
    
        for i in range(3): # Strategy Selector
            for j in Fraction: # Fraction Selector
                for k in range(3): # Target Selector
                    if Display_Output:print((i,j,k))
                    data_two[i,j,k] = PerturbationEnsembling(M_df,i,j,k,N=Ensembles,Display_Output = Display_Output)

        os.makedirs(os.path.dirname(FileName), exist_ok=True)
        PickleObj(data_two, FileName)
        #print(tac(t_seg02))
        if Display_Output:
            data_two_pkl = UnpickleObj(FileName)
            assert (data_two.keys() == data_two_pkl.keys())
            bool_arr = []
            for key in data_two.keys():
                #print(key)
                for item in range(len(data_two[key])):
                    bool_arr.append(all(data_two[key][item] == data_two_pkl[key][item]))
            print(any(bool_arr))
    
    cnt = 0
    for key in data_two.keys():
        cnt += len(data_two[key])
    File_Obj.Log_Entry("{}: {} Combinations; {}".format(NetworkName, cnt, tac(t_seg02)))
    if Display_Count:print("Number of Combinations: ", cnt)

#############################################################
"""
Get Overlap in Ensembles
"""
#############################################################
def getOverlap(NetworkName, data = None):
    if data == None:
        os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_two.pkl")
        data_two = UnpickleObj(FileName)
    else:
        data_two = data
    Overlap = {}
    cnt = 0
    for key in data_two.keys():
        i,j,k = key
        Overlap[key] = {}
        for ensembles_outer in range(len(data_two[key])):
            Overlap[key][ensembles_outer] = []
            for strategy in range(3):
                if i == strategy:
                    pass
                else:
                    for ensembles_inner in range(len(data_two[(strategy,j,k)])):
                        if set(data_two[key][ensembles_outer]) == set(data_two[(strategy,j,k)][ensembles_inner]):
                            Overlap[key][ensembles_outer].append([(strategy,j,k),ensembles_inner])
                            cnt+=1
    print('Overlapping Ensembles found: ', cnt)
    return Overlap

#############################################################
"""
Data Preparation
"""
#############################################################
def get_WebOfLife_ID():
    WebOfLife_ID = ["M_PL_0{}".format(ID) if ID > 9 else "M_PL_00{}".format(ID) for ID in range(1,73) if ID not in [60,61,69,72]]        
    #060: 01 - 24
    WebOfLife_ID.extend(["M_PL_060_{}".format(ID) if ID > 9 else "M_PL_060_0{}".format(ID) for ID in range(1,25)])        
    #061: 01 - 48
    WebOfLife_ID.extend(["M_PL_061_{}".format(ID) if ID > 9 else "M_PL_061_0{}".format(ID) for ID in range(1,49)])        
    #069: 01 - 03
    WebOfLife_ID.extend(["M_PL_069_0{}".format(ID) for ID in range(1,4)])        
    #072: 01 - 05
    WebOfLife_ID.extend(["M_PL_072_0{}".format(ID) for ID in range(1,6)])        
    WebOfLife_ID.sort()
    return WebOfLife_ID

def sort_NetworkName():
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"Network_Solved_{DATA_TYPE.split('_')[0]}.pkl")
    DF = UnpickleObj(FileName)
    return list(DF.NetworkName)

def get_Segment01(WebOfLife_ID = get_WebOfLife_ID(), param_Segment01 = [False, 0, None, 1], File_Obj = None, Record_Obj = None):
    param_Reset, Start, End, Step = param_Segment01  
    if File_Obj != None:
        File_Obj.Log_Entry(".......Segment 01.........")
        File_Obj.Log_Entry("Reset: {}".format(param_Reset))  
    for NetworkName in itertools.islice(WebOfLife_ID,Start,End,Step):
        Segment_01(
            NetworkName = NetworkName,
            File_Obj = File_Obj
            );

def get_Segment02(WebOfLife_ID = get_WebOfLife_ID(), param_Segment02 = [100, np.arange(9), 0, None, 1, True, False, False], File_Obj = None, Record_Obj = None):
    param_Ensembles = param_Segment02[0]
    param_Fraction = param_Segment02[1]
        
    Start, End, Step = param_Segment02[2:5]
    Overwrite, Display_Output, Display_Count = param_Segment02[5:8]
        
    File_Obj.Log_Entry(".......Segment 02.........")
    File_Obj.Log_Entry("Ensembles: {}".format(param_Ensembles))
    File_Obj.Log_Entry("Fraction: {}".format(param_Fraction))
        
    for NetworkName in itertools.islice(WebOfLife_ID, Start, End, Step):
        Segment_02(
            NetworkName,
            Ensembles = param_Ensembles,
            Fraction = param_Fraction,
            Overwrite = Overwrite,
            Display_Output = Display_Output,
            Display_Count = Display_Count,
            File_Obj = File_Obj
            )

def Data_Preparation(param_Segment01, param_Segment02, File_Obj = None, Record_Obj = None, Record_Reset = False):
    #WebOfLife_ID = get_WebOfLife_ID()
    WebOfLife_ID = sort_NetworkName()
    if param_Segment01 != None:
        get_Segment01(
            WebOfLife_ID = WebOfLife_ID,
            param_Segment01 = param_Segment01, 
            File_Obj = File_Obj, 
            Record_Obj = Record_Obj
            )
    get_Segment02(
        WebOfLife_ID = WebOfLife_ID, 
        param_Segment02 = param_Segment02, 
        File_Obj = File_Obj
        )
    #return WebOfLife_ID

#############################################################
"""
Species Restoration Methods
"""
#############################################################
def System_Driven_Restoration(NetworkName, key, ensembles_ID, k, M_df, Node_del, Overlap, Overwrite = False):
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_three_{key}_{ensembles_ID}_{k}.pkl")
    if not Overwrite:
        try:
            data = UnpickleObj(FileName)
            #os.makedirs(os.path.dirname(FileName), exist_ok=True)
            #PickleObj(data, FileName)
            print(FileName, ':Overwrite:')
            return data
        except:
            pass
    if len(Overlap[key][ensembles_ID]) != 0:
        for Overlap_index in range(len(Overlap[key][ensembles_ID])):
            key_overlap = Overlap[key][ensembles_ID][Overlap_index][0]
            ensemblesID_overlap = Overlap[key][ensembles_ID][Overlap_index][1]
            try:
                FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_three_{key_overlap}_{ensemblesID_overlap}_{k}.pkl")
                data = UnpickleObj(FileName)
                FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_three_{key}_{ensembles_ID}_{k}.pkl")
                os.makedirs(os.path.dirname(FileName), exist_ok=True)
                PickleObj(data, FileName)
                print(FileName, ':Overlap:', [key_overlap, ensemblesID_overlap, k])
                return data
            except:
                pass

    t_sys_res = tic()
    M_per = RemovalFromNodeList(Node_del,M_df,axis=key[2])
    
    Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
    np.random.shuffle(Nodes_to_restore)
    if k == 0:
        A_per, _ = getProjectedNet(M_per,[True, False])
        beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(A_per))
    if k == 1:
        _, B_per = getProjectedNet(M_per,[False, True])
        beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(B_per))
    restoration_sequence = []
    beta_eff_log = [beta_eff]
    x_eff_log = [x_eff]
    xl_log = [xl]
            
    for p in range(len(Nodes_to_restore)):
        beta_diff = np.zeros((len(Nodes_to_restore),))
        for q in range(len(Nodes_to_restore)):
            M_per_temp = M_per.copy(deep = True)
            if k == 0:M_per_temp = M_per_temp.append(M_df.loc[Nodes_to_restore[q]])
            if k == 1:M_per_temp[Nodes_to_restore[q]] = M_df[Nodes_to_restore[q]]
            M_per_temp[M_per_temp.isnull()] = M_df
            if k == 0:
                A_per_temp, _ = getProjectedNet(M_per_temp, [True, False])
                beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=50,x_low=0,A=np.array(A_per_temp))
            if k == 1:
                _, B_per_temp = getProjectedNet(M_per_temp,[False, True])
                beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=50,x_low=0,A=np.array(B_per_temp))
            beta_diff[q] = beta_eff
        Node_choice = Nodes_to_restore[np.argmax(beta_diff)]
        if k == 0:M_per = M_per.append(M_df.loc[Node_choice])
        if k == 1:M_per[Node_choice] = M_df[Node_choice]
        M_per[M_per.isnull()] = M_df
        A_per, B_per = getProjectedNet(M_per)
        if k == 0:
            A_per, _ = getProjectedNet(M_per,[True, False])
            beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(A_per))
        if k == 1:
            _, B_per = getProjectedNet(M_per,[False, True])
            beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(B_per))
        restoration_sequence.append(Node_choice)
        beta_eff_log.append(beta_eff)
        x_eff_log.append(x_eff)
        xl_log.append(xl)
        Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
        np.random.shuffle(Nodes_to_restore)
        
    data = [restoration_sequence, beta_eff_log, x_eff_log, xl_log]
    
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_three_{key}_{ensembles_ID}_{k}.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    
    PickleObj(data, FileName)
    
    print(FileName,".....",tac(t_sys_res))
    
    return data

def Centrality_Driven_Restoration(NetworkName, key, ensembles_ID, k, Centrality, M_df, Node_del, Overlap, Overwrite = False):
    if not Overwrite:
        try:
            if Centrality == 'degree':
                FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_four_{key}_{ensembles_ID}_{k}.pkl")
            if Centrality == 'closeness':
                FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_five_{key}_{ensembles_ID}_{k}.pkl")
            if Centrality == 'betweenness':
                FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_six_{key}_{ensembles_ID}_{k}.pkl")
            data = UnpickleObj(FileName)
            return data
        except:
            pass
    if len(Overlap[key][ensembles_ID]) != 0:
        for Overlap_index in range(len(Overlap[key][ensembles_ID])):
            key_overlap = Overlap[key][ensembles_ID][Overlap_index][0]
            ensemblesID_overlap = Overlap[key][ensembles_ID][Overlap_index][1]
            try:
                if Centrality == 'degree':
                    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_four_{key_overlap}_{ensemblesID_overlap}_{k}.pkl")
                    data = UnpickleObj(FileName)
                    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_four_{key}_{ensembles_ID}_{k}.pkl")
                if Centrality == 'closeness':
                    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_five_{key_overlap}_{ensemblesID_overlap}_{k}.pkl")
                    data = UnpickleObj(FileName)
                    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_five_{key}_{ensembles_ID}_{k}.pkl")
                if Centrality == 'betweenness':
                    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_six_{key_overlap}_{ensemblesID_overlap}_{k}.pkl")
                    data = UnpickleObj(FileName)
                    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_six_{key}_{ensembles_ID}_{k}.pkl")
                os.makedirs(os.path.dirname(FileName), exist_ok=True)
                PickleObj(data, FileName)
                print(FileName, ':', [key_overlap, ensemblesID_overlap, k])
                return data
            except:
                pass
    t_net_res = tic()
    M_per = RemovalFromNodeList(Node_del,M_df,axis=key[2])

    Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
    np.random.shuffle(Nodes_to_restore)
    
    if DATA_TYPE.split('_')[1] == '1D':
        if k == 0:
            A_per, _ = getProjectedNet(M_per, [True, False])
            #NetM_per, NetA_per, NetB_per = getNetDefinitions(M_per,A_per,B_per)
            beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(A_per))
        if k == 1:
            _, B_per = getProjectedNet(M_per, [False, True])
            #NetM_per, NetA_per, NetB_per = getNetDefinitions(M_per,A_per,B_per)
            beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(B_per))
        restoration_sequence = []
        beta_eff_log = [beta_eff]
        x_eff_log = [x_eff]
        xl_log = [xl]

        for p in range(len(Nodes_to_restore)):
            #print(p,":",Nodes_to_restore)
            centrality_restoration = {}
            for q in range(len(Nodes_to_restore)):
                M_per_temp = M_per.copy(deep = True)
                
                if k == 0:
                    M_per_temp = M_per_temp.append(M_df.loc[Nodes_to_restore[q]])
                if k == 1:
                    M_per_temp[Nodes_to_restore[q]] = M_df[Nodes_to_restore[q]]
                
                M_per_temp[M_per_temp.isnull()] = M_df
                #A_per_temp, B_per_temp = getProjectedNet(M_per_temp)
                NetM_per_temp, _, _ = getNetDefinitions(M_per_temp,None,None,[True, False, False])
                
                if Centrality == 'degree':
                    centrality_dict = nx.degree_centrality(NetM_per_temp)
                if Centrality == 'closeness':
                    centrality_dict = nx.closeness_centrality(NetM_per_temp)
                if Centrality == 'betweenness':
                    centrality_dict = nx.betweenness_centrality(NetM_per_temp)
                
                centrality_restoration[Nodes_to_restore[q]] = centrality_dict[Nodes_to_restore[q]]
            
            Node_choice = max(centrality_restoration, key = centrality_restoration.get)
            
            if k == 0:
                M_per = M_per.append(M_df.loc[Node_choice])
            if k == 1:
                M_per[Node_choice] = M_df[Node_choice]
            M_per[M_per.isnull()] = M_df
            
            if k == 0:
                A_per, _ = getProjectedNet(M_per,[True, False])
                beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(A_per))
            if k == 1:
                _, B_per = getProjectedNet(M_per,[False, True])
                beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(B_per))

            restoration_sequence.append(Node_choice)
            beta_eff_log.append(beta_eff)
            x_eff_log.append(x_eff)
            xl_log.append(xl)

            Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
            np.random.shuffle(Nodes_to_restore)

            data = [restoration_sequence, beta_eff_log, x_eff_log, xl_log]

    elif DATA_TYPE.split('_')[1] == '2D':
        xl, xP, xA = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=M_per)
        restoration_sequence = []
        xl_log = [xl]
        xP_log = [xP]
        xA_log = [xA]

        for p in range(len(Nodes_to_restore)):
            #print(p,":",Nodes_to_restore)
            centrality_restoration = {}
            for q in range(len(Nodes_to_restore)):
                M_per_temp = M_per.copy(deep = True)
                
                if k == 0:
                    M_per_temp = M_per_temp.append(M_df.loc[Nodes_to_restore[q]])
                if k == 1:
                    M_per_temp[Nodes_to_restore[q]] = M_df[Nodes_to_restore[q]]
                
                M_per_temp[M_per_temp.isnull()] = M_df
                NetM_per_temp = getNetDefinitions(M_per_temp)
                
                if Centrality == 'degree':
                    centrality_dict = nx.degree_centrality(NetM_per_temp)
                if Centrality == 'closeness':
                    centrality_dict = nx.closeness_centrality(NetM_per_temp)
                if Centrality == 'betweenness':
                    centrality_dict = nx.betweenness_centrality(NetM_per_temp)
                
                centrality_restoration[Nodes_to_restore[q]] = centrality_dict[Nodes_to_restore[q]]
            
            Node_choice = max(centrality_restoration, key = centrality_restoration.get)
            
            if k == 0:
                M_per = M_per.append(M_df.loc[Node_choice])
            if k == 1:
                M_per[Node_choice] = M_df[Node_choice]
            M_per[M_per.isnull()] = M_df
            
            xl, xP, xA = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=M_per)

            restoration_sequence.append(Node_choice)
            xl_log.append(xl)
            xP_log.append(xP)
            xA_log.append(xA)

            Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
            np.random.shuffle(Nodes_to_restore)
    
            data = [restoration_sequence, xP_log, xA_log, xl_log]
    if Centrality == 'degree':
        FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_four_{key}_{ensembles_ID}_{k}.pkl")
    if Centrality == 'closeness':
        FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_five_{key}_{ensembles_ID}_{k}.pkl")
    if Centrality == 'betweenness':
        FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_six_{key}_{ensembles_ID}_{k}.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    PickleObj(data, FileName)
    
    print(FileName,".....",tac(t_net_res))
    
    return data
 
def Random_Restoration(NetworkName, key, ensembles_ID, k, M_df, Node_del, Overwrite = False):
    if not Overwrite:
        try:
            FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_rand_{key}_{ensembles_ID}_{k}.pkl")
            data = UnpickleObj(FileName)
            #os.makedirs(os.path.dirname(FileName), exist_ok=True)
            #PickleObj(data, FileName)
            print(FileName, ':Overwrite:')
            return data
        except:
            pass

    t_sys_res = tic()
    M_per = RemovalFromNodeList(Node_del,M_df,axis=key[2])
    
    Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
    np.random.shuffle(Nodes_to_restore)

    if DATA_TYPE.split('_')[1] == '1D':
        if k == 0:
            A_per, _ = getProjectedNet(M_per,[True, False])
            beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(A_per))
        if k == 1:
            _, B_per = getProjectedNet(M_per,[False, True])
            beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(B_per))
        restoration_sequence = []
        beta_eff_log = [beta_eff]
        x_eff_log = [x_eff]
        xl_log = [xl]
                
        for p in range(len(Nodes_to_restore)):
            Node_choice = Nodes_to_restore[p]
            if k == 0:M_per = M_per.append(M_df.loc[Node_choice])
            if k == 1:M_per[Node_choice] = M_df[Node_choice]
            M_per[M_per.isnull()] = M_df
            if k == 0:
                A_per, _ = getProjectedNet(M_per,[True, False])
                beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(A_per))
            if k == 1:
                _, B_per = getProjectedNet(M_per,[False, True])
                beta_eff, x_eff, xl = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=np.array(B_per))
            restoration_sequence.append(Node_choice)
            beta_eff_log.append(beta_eff)
            x_eff_log.append(x_eff)
            xl_log.append(xl)
            #Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
            #np.random.shuffle(Nodes_to_restore)
            
        data = [restoration_sequence, beta_eff_log, x_eff_log, xl_log]

    elif DATA_TYPE.split('_')[1] == '2D':
        xl, xP, xA = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=M_per)
        restoration_sequence = []
        xl_log = [xl]
        xP_log = [xP]
        xA_log = [xA]
                
        for p in range(len(Nodes_to_restore)):
            Node_choice = Nodes_to_restore[p]
            if k == 0:M_per = M_per.append(M_df.loc[Node_choice])
            if k == 1:M_per[Node_choice] = M_df[Node_choice]
            M_per[M_per.isnull()] = M_df
            xl, xP, xA = ODE_Solve_Low(t0=0,tf=100,num=300,x_low=0,A=M_per)
            restoration_sequence.append(Node_choice)
            xl_log.append(xl)
            xP_log.append(xP)
            xA_log.append(xA)
            #Nodes_to_restore = getProspectiveSpecies(M_per, M_df, axis = k)
            #np.random.shuffle(Nodes_to_restore)
            
        data = [restoration_sequence, xP_log, xA_log, xl_log]
    
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_rand_{key}_{ensembles_ID}_{k}.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    
    PickleObj(data, FileName)
    
    print(FileName,".....",tac(t_sys_res))
    
    return data

#############################################################
"""
Solve for Network
"""
#############################################################
def Generator_iter(data_two):
    for key in list(data_two.keys()):
        for ensembles_ID in range(len(data_two[key])):
            if key[2] in [0,1]:
                yield (key, ensembles_ID, key[2])
            elif key[2] == 2:
                if DATA_TYPE.split('_')[0] == 'Real':
                    for target in range(key[2]):
                        yield (key, ensembles_ID, target)
                elif DATA_TYPE.split('_')[0] == 'Syn':
                    yield (key, ensembles_ID, 0)

def Solve_For_Network(NetworkName, Ensembles_batch, Keys_Filter, Flag = [True, True, True, True, True], Overwrite = False, File_Obj = None):
    M_df = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_one.pkl"))
    data_two = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_two.pkl"))
    data_two = {k: v[Ensembles_batch[0]:min(Ensembles_batch[1],len(v))] for k,v in data_two.items() if k in set(list(itertools.product(*Keys_Filter)))}
    Overlap = getOverlap(NetworkName, data_two)
    
    if File_Obj != None:
        File_Obj.Log_Entry('Network ID: ' + NetworkName)
        File_Obj.Log_Entry("Shape: {} ({})".format(M_df.shape, sum(M_df.shape)))
        File_Obj.Log_Entry("Ensembles Run: {}".format(Ensembles_batch))
        File_Obj.Log_Entry("Key Filter: {}".format(Keys_Filter))
    
    if Flag[0]:
        t_start = tic()
        with Parallel(n_jobs=-1, verbose = 10) as parallel:
            data = parallel(delayed(System_Driven_Restoration)(NetworkName, key, ensembles_ID, target, M_df, data_two[key][ensembles_ID], Overlap, Overwrite = Overwrite) for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1))
        if File_Obj != None:
            File_Obj.Log_Entry("System Driven Restoration: "+"....."+tac(t_start))
    
    if Flag[1]:
        t_start = tic()
        with Parallel(n_jobs=-1, verbose = 10) as parallel:
            data = parallel(delayed(Centrality_Driven_Restoration)(NetworkName, key, ensembles_ID, target, 'degree', M_df, data_two[key][ensembles_ID], Overlap, Overwrite = Overwrite) for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1))
        if File_Obj != None:
            File_Obj.Log_Entry("Degree Centrality Driven Restoration: "+"....."+tac(t_start))

    
    if Flag[2]:
        t_start = tic()
        with Parallel(n_jobs=-1, verbose = 10) as parallel:
            data = parallel(delayed(Centrality_Driven_Restoration)(NetworkName, key, ensembles_ID, target, 'closeness', M_df, data_two[key][ensembles_ID], Overlap, Overwrite = Overwrite) for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1))
        if File_Obj != None:
            File_Obj.Log_Entry("Closeness Centrality Driven Restoration: "+"....."+tac(t_start))
    
    if Flag[3]:
        t_start = tic()
        with Parallel(n_jobs=-1, verbose = 10) as parallel:
            data = parallel(delayed(Centrality_Driven_Restoration)(NetworkName, key, ensembles_ID, target, 'betweenness', M_df, data_two[key][ensembles_ID], Overlap, Overwrite = Overwrite) for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1))
        if File_Obj != None:
            File_Obj.Log_Entry("Betweenness Centrality Driven Restoration: "+"....."+tac(t_start))
    
    if Flag[4]:
        t_start = tic()
        with Parallel(n_jobs=-1, verbose = 10) as parallel:
            data = parallel(delayed(Random_Restoration)(NetworkName, key, ensembles_ID, target, M_df, data_two[key][ensembles_ID], Overwrite = Overwrite) for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1))
        if File_Obj != None:
            File_Obj.Log_Entry("Random Restoration: "+"....."+tac(t_start))

def Solve_For_Network2(NetworkName, Ensembles_batch, Keys_Filter, Flag = [True, True, True, True, True], Overwrite = False, File_Obj = None):
    M_df = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_one.pkl"))
    data_two = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_two.pkl"))
    data_two = {k: v[Ensembles_batch[0]:min(Ensembles_batch[1],len(v))] for k,v in data_two.items() if k in set(list(itertools.product(*Keys_Filter)))}
    Overlap = getOverlap(NetworkName, data_two)
    
    if File_Obj != None:
        File_Obj.Log_Entry('Network ID: ' + NetworkName)
        File_Obj.Log_Entry("Shape: {} ({})".format(M_df.shape, sum(M_df.shape)))
        File_Obj.Log_Entry("Ensembles Run: {}".format(Ensembles_batch))
        File_Obj.Log_Entry("Key Filter: {}".format(Keys_Filter))
    
    if Flag[0]:
        t_start = tic()
        for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1):
            data = System_Driven_Restoration(
                NetworkName = NetworkName, 
                key = key, 
                ensembles_ID = ensembles_ID, 
                k = target, 
                M_df = M_df, 
                Node_del = data_two[key][ensembles_ID], 
                Overlap = Overlap,
                Overwrite = Overwrite
                )
        if File_Obj != None:
            File_Obj.Log_Entry("System Driven Restoration: "+"....."+tac(t_start))
    
    if Flag[1]:
        t_start = tic()
        for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1):
            data = Centrality_Driven_Restoration(
                NetworkName = NetworkName, 
                key = key, 
                ensembles_ID = ensembles_ID, 
                k = target, 
                Centrality = 'degree', 
                M_df = M_df, 
                Node_del = data_two[key][ensembles_ID], 
                Overwrite = Overwrite
                )
        if File_Obj != None:
            File_Obj.Log_Entry("Degree Centrality Driven Restoration: "+"....."+tac(t_start))

    
    if Flag[2]:
        t_start = tic()
        for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1):
            data = Centrality_Driven_Restoration(
                NetworkName = NetworkName, 
                key = key, 
                ensembles_ID = ensembles_ID, 
                k = target, 
                Centrality = 'closeness', 
                M_df = M_df, 
                Node_del = data_two[key][ensembles_ID], 
                Overwrite = Overwrite
                )
        if File_Obj != None:
            File_Obj.Log_Entry("Closeness Centrality Driven Restoration: "+"....."+tac(t_start))
    
    if Flag[3]:
        t_start = tic()
        for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1):
            data = Centrality_Driven_Restoration(
                NetworkName = NetworkName, 
                key = key, 
                ensembles_ID = ensembles_ID, 
                k = target, 
                Centrality = 'betweenness', 
                M_df = M_df, 
                Node_del = data_two[key][ensembles_ID], 
                Overwrite = Overwrite
                )
        if File_Obj != None:
            File_Obj.Log_Entry("Betweenness Centrality Driven Restoration: "+"....."+tac(t_start))
    
    if Flag[4]:
        t_start = tic()
        for (key, ensembles_ID, target) in itertools.islice(Generator_iter(data_two),0,None,1):
            data = Random_Restoration(
                NetworkName = NetworkName, 
                key = key, 
                ensembles_ID = ensembles_ID, 
                k = target, 
                M_df = M_df, 
                Node_del = data_two[key][ensembles_ID], 
                Overwrite = Overwrite
                )
        if File_Obj != None:
            File_Obj.Log_Entry("Random Restoration: "+"....."+tac(t_start))

#############################################################
"""
Processing Simulation
"""
#############################################################
def Generator_getDict(data_two):
    for key in list(data_two.keys()):
        ensembles_num = len(data_two[key])
        if key[2] in [0,1]:
            yield (key, ensembles_num, key[2])
        elif key[2] == 2:
            if DATA_TYPE.split('_')[0] == 'Real':
                for target in range(key[2]):
                    yield (key, ensembles_num, target)
            elif DATA_TYPE.split('_')[0] == 'Syn':
                yield (key, ensembles_num, 0)

def getDict_x(ID, data_two, NetworkName, Shape, Steady_State_Correction = True):
    if DATA_TYPE.split('_')[1] == '1D':
        getDict_x_1D(ID, data_two, NetworkName, Shape, Steady_State_Correction)
    elif DATA_TYPE.split('_')[1] == '2D':
        getDict_x_2D(ID, data_two, NetworkName, Shape, Steady_State_Correction)

def getDict_x_1D(ID, data_two, NetworkName, Shape, Steady_State_Correction = True):
    Data = {}
    Data_Steady = {}
    #ID = 'Data_three'
    #Steady_State_Correction = True
    #Shape = M_df.shape

    for (key, ensembles_num, target) in itertools.islice(Generator_getDict(data_two),0,None,1):
        Data[(key, target)] = []
        Data_Steady[(key,target)] = {}
        FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-{ID}_{key}_0_{target}.pkl")
        x_beta = UnpickleObj(FileName)[1]
        x_abundance = UnpickleObj(FileName)[3]
        
        x_avg_abundance = np.zeros(len(x_abundance))
        x_persistence = np.zeros(len(x_abundance))
        X_Persistence = np.zeros(len(x_abundance))
        
        Steady_Abundance = np.zeros((len(x_abundance),ensembles_num))
        Steady_Persistence = np.zeros((len(x_abundance),ensembles_num))
        Steady_Resilience = np.zeros((len(x_abundance),ensembles_num))
        Steady_Beta = np.zeros((len(x_abundance),ensembles_num))
        
        if Steady_State_Correction:
            for step in range(len(x_abundance)):
                x_avg_abundance[step] = np.mean(np.array(list(map(isExtinct,x_abundance[step][-1,:]))))
                X_Persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / Shape[target]
        else:
            for step in range(len(x_abundance)):
                x_avg_abundance[step] = np.mean(np.array(x_abundance[step][-1,:]))
                X_Persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / Shape[target]
        X_Beta = np.array(x_beta)
        X_Abundance = x_avg_abundance
        X_Resilience = np.array(list(map(getResilience, x_abundance)))
        
        Steady_Abundance[:, 0] = x_avg_abundance[:]
        Steady_Resilience[:,0] = X_Resilience[:]
        Steady_Persistence[:,0] = X_Persistence[:]
        Steady_Beta[:,0] = X_Beta[:]
        
        for ensembles_ID in range(1,ensembles_num):
            FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-{ID}_{key}_{ensembles_ID}_{target}.pkl")
            x_beta = UnpickleObj(FileName)[1]
            x_abundance = UnpickleObj(FileName)[3]
            
            x_avg_abundance = np.zeros(len(x_abundance))
            x_persistence = np.zeros(len(x_abundance))
            
            if Steady_State_Correction:
                for step in range(len(x_abundance)):
                    x_avg_abundance[step] = np.mean(np.array(list(map(isExtinct,x_abundance[step][-1,:]))))
                    x_persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / Shape[target]
            else:
                for step in range(len(x_abundance)):
                    x_avg_abundance[step] = np.mean(np.array(x_abundance[step][-1,:]))
                    x_persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / Shape[target]
                    
            X_Beta = np.add(X_Beta,np.array(x_beta))
            X_Abundance = np.add(X_Abundance,x_avg_abundance)
            X_Resilience = np.add(X_Resilience,np.array(list(map(getResilience, x_abundance))))
            X_Persistence = np.add(X_Persistence,x_persistence)
            
            Steady_Abundance[:, ensembles_ID] = np.array(x_avg_abundance[:])
            Steady_Resilience[:,ensembles_ID] = np.array(list(map(getResilience, x_abundance)))
            Steady_Persistence[:,ensembles_ID] = x_persistence[:]
            Steady_Beta[:,ensembles_ID] = np.array(x_beta[:])
        
        X_Beta = np.divide(X_Beta, ensembles_num)
        X_Abundance = np.divide(X_Abundance, ensembles_num)
        X_Resilience = np.divide(X_Resilience, ensembles_num)
        X_Persistence = np.divide(X_Persistence, ensembles_num)
        
        Data[(key, target)] = [X_Abundance, X_Beta, X_Resilience, X_Persistence]
        
        Data_Steady[(key, target)]['Abundance'] = Steady_Abundance
        Data_Steady[(key, target)]['Beta'] = Steady_Beta
        Data_Steady[(key, target)]['Resilience'] = Steady_Resilience
        Data_Steady[(key, target)]['Persistence'] = Steady_Persistence
        
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-{ID}.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    PickleObj(Data, FileName)

    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-{ID}_All.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    PickleObj(Data_Steady, FileName)
        
    return [Data, Data_Steady]

def getDict_x_2D(ID, data_two, NetworkName, Shape, Steady_State_Correction = True):
    Data = {}
    Data_Steady = {}
    #ID = 'Data_three'
    #Steady_State_Correction = True
    #Shape = M_df.shape
    for (key, ensembles_num, target) in itertools.islice(Generator_getDict(data_two),0,None,1):
        Data[(key, target)] = []
        Data_Steady[(key,target)] = {}
        FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-{ID}_{key}_0_{target}.pkl")
        #x_beta = UnpickleObj(FileName)[1]
        x_abundance = UnpickleObj(FileName)[3]
        
        x_avg_abundance = np.zeros(len(x_abundance))        
        x_persistence = np.zeros(len(x_abundance))
        X_Persistence = np.zeros(len(x_abundance))
        
        Steady_Abundance = np.zeros((len(x_abundance),ensembles_num))
        Steady_Persistence = np.zeros((len(x_abundance),ensembles_num))
        Steady_Resilience = np.zeros((len(x_abundance),ensembles_num))
        #Steady_Beta = np.zeros((len(x_abundance),ensembles_num))
        
        if Steady_State_Correction:
            for step in range(len(x_abundance)):
                x_avg_abundance[step] = np.mean(np.array(list(map(isExtinct,x_abundance[step][-1,:]))))
                X_Persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / (Shape[0]+Shape[1])
        else:
            for step in range(len(x_abundance)):
                x_avg_abundance[step] = np.mean(np.array(x_abundance[step][-1,:]))
                X_Persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / (Shape[0]+Shape[1])
        #X_Beta = np.array(x_beta)
        X_Abundance = x_avg_abundance
        X_Resilience = np.array(list(map(getResilience, x_abundance)))
        
        Steady_Abundance[:, 0] = x_avg_abundance[:]
        Steady_Resilience[:,0] = X_Resilience[:]
        Steady_Persistence[:,0] = X_Persistence[:]
        #Steady_Beta[:,0] = X_Beta[:]
        
        for ensembles_ID in range(1,ensembles_num):
            FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-{ID}_{key}_{ensembles_ID}_{target}.pkl")
            #x_beta = UnpickleObj(FileName)[1]
            x_abundance = UnpickleObj(FileName)[3]
            
            x_avg_abundance = np.zeros(len(x_abundance))
            x_persistence = np.zeros(len(x_abundance))
            
            if Steady_State_Correction:
                for step in range(len(x_abundance)):
                    x_avg_abundance[step] = np.mean(np.array(list(map(isExtinct,x_abundance[step][-1,:]))))
                    x_persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / (Shape[0]+Shape[1])
            else:
                for step in range(len(x_abundance)):
                    x_avg_abundance[step] = np.mean(np.array(x_abundance[step][-1,:]))
                    x_persistence[step] = getPersistence(np.array(list(map(isExtinct,x_abundance[step][-1,:])))) / (Shape[0]+Shape[1])
                    
            #X_Beta = np.add(X_Beta,np.array(x_beta))
            X_Abundance = np.add(X_Abundance,x_avg_abundance)
            X_Resilience = np.add(X_Resilience,np.array(list(map(getResilience, x_abundance))))
            X_Persistence = np.add(X_Persistence,x_persistence)
            
            Steady_Abundance[:, ensembles_ID] = np.array(x_avg_abundance[:])
            Steady_Resilience[:,ensembles_ID] = np.array(list(map(getResilience, x_abundance)))
            Steady_Persistence[:,ensembles_ID] = x_persistence[:]
            #Steady_Beta[:,ensembles_ID] = np.array(x_beta[:])
        
        #X_Beta = np.divide(X_Beta, ensembles_num)
        X_Abundance = np.divide(X_Abundance, ensembles_num)
        X_Resilience = np.divide(X_Resilience, ensembles_num)
        X_Persistence = np.divide(X_Persistence, ensembles_num)
        
        Data[(key, target)] = [X_Abundance, [], X_Resilience, X_Persistence]
        
        Data_Steady[(key, target)]['Abundance'] = Steady_Abundance
        Data_Steady[(key, target)]['Beta'] = []
        Data_Steady[(key, target)]['Resilience'] = Steady_Resilience
        Data_Steady[(key, target)]['Persistence'] = Steady_Persistence
        
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-{ID}.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    PickleObj(Data, FileName)

    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-{ID}_All.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    PickleObj(Data_Steady, FileName)
        
    return [Data, Data_Steady]

#############################################################
"""
Nestedness
"""
#############################################################
class NestednessCalculator(object):
    """Calculates the nestedness of the input matrix.
    The algorithms that have been implemented are:
        - NODF (Nestedness based on Overlap and Decreasing Fill)
    """
    def __init__(self, mat):
        """Initialize the Nestedness calculator and check the input matrix.
        :param mat: binary input matrix
        :type mat: numpy.array
        """
        self.check_input_matrix_is_binary(mat)
        self.check_degrees(mat)

    @staticmethod
    def check_input_matrix_is_binary(mat):
        """Check that the input matrix is binary, i.e. entries are 0 or 1.
        :param mat: binary input matrix
        :type mat: numpy.array
        :raise AssertionError: raise an error if the input matrix is not
            binary
        """
        assert np.all(np.logical_or(mat == 0, mat == 1)), \
            "Input matrix is not binary."

    @staticmethod
    def check_degrees(mat):
        """Check that rows and columns are not completely zero.
        :param mat: binary input matrix
        :type mat: numpy.array
        :raise AssertionError: raise an error if the input matrix has
            completely zero rows or columns.
        """
        assert np.all(mat.sum(axis=1) != 0), \
            "Input matrix rows with only zeros, abort."
        assert np.all(mat.sum(axis=0) != 0), \
            "Input matrix columns with only zeros, abort."

    ################################################################################
    # NODF - Nestedness based on Overlap and Decreasing Fill
    ################################################################################

    def get_paired_nestedness(self, mat, rows=True):
        """Calculate the paired nestedness along the rows or columns of the.
        :param mat: binary input matrix
        :type mat: numpy.array
        :param rows: if True, pairs are calculated along the rows, if False
            along the columns
        :type rows: bool
        :returns: degree of paired nestedness
        :rtype: float
        The method uses the algorithm described in the `BiMat framework for
        MATLAB <https://bimat.github.io/alg/nestedness.html>`_.
        """
        if rows:
            # consider rows
            po_mat = np.dot(mat, mat.T)
            degrees = mat.sum(axis=1)
        else:
            # consider cols
            po_mat = np.dot(mat.T, mat)
            degrees = mat.sum(axis=0)
        assert len(degrees) == len(po_mat)

        neg_delta = (degrees != degrees[:, np.newaxis])
        deg_matrix = degrees * np.ones_like(po_mat)
        deg_minima = np.minimum(deg_matrix, deg_matrix.T)
        n_pairs = po_mat[neg_delta] / (2. * deg_minima[neg_delta])
        return n_pairs.sum()

    def nodf(self, mat):
        """Calculate the NODF nestedness of the input matrix [AlmeidaNeto]_.
        :param mat: binary input matrix
        :type mat: numpy.array
        :returns: NODF nestedness of the input matrix
        :rtype: float
        The algorithm has been tested by comparison with the `online tool
        provided at <http://ecosoft.alwaysdata.net/>`_
        """
        n_pairs_rows = self.get_paired_nestedness(mat, rows=True)
        n_pairs_cols = self.get_paired_nestedness(mat, rows=False)
        norm = np.sum(np.array(mat.shape) * (np.array(mat.shape) - 1) / 2.)
        nodf = (n_pairs_rows + n_pairs_cols) / norm
        return nodf

#############################################################
"""
Synthetic Data Generation
"""
#############################################################
def get_SyntheticData_type01(A = 0.5, C = 0.2, S = 50, seed = 0):
    N = round(S / (1+A))
    M = round((S*A) / (1+A))
    L = round(C*N*M)

    X = np.zeros((N,M),int)
    np.fill_diagonal(X,1)
    
    np.random.seed(seed)
    
    non_filled_rows = np.array(list(itertools.product(np.arange(M,N),np.arange(M))))
    idx = np.array(np.random.choice(len(non_filled_rows),N-M))
    
    for i in range(N-M):
        X[non_filled_rows[idx[i]][0],non_filled_rows[idx[i]][1]] = 1
    
    non_filled_cells = np.argwhere(X==0)
    idx = np.array(np.random.choice(len(non_filled_cells),L - N))
    
    for i in range(L - N):
        X[non_filled_cells[idx[i]][0],non_filled_cells[idx[i]][1]] = 1
    
    if np.all(np.sum(X,axis = 1) > 0) and np.all(np.sum(X,axis = 0) > 0):
        nodf_score = NestednessCalculator(X).nodf(X)
        #print(nodf_score)
        
        M_df = pd.DataFrame(X)
        M_df.columns = np.char.add('B',np.array(range(len(M_df.columns)), dtype = 'str'))
        M_df.index = np.char.add('A',np.array(range(len(M_df.index)), dtype = 'str'))
        
        return M_df, nodf_score
    else:
        return pd.DataFrame(np.zeros((N,M),int)),0.0

def get_SyntheticData_type02(N = 33, M = 17, L = 112, seed = 0):
    X = np.zeros((N,M),int)
    np.fill_diagonal(X,1)
    
    np.random.seed(seed)
    
    non_filled_rows = np.array(list(itertools.product(np.arange(M,N),np.arange(M))))
    idx = np.array(np.random.choice(len(non_filled_rows),N-M))
    
    for i in range(N-M):
        X[non_filled_rows[idx[i]][0],non_filled_rows[idx[i]][1]] = 1
    
    non_filled_cells = np.argwhere(X==0)
    idx = np.array(np.random.choice(len(non_filled_cells),L - N))
    
    for i in range(L - N):
        X[non_filled_cells[idx[i]][0],non_filled_cells[idx[i]][1]] = 1
    
    if np.all(np.sum(X,axis = 1) > 0) and np.all(np.sum(X,axis = 0) > 0):    
        nodf_score = NestednessCalculator(X).nodf(X)
        #print(nodf_score)
        
        M_df = pd.DataFrame(X)
        M_df.columns = np.char.add('B',np.array(range(len(M_df.columns)), dtype = 'str'))
        M_df.index = np.char.add('A',np.array(range(len(M_df.index)), dtype = 'str'))
        
        return M_df,nodf_score
    else:
        return pd.DataFrame(np.zeros((N,M),int)),0.0

def init_SyntheticData(A = 0.5, C = 0.2, S = 50, ind = 0, FileObj = None):
    M_df_collec = []
    nodf_collec = []
    for seed in range(1000):
        M_df, nodf = get_SyntheticData_type01(A = A, C = C, S = S, seed = seed)
        if nodf != 0:
            M_df_collec.append(M_df)
            nodf_collec.append(nodf)
    del seed,M_df,nodf
    X = pd.DataFrame()
    X['idx'] = np.arange(len(nodf_collec))
    X['nodf'] = nodf_collec
    X = X.sort_values(by = 'nodf')
    
    M_df_selected  = [
        M_df_collec[int(X.iloc[0]['idx'])],
        M_df_collec[int(X.iloc[int(len(M_df_collec)/2)]['idx'])],
        M_df_collec[int(X.iloc[-1]['idx'])]
        ]
    nodf_selected  = [
        nodf_collec[int(X.iloc[0]['idx'])],
        nodf_collec[int(X.iloc[int(len(M_df_collec)/2)]['idx'])],
        nodf_collec[int(X.iloc[-1]['idx'])]
        ]
    if FileObj != None:FileObj.Log_Entry("S = {}, A = {}, C = {}".format(S, A, C))
    for ind_2 in range(3):
        FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"M_PL_{S}_{ind}_{ind_2}", f"M_PL_{S}_{ind}_{ind_2}-Data_one.pkl")
        os.makedirs(os.path.dirname(FileName), exist_ok=True)
        PickleObj(M_df_selected[ind_2], FileName)
        if FileObj != None:FileObj.Log_Entry(str(nodf_selected[ind_2]) + ' : ' + f"M_PL_{S}_{ind}_{ind_2}")

