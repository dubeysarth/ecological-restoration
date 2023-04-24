import os
def add_path():
    sys.path.insert(1,os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions'))
from Species_Reintroductions import *
global DATA_TYPE
DATA_TYPE = 'Real_1D'
CASE_TO_SOLVE_init(DATA_TYPE)
get_CASE()
'''
def CASE_init(CASE_TO_SOLVE):
    DATA_TYPE = CASE_TO_SOLVE
    CASE_TO_SOLVE_init(DATA_TYPE)
    NetworkName_Solved_df = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"Network_Solved_{DATA_TYPE.split('_')[0]}.pkl"))
    M = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"M_{DATA_TYPE.split('_')[0]}.pkl"))
    if DATA_TYPE.split('_')[0] == 'Syn':
        M = {k:M[k] for k in M.keys() if int(k.split('_')[2]) == 100}
    return DATA_TYPE, NetworkName_Solved_df, M

from scipy.optimize import fsolve
from scipy import stats
import scipy.linalg as la
from sklearn.linear_model import LinearRegression
#%%
def G_iter(data_two):
    for key in list(data_two.keys()):
        for ensembles_ID in range(len(data_two[key])):
            yield (key, ensembles_ID, 0)

def get_Data(NetworkName, loc, File_Obj = None):
    Data_Scatter = []
    #print(NetworkName)
    M_df = M[NetworkName]
    #data_two = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", f"{NetworkName}", f"{NetworkName}-Data_two.pkl"))
    #data_two = UnpickleObj(os.path.join(r"D:\WORK\EN\Documents\Ecological_Restoration_01\Real_Data", f"{NetworkName}\{NetworkName}-Data_two.pkl"))
    data_two = UnpickleObj(f"/home/udit/Documents/Ecological_Restoration_{loc}/{DATA_TYPE.split('_')[0]}_Data/{NetworkName}/{NetworkName}-Data_two.pkl")
    Keys_Filter = [[0,1,2], [2,5,8], [0,2]]
    Ensembles_batch = [0,10]
    data_two = {k: v[Ensembles_batch[0]:min(Ensembles_batch[1],len(v))] for k,v in data_two.items() if k in set(list(itertools.product(*Keys_Filter)))}
    del Keys_Filter, Ensembles_batch
    P = {i:j for i,j in zip(['a', 'b', 'mu', 'h', 'g0', 't', 'k'],[0.3, 1.0, 1e-4, 0.7, 1.0, 0.5, 0.0])}
    for approach in ['three','four','five','six','rand']:
        for key, E, T in itertools.islice(G_iter(data_two),0,None,1):
            try:
                #X = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", NetworkName, f"{NetworkName}-Data_{approach}_{key}_{E}_{T}.pkl"))
                #X = UnpickleObj(os.path.join(r"D:\WORK\EN\Documents\Ecological_Restoration_04\Real_Data", f"{NetworkName}\{NetworkName}-Data_{approach}_{key}_{E}_{T}.pkl"))
                X = UnpickleObj(f"/home/udit/Documents/Ecological_Restoration_{loc}/{DATA_TYPE.split('_')[0]}_Data/{NetworkName}/{NetworkName}-Data_{approach}_{key}_{E}_{T}.pkl")
                M_per = RemovalFromNodeList(data_two[key][E],M_df,axis=key[2])
                M_per = M_per.loc[(M_per!=0).any(axis=1)] #Cleaning Row isoloates
                M_per = M_per.loc[:, (M_per!=0).any(axis=0)] #Cleaning Column isoloates
                
                M_per_dict = {}
                
                Degree_P = list(np.sum(np.array(M_per), axis = 1))
                Degree_A = list(np.sum(np.array(M_per), axis = 0))
                g_p = np.sum(P['g0'] * np.power(np.array(Degree_P),1-P['t'])) / len(Degree_P)
                g_a = np.sum(P['g0'] * np.power(np.array(Degree_A),1-P['t'])) / len(Degree_A)
                M_per_dict['g_p'] = g_p
                M_per_dict['g_a'] = g_a
                beta_eff_p, _, _ = ODE_Solve_Low(t0=0,tf=100,num=50,x_low=0,A=np.array(getProjectedNet(M_per, flag = [True, False])[0]))
                beta_eff_a, _, _ = ODE_Solve_Low(t0=0,tf=100,num=50,x_low=0,A=np.array(getProjectedNet(M_per, flag = [False, True])[1]))
                M_per_dict['beta_eff_p'] = beta_eff_p
                M_per_dict['beta_eff_a'] = beta_eff_a
                
                M_per_dict['n'] = M_per.shape[0]
                M_per_dict['m'] = M_per.shape[1]
                M_per_dict['L'] = sum(np.array(M_per).flatten())
                M_per_dict['S'] = M_per_dict['n'] + M_per_dict['m']
                M_per_dict['A'] = M_per_dict['m'] / M_per_dict['n']
                M_per_dict['C'] = M_per_dict['L'] / (M_per_dict['m']*M_per_dict['n'])
                M_per_dict['N'] = NestednessCalculator(np.array(M_per)).nodf(np.array(M_per))
                
                Z1 = np.mean(np.array(X[3][0][-1,:]))
                Z2 = np.mean(np.array(list(map(isExtinct,X[3][0][-1,:]))))
                Z3 = getResilience(X[3][0])
                Z4 = getPersistence(np.array(list(map(isExtinct,X[3][0][-1,:])))) / M_df.shape[T]
                
                temp = [M_per_dict[x] for x in ['n','m','L','S','A','C','N','g_p','g_a','beta_eff_p','beta_eff_a']] + [Z1,Z2,Z3,Z4]
                Data_Scatter += [temp]
                
                for Step in list(range(1,len(X[0])+1)):
                    M_per = M_per.append(M_df.loc[X[0][Step-1]])
                    M_per[M_per.isnull()] = M_df
                    M_per = M_per.loc[(M_per!=0).any(axis=1)] #Cleaning Row isoloates
                    M_per = M_per.loc[:, (M_per!=0).any(axis=0)] #Cleaning Column isoloates
                    
                    Degree_P = list(np.sum(np.array(M_per), axis = 1))
                    Degree_A = list(np.sum(np.array(M_per), axis = 0))
                    g_p = np.sum(P['g0'] * np.power(np.array(Degree_P),1-P['t'])) / len(Degree_P)
                    g_a = np.sum(P['g0'] * np.power(np.array(Degree_A),1-P['t'])) / len(Degree_A)
                    M_per_dict['g_p'] = g_p
                    M_per_dict['g_a'] = g_a
                    beta_eff_p, _, _ = ODE_Solve_Low(t0=0,tf=100,num=50,x_low=0,A=np.array(getProjectedNet(M_per, flag = [True, False])[0]))
                    beta_eff_a, _, _ = ODE_Solve_Low(t0=0,tf=100,num=50,x_low=0,A=np.array(getProjectedNet(M_per, flag = [False, True])[1]))
                    M_per_dict['beta_eff_p'] = beta_eff_p
                    M_per_dict['beta_eff_a'] = beta_eff_a

                    M_per_dict['n'] = M_per.shape[0]
                    M_per_dict['m'] = M_per.shape[1]
                    M_per_dict['L'] = sum(np.array(M_per).flatten())
                    M_per_dict['S'] = M_per_dict['n'] + M_per_dict['m']
                    M_per_dict['A'] = M_per_dict['m'] / M_per_dict['n']
                    M_per_dict['C'] = M_per_dict['L'] / (M_per_dict['m']*M_per_dict['n'])
                    M_per_dict['N'] = NestednessCalculator(np.array(M_per)).nodf(np.array(M_per))
                    
                    Z1 = np.mean(np.array(X[3][Step][-1,:]))
                    Z2 = np.mean(np.array(list(map(isExtinct,X[3][Step][-1,:]))))
                    Z3 = getResilience(X[3][Step])
                    Z4 = getPersistence(np.array(list(map(isExtinct,X[3][Step][-1,:])))) / M_df.shape[T]
                    
                    temp = [M_per_dict[x] for x in ['n','m','L','S','A','C','N','g_p','g_a','beta_eff_p','beta_eff_a']] + [Z1,Z2,Z3,Z4]
                    Data_Scatter += [temp]
            except:
                #print(key, E, T)
                pass
    DF = pd.DataFrame(Data_Scatter, columns = ['n','m','L','S','A','C','N','g_p','g_a','beta_eff_p','beta_eff_a','Abundance','Ob Abundance','Settling Time','Persistence'])
    #DF.drop_duplicates(inplace=True, ignore_index=True)
    FileName = os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '05_Analysis', 'Dyna_of_SR', f"{loc}", f"{NetworkName}.pkl")
    os.makedirs(os.path.dirname(FileName), exist_ok=True)
    if DF.shape[0] != 0:
        if File_Obj != None:
            File_Obj.Log_Entry(f"{NetworkName} : {DF.shape}")
        PickleObj(DF,FileName)
    else:
        File_Obj.Log_Entry(f"{NetworkName} : {DF.shape}")
#%%
Log_Dyna = Write_Log(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '05_Analysis', 'Dyna_of_SR', 'Log_Dyna.txt'), True)
for CASE_TO_SOLVE, loc in zip(['Real_1D', 'Syn_01', 'Syn_2D', 'Real_2D'],['01','02','03','04']):
    DATA_TYPE = CASE_TO_SOLVE
    CASE_TO_SOLVE_init(DATA_TYPE)
    NetworkName_Solved_df = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"Network_Solved_{DATA_TYPE.split('_')[0]}.pkl"))
    M = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"M_{DATA_TYPE.split('_')[0]}.pkl"))

    if DATA_TYPE.split('_')[0] == 'Syn':
        M = {k:M[k] for k in M.keys() if int(k.split('_')[2]) == 100}
    
    with Parallel(n_jobs=5, verbose = 10) as parallel:
            data = parallel(delayed(get_Data)(NetworkName,loc) for NetworkName in itertools.islice(M.keys(),0,None,1))
Log_Dyna.close_Log()
'''