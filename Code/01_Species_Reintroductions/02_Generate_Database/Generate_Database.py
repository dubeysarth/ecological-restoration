import os
while os.getcwd().split('\\')[-1] != 'ecological-networks':
    %cd ..
import setup_paths
setup_paths.add_path()
from Species_Reintroductions import *
DATA_TYPE = 'Real_1D'
CASE_TO_SOLVE_init(DATA_TYPE)
print(get_CASE())
'''
M = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"M_{DATA_TYPE.split('_')[0]}.pkl"))
NetworkName_Solved_df = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"Network_Solved_{DATA_TYPE.split('_')[0]}.pkl"))

if DATA_TYPE.split('_')[0] == 'Syn':
    M = {k:M[k] for k in M.keys() if int(k.split('_')[2]) == 100}
'''

#%%
###########################################################################
###########################################################################
'''
Getting Heatmap [DONE]
'''
###########################################################################
###########################################################################
def get_Scores_1D(NetworkName, Flag):
    
    if len(set(Flag).difference({0,1,2,3,4})) != 0:
        return None, None
    
    Strategy_Name = ['System','Degree','Closeness','Betweenness','Random']
    Flag_Strategy = {k:Strategy_Name[k] for k in Flag}
    del Strategy_Name

    ###########################################
    # Get Data
    ###########################################
    Data = [
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_three.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_four.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_five.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_six.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_rand.pkl"))
            ]
    Data_All= [
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_three_All.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_four_All.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_five_All.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_six_All.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_rand_All.pkl"))
            ]
    Data_Var = []
    for i in range(len(Data_All)):
        Data_Var.append({})
        for key,value in Data_All[i].items():
            
            Var_Abundance = np.array([np.std(ensemble) for ensemble in list(Data_All[i][key]['Abundance'])])
            Var_Beta = np.array([np.std(ensemble) for ensemble in list(Data_All[i][key]['Beta'])])
            Var_Resilience = np.array([np.std(ensemble) for ensemble in list(Data_All[i][key]['Resilience'])])
            Var_Persistence = np.array([np.std(ensemble) for ensemble in list(Data_All[i][key]['Persistence'])])
            
            Data_Var[i][key] = [Var_Abundance, Var_Beta, Var_Resilience, Var_Persistence]
    del i, key, value
    del Var_Abundance, Var_Beta, Var_Resilience, Var_Persistence
    del Data_All

    ###########################################
    # Heatmap Scores
    ###########################################
    Score_Abs_01 = {}
    Score_Abs_02 = {}
    Score_Abs_03 = {}
    
    Score_Var_01 = {}
    Score_Var_02 = {}
    Score_Var_03 = {}
    
    def get_sampling():
        for i in [0,1,2]:
            for j in [2,5,8]:
                for k in [0,1,2]:
                    if k in [0,1]:
                        yield (i,j,k),k
                    elif k == 2:
                        for target in [0,1]:
                            yield (i,j,k),target
    
    def Summarize(key,target, Var_ind, Intent = 'Min', Ref = 'Absolute'):
        if Ref == 'Absolute':
            Temp = []
            for ind in Flag:
                Temp.append(Data[ind][key,target][Var_ind])
        elif Ref == 'Variability':
            Temp = []
            for ind in Flag:
                Temp.append(Data_Var[ind][key,target][Var_ind])
        
        Val = np.zeros((len(Temp),len(Temp[0])-2))
        for i in range(1,len(Temp[0])-1):
            #x = [temp0[i],temp1[i],temp2[i],temp3[i],temp4[i]]
            x =[]
            for ind in range(len(Flag)):
                x.append(Temp[ind][i]) 
            if Intent == 'Max':
                y = max(x)
            elif Intent == 'Min':
                y = min(x)
            Val[:,i-1] = [1 if z == y else 0 for z in x]
        Val_sum = list(np.sum(Val, axis = 1) / Val.shape[1])
        return Val_sum
    
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_01[key,target] = Summarize(key,target,0,'Max','Absolute') ## Abundance
        Score_Abs_02[key,target] = Summarize(key,target,2,'Min','Absolute') ## Resilience
        Score_Abs_03[key,target] = Summarize(key,target,3,'Max','Absolute') ## Persistence
        
        Score_Var_01[key,target] = Summarize(key,target,0,'Min','Variability') 
        Score_Var_02[key,target] = Summarize(key,target,2,'Min','Variability')
        Score_Var_03[key,target] = Summarize(key,target,3,'Min','Variability')
    del key, target
    
    Score_Heatmap = [Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03, Flag]
    del Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03
    
    ###########################################
    # Strategy Scores
    ###########################################
    Score_Abs_Res_Strategy_01 = {}
    Score_Abs_Res_Strategy_02 = {}
    Score_Abs_Res_Strategy_03 = {}
    
    Score_Var_Res_Strategy_01 = {}
    Score_Var_Res_Strategy_02 = {}
    Score_Var_Res_Strategy_03 = {}
    
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_Res_Strategy_01[key,target] = Flag[Score_Heatmap[0][key,target].index(max(Score_Heatmap[0][key,target]))]
        Score_Abs_Res_Strategy_02[key,target] = Flag[Score_Heatmap[1][key,target].index(max(Score_Heatmap[1][key,target]))]
        Score_Abs_Res_Strategy_03[key,target] = Flag[Score_Heatmap[2][key,target].index(max(Score_Heatmap[2][key,target]))]
        
        Score_Var_Res_Strategy_01[key,target] = Flag[Score_Heatmap[3][key,target].index(max(Score_Heatmap[3][key,target]))]
        Score_Var_Res_Strategy_02[key,target] = Flag[Score_Heatmap[4][key,target].index(max(Score_Heatmap[4][key,target]))]
        Score_Var_Res_Strategy_03[key,target] = Flag[Score_Heatmap[5][key,target].index(max(Score_Heatmap[5][key,target]))]
    del key, target
    
    Score_Strategy = [Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03, Flag_Strategy]
    del Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03
    
    ###########################################
    # Strategy Scores (New mapping)
    ###########################################
    Score_Abs_Res_Strategy_01 = {}
    Score_Abs_Res_Strategy_02 = {}
    Score_Abs_Res_Strategy_03 = {}
    
    Score_Var_Res_Strategy_01 = {}
    Score_Var_Res_Strategy_02 = {}
    Score_Var_Res_Strategy_03 = {}
    
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_Res_Strategy_01[key,target] = [1 if Score_Heatmap[0][key,target][k] == max(Score_Heatmap[0][key,target]) else 0 for k in range(len(Score_Heatmap[0][key,target]))]
        Score_Abs_Res_Strategy_02[key,target] = [1 if Score_Heatmap[1][key,target][k] == max(Score_Heatmap[1][key,target]) else 0 for k in range(len(Score_Heatmap[1][key,target]))]
        Score_Abs_Res_Strategy_03[key,target] = [1 if Score_Heatmap[2][key,target][k] == max(Score_Heatmap[2][key,target]) else 0 for k in range(len(Score_Heatmap[2][key,target]))]
        
        Score_Var_Res_Strategy_01[key,target] = [1 if Score_Heatmap[3][key,target][k] == max(Score_Heatmap[3][key,target]) else 0 for k in range(len(Score_Heatmap[3][key,target]))]
        Score_Var_Res_Strategy_02[key,target] = [1 if Score_Heatmap[4][key,target][k] == max(Score_Heatmap[4][key,target]) else 0 for k in range(len(Score_Heatmap[4][key,target]))]
        Score_Var_Res_Strategy_03[key,target] = [1 if Score_Heatmap[5][key,target][k] == max(Score_Heatmap[5][key,target]) else 0 for k in range(len(Score_Heatmap[5][key,target]))]
    del key, target
    
    Score_Strategy02 = [Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03, Flag_Strategy]
    del Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03
    
    ###########################################
    # Strategy Scores (New mapping 02)
    ###########################################
    def Summarize(key,target, Var_ind, Ref = 'Absolute'):
        if Ref == 'Absolute':
            Temp = []
            for ind in Flag:
                Temp.append(Data[ind][key,target][Var_ind])
        elif Ref == 'Variability':
            Temp = []
            for ind in Flag:
                Temp.append(Data_Var[ind][key,target][Var_ind])
        
        Val = np.zeros((len(Temp),3))
        for i in range(Val.shape[0]):
            Val[i,0] = np.mean(Temp[i])
            Val[i,1] = np.min(Temp[i])
            Val[i,2] = np.max(Temp[i])
        return Val
    Score_Abs_01 = {}
    Score_Abs_02 = {}
    Score_Abs_03 = {}
    
    Score_Var_01 = {}
    Score_Var_02 = {}
    Score_Var_03 = {}
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_01[key,target] = Summarize(key,target,0,'Absolute') ## Abundance
        Score_Abs_02[key,target] = Summarize(key,target,2,'Absolute') ## Resilience
        Score_Abs_03[key,target] = Summarize(key,target,3,'Absolute') ## Persistence
        
        Score_Var_01[key,target] = Summarize(key,target,0,'Variability') 
        Score_Var_02[key,target] = Summarize(key,target,2,'Variability')
        Score_Var_03[key,target] = Summarize(key,target,3,'Variability')
    del key, target
    
    Score_Heatmap02 = [Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03, Flag]
    del Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03
    
    ###########################################
    # Strategy Scores (Raw mapping)
    ###########################################
    def Summarize(key,target, Var_ind, Ref = 'Absolute'):
        if Ref == 'Absolute':
            Temp = []
            for ind in Flag:
                Temp.append(Data[ind][key,target][Var_ind])
        elif Ref == 'Variability':
            Temp = []
            for ind in Flag:
                Temp.append(Data_Var[ind][key,target][Var_ind])
        
        Val = np.array(Temp)
        return Val
    Score_Abs_01 = {}
    Score_Abs_02 = {}
    Score_Abs_03 = {}
    
    Score_Var_01 = {}
    Score_Var_02 = {}
    Score_Var_03 = {}
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_01[key,target] = Summarize(key,target,0,'Absolute') ## Abundance
        Score_Abs_02[key,target] = Summarize(key,target,2,'Absolute') ## Resilience
        Score_Abs_03[key,target] = Summarize(key,target,3,'Absolute') ## Persistence
        
        Score_Var_01[key,target] = Summarize(key,target,0,'Variability') 
        Score_Var_02[key,target] = Summarize(key,target,2,'Variability')
        Score_Var_03[key,target] = Summarize(key,target,3,'Variability')
    del key, target
    
    Score_Heatmap03 = [Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03, Flag]
    del Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03
    
    return Score_Heatmap, Score_Strategy, Score_Strategy02, Score_Heatmap02, Score_Heatmap03

#%%
def get_Scores_2D(NetworkName, Flag):
    
    if len(set(Flag).difference({0,1,2,3,4})) != 0:
        return None, None
    
    Strategy_Name = ['System','Degree','Closeness','Betweenness','Random']
    Flag_Strategy = {k:Strategy_Name[k] for k in Flag}
    del Strategy_Name

    ###########################################
    # Get Data
    ###########################################
    Data = [
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_four.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_five.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_six.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_rand.pkl"))
            ]
    Data_All= [
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_four_All.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_five_All.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_six_All.pkl")),
            UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"{DATA_TYPE}_Data", 'Averaged', NetworkName, f"{NetworkName}-Data_rand_All.pkl"))
            ]
    Data_Var = []
    for i in range(len(Data_All)):
        Data_Var.append({})
        for key,value in Data_All[i].items():
            
            Var_Abundance = np.array([np.std(ensemble) for ensemble in list(Data_All[i][key]['Abundance'])])
            Var_Resilience = np.array([np.std(ensemble) for ensemble in list(Data_All[i][key]['Resilience'])])
            Var_Persistence = np.array([np.std(ensemble) for ensemble in list(Data_All[i][key]['Persistence'])])
            
            Data_Var[i][key] = [Var_Abundance, [], Var_Resilience, Var_Persistence]
    del i, key, value
    del Var_Abundance, Var_Resilience, Var_Persistence
    del Data_All

    ###########################################
    # Heatmap Scores
    ###########################################
    Score_Abs_01 = {}
    Score_Abs_02 = {}
    Score_Abs_03 = {}
    
    Score_Var_01 = {}
    Score_Var_02 = {}
    Score_Var_03 = {}

    def get_sampling():
        for i in [0,1,2]:
            for j in [2,5,8]:
                for k in [0,2]:
                    yield (i,j,k),0
    
    def Summarize(key,target, Var_ind, Intent = 'Min', Ref = 'Absolute'):
        if Ref == 'Absolute':
            Temp = []
            for ind in Flag:
                Temp.append(Data[ind-1][key,target][Var_ind])
        elif Ref == 'Variability':
            Temp = []
            for ind in Flag:
                Temp.append(Data_Var[ind-1][key,target][Var_ind])
        
        Val = np.zeros((len(Temp),len(Temp[0])-2))
        for i in range(1,len(Temp[0])-1):
            #x = [temp0[i],temp1[i],temp2[i],temp3[i],temp4[i]]
            x =[]
            for ind in range(len(Flag)):
                x.append(Temp[ind][i]) 
            if Intent == 'Max':
                y = max(x)
            elif Intent == 'Min':
                y = min(x)
            Val[:,i-1] = [1 if z == y else 0 for z in x]
        Val_sum = list(np.sum(Val, axis = 1) / Val.shape[1])
        return Val_sum
    
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_01[key,target] = Summarize(key,target,0,'Max','Absolute') ## Abundance
        Score_Abs_02[key,target] = Summarize(key,target,2,'Min','Absolute') ## Resilience
        Score_Abs_03[key,target] = Summarize(key,target,3,'Max','Absolute') ## Persistence
        
        Score_Var_01[key,target] = Summarize(key,target,0,'Min','Variability') 
        Score_Var_02[key,target] = Summarize(key,target,2,'Min','Variability')
        Score_Var_03[key,target] = Summarize(key,target,3,'Min','Variability')
    del key, target
    
    Score_Heatmap = [Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03, Flag]
    del Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03
    
    ###########################################
    # Strategy Scores
    ###########################################
    Score_Abs_Res_Strategy_01 = {}
    Score_Abs_Res_Strategy_02 = {}
    Score_Abs_Res_Strategy_03 = {}
    
    Score_Var_Res_Strategy_01 = {}
    Score_Var_Res_Strategy_02 = {}
    Score_Var_Res_Strategy_03 = {}
    
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_Res_Strategy_01[key,target] = Flag[Score_Heatmap[0][key,target].index(max(Score_Heatmap[0][key,target]))]
        Score_Abs_Res_Strategy_02[key,target] = Flag[Score_Heatmap[1][key,target].index(max(Score_Heatmap[1][key,target]))]
        Score_Abs_Res_Strategy_03[key,target] = Flag[Score_Heatmap[2][key,target].index(max(Score_Heatmap[2][key,target]))]
        
        Score_Var_Res_Strategy_01[key,target] = Flag[Score_Heatmap[3][key,target].index(max(Score_Heatmap[3][key,target]))]
        Score_Var_Res_Strategy_02[key,target] = Flag[Score_Heatmap[4][key,target].index(max(Score_Heatmap[4][key,target]))]
        Score_Var_Res_Strategy_03[key,target] = Flag[Score_Heatmap[5][key,target].index(max(Score_Heatmap[5][key,target]))]
    del key, target
    
    Score_Strategy = [Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03, Flag_Strategy]
    del Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03
    
    ###########################################
    # Strategy Scores (New mapping)
    ###########################################
    Score_Abs_Res_Strategy_01 = {}
    Score_Abs_Res_Strategy_02 = {}
    Score_Abs_Res_Strategy_03 = {}
    
    Score_Var_Res_Strategy_01 = {}
    Score_Var_Res_Strategy_02 = {}
    Score_Var_Res_Strategy_03 = {}
    
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_Res_Strategy_01[key,target] = [1 if Score_Heatmap[0][key,target][k] == max(Score_Heatmap[0][key,target]) else 0 for k in range(len(Score_Heatmap[0][key,target]))]
        Score_Abs_Res_Strategy_02[key,target] = [1 if Score_Heatmap[1][key,target][k] == max(Score_Heatmap[1][key,target]) else 0 for k in range(len(Score_Heatmap[1][key,target]))]
        Score_Abs_Res_Strategy_03[key,target] = [1 if Score_Heatmap[2][key,target][k] == max(Score_Heatmap[2][key,target]) else 0 for k in range(len(Score_Heatmap[2][key,target]))]
        
        Score_Var_Res_Strategy_01[key,target] = [1 if Score_Heatmap[3][key,target][k] == max(Score_Heatmap[3][key,target]) else 0 for k in range(len(Score_Heatmap[3][key,target]))]
        Score_Var_Res_Strategy_02[key,target] = [1 if Score_Heatmap[4][key,target][k] == max(Score_Heatmap[4][key,target]) else 0 for k in range(len(Score_Heatmap[4][key,target]))]
        Score_Var_Res_Strategy_03[key,target] = [1 if Score_Heatmap[5][key,target][k] == max(Score_Heatmap[5][key,target]) else 0 for k in range(len(Score_Heatmap[5][key,target]))]
    del key, target
    
    Score_Strategy02 = [Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03, Flag_Strategy]
    del Score_Abs_Res_Strategy_01, Score_Abs_Res_Strategy_02, Score_Abs_Res_Strategy_03, Score_Var_Res_Strategy_01, Score_Var_Res_Strategy_02, Score_Var_Res_Strategy_03
    
    ###########################################
    # Strategy Scores (New mapping 02)
    ###########################################
    def Summarize(key,target, Var_ind, Ref = 'Absolute'):
        if Ref == 'Absolute':
            Temp = []
            for ind in Flag:
                Temp.append(Data[ind-1][key,target][Var_ind])
        elif Ref == 'Variability':
            Temp = []
            for ind in Flag:
                Temp.append(Data_Var[ind-1][key,target][Var_ind])
        
        Val = np.zeros((len(Temp),3))
        for i in range(Val.shape[0]):
            Val[i,0] = np.mean(Temp[i])
            Val[i,1] = np.min(Temp[i])
            Val[i,2] = np.max(Temp[i])
        return Val
    Score_Abs_01 = {}
    Score_Abs_02 = {}
    Score_Abs_03 = {}
    
    Score_Var_01 = {}
    Score_Var_02 = {}
    Score_Var_03 = {}
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_01[key,target] = Summarize(key,target,0,'Absolute') ## Abundance
        Score_Abs_02[key,target] = Summarize(key,target,2,'Absolute') ## Resilience
        Score_Abs_03[key,target] = Summarize(key,target,3,'Absolute') ## Persistence
        
        Score_Var_01[key,target] = Summarize(key,target,0,'Variability') 
        Score_Var_02[key,target] = Summarize(key,target,2,'Variability')
        Score_Var_03[key,target] = Summarize(key,target,3,'Variability')
    del key, target
    
    Score_Heatmap02 = [Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03, Flag]
    del Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03
    
    ###########################################
    # Strategy Scores (Raw mapping)
    ###########################################
    def Summarize(key,target, Var_ind, Ref = 'Absolute'):
        if Ref == 'Absolute':
            Temp = []
            for ind in Flag:
                Temp.append(Data[ind-1][key,target][Var_ind])
        elif Ref == 'Variability':
            Temp = []
            for ind in Flag:
                Temp.append(Data_Var[ind-1][key,target][Var_ind])
        
        Val = np.array(Temp)
        return Val
    Score_Abs_01 = {}
    Score_Abs_02 = {}
    Score_Abs_03 = {}
    
    Score_Var_01 = {}
    Score_Var_02 = {}
    Score_Var_03 = {}
    for key,target in itertools.islice(get_sampling(),0,None,1):
        Score_Abs_01[key,target] = Summarize(key,target,0,'Absolute') ## Abundance
        Score_Abs_02[key,target] = Summarize(key,target,2,'Absolute') ## Resilience
        Score_Abs_03[key,target] = Summarize(key,target,3,'Absolute') ## Persistence
        
        Score_Var_01[key,target] = Summarize(key,target,0,'Variability') 
        Score_Var_02[key,target] = Summarize(key,target,2,'Variability')
        Score_Var_03[key,target] = Summarize(key,target,3,'Variability')
    del key, target
    
    Score_Heatmap03 = [Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03, Flag]
    del Score_Abs_01, Score_Abs_02, Score_Abs_03, Score_Var_01, Score_Var_02, Score_Var_03
    
    return Score_Heatmap, Score_Strategy, Score_Strategy02, Score_Heatmap02, Score_Heatmap03

#%%
for CASE_TO_SOLVE in ['Real_1D', 'Syn_1D', 'Syn_2D', 'Real_2D']:
    DATA_TYPE = CASE_TO_SOLVE
    CASE_TO_SOLVE_init(DATA_TYPE)
    M = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"M_{DATA_TYPE.split('_')[0]}.pkl"))
    NetworkName_Solved_df = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', f"Network_Solved_{DATA_TYPE.split('_')[0]}.pkl"))
    if DATA_TYPE.split('_')[0] == 'Syn':
        M = {k:M[k] for k in M.keys() if int(k.split('_')[2]) == 100}

    Heatmap01_All = {}
    Heatmap02_All = {}
    Heatmap03_All = {}
    UPDATE_HDATA = False
    for NetworkName in itertools.islice(NetworkName_Solved_df.NetworkName,0,None,1):
        if DATA_TYPE.split('_')[1] == '1D':
            Heatmap01_All[NetworkName] = get_Scores_1D(NetworkName, Flag = [0,1,2,3,4])
            Heatmap02_All[NetworkName] = get_Scores_1D(NetworkName, Flag = [0,1,4])
            Heatmap03_All[NetworkName] = get_Scores_1D(NetworkName, Flag = [1,2,3,4])
        elif DATA_TYPE.split('_')[1] == '2D':
            Heatmap01_All[NetworkName] = get_Scores_2D(NetworkName, Flag = [1,2,3,4]) ## 0,1,2,3,4 / 1,2,3,4
            Heatmap02_All[NetworkName] = get_Scores_2D(NetworkName, Flag = [1,2,4]) ## 0,1,4 / 1,2,4
            Heatmap03_All[NetworkName] = get_Scores_2D(NetworkName, Flag = [1,3,4]) ## 1,2,3,4 / 1,3,4
    if UPDATE_HDATA:
        PickleObj(Heatmap01_All, os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"Heatmap01_{DATA_TYPE}.pkl"))
        PickleObj(Heatmap02_All, os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"Heatmap02_{DATA_TYPE}.pkl"))
        PickleObj(Heatmap03_All, os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"Heatmap03_{DATA_TYPE}.pkl"))
    del NetworkName

#%%
def init_HeatmapData(Case, Start = 0, End = -1):
    if End == -1:
        End = Start + 1
    if Case == 1:
        Heatmap01_All = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"Heatmap01_{DATA_TYPE}.pkl"))
        return {k:Heatmap01_All[k][Start:End] for k in Heatmap01_All.keys()}
    if Case == 2:
        Heatmap02_All = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"Heatmap02_{DATA_TYPE}.pkl"))
        return {k:Heatmap02_All[k][Start:End] for k in Heatmap02_All.keys()}
    if Case == 3:
        Heatmap03_All = UnpickleObj(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database', f"Heatmap03_{DATA_TYPE}.pkl"))
        return {k:Heatmap03_All[k][Start:End] for k in Heatmap03_All.keys()}
