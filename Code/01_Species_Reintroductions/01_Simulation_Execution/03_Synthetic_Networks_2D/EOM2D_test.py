from EcologicalNetwork_Restoration_03 import *
#%%
def EOM2D(x, t, M0):
    alpha = 0.3
    h = 0.7
    mu = 1e-4
    gamma0 = 1.0
    p = 0.5
    bii = 1
    bij = 0.01
    
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
#%%
M_df = UnpickleObj(r'G:\My Drive\MyWork\Thesis\Sarth_Shared\Code\Codes\Final Codes\Ecological_Restoration_02\Syn_Data\M_PL_50_11_0\M_PL_50_11_0-Data_one.pkl')
#data_two = UnpickleObj(r'G:\My Drive\MyWork\Thesis\Sarth_Shared\Code\Codes\Final Codes\Ecological_Restoration_02\Syn_Data\M_PL_50_11_0\M_PL_50_11_0-Data_two.pkl')
#%%
t = np.linspace(0,100,101)
n,m = M_df.shape
x0 = np.ones(n+m)*1e-6
#%%
xl = odeint(EOM2D,x0,t,args=(M_df,))
#%%
plt.figure()
for i in range(xl.shape[1]):
    if i < M_df.shape[0]:
        plt.plot(np.arange(101),xl[:,i],'blue')
    else:
        plt.plot(np.arange(101),xl[:,i],'red')
#%%
M = np.array(
    [
     [1,0,0,0],
     [0,1,1,0],
     [0,1,1,0],
     [0,0,1,0],
     [0,0,0,0]
     ]
    )
M = pd.DataFrame(M)
t = np.linspace(0,100,101)
n,m = M.shape
x0 = np.ones(n+m)*0
xl = odeint(EOM2D,x0,t,args=(M,))
plt.figure()
for i in range(xl.shape[1]):
    plt.plot(np.arange(101),xl[:,i], label = str(i))
plt.legend(loc = 'upper left')
#%%
