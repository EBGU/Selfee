import numpy as np
from scipy import stats

def RCAF(y_ture,y_predict):
    idx_sort = np.argsort(y_ture)
    sorted_true = y_ture[idx_sort]
    sorted_predict = y_predict[idx_sort]
    value,ind,num = np.unique(sorted_true,return_index=True,return_counts=True)
    ind = np.append(ind,len(sorted_true))
    M = int(np.ceil(num.max()/num.min())) #sampling will be repeated M times
    sample_num = num.min()
    inv_r2 = []
    p_value = []
    for _ in range(M):
        #sampling
        sample_id = []
        for i in range(len(value)):
            sample_id.append(np.random.choice(np.arange(ind[i],ind[i+1]),sample_num,False))
        sample_id = np.concatenate(sample_id)
        sampled_true = sorted_true[sample_id]
        sampled_predict = sorted_predict[sample_id]
        r,p = stats.pearsonr(sampled_true,sampled_predict)
        #r,p = stats.kendalltau(sampled_true,sampled_predict)
        inv_r2.append(1/r**2)
        p_value.append(p)
    inv_r2 = np.array(inv_r2).mean()
    corr = 1/inv_r2**0.5
    p_value = np.array(p_value).mean()
    return corr,p_value

#RCAF(np.random.randint(0,7,100),np.random.randint(0,7,100))