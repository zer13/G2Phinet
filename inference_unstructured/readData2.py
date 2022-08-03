import numpy as np
from scipy import io
import matplotlib.pyplot as plt

def one_hot(x,num_labels = 5):
    # x: 1d array
    num_cases = x.shape[0]
    x_oh = np.zeros((num_cases,num_labels))
    for idata in range(num_cases):
        x_oh[idata,np.int(np.round(x[idata]))] = 1
    return x_oh

def import_data2(num_types, test_set, id, choice_numx):
    # test_set: 1 to 5(5)
    # id: 0 to 4(3)
    data = io.loadmat('../data/data_unstructured.mat')
    dat_all = data['dat_all']
    gtp_all = data['gtp_all']
    num_points = data['num_points'].astype('int')
    num_points_types = data['num_points_types'].astype('int')

    if (test_set == 1):
        ids = [0,8,16,21,26]
    elif (test_set == 2):
        ids = [1,9,17,22,27]
    elif (test_set == 3):
        ids = [2,10,18,23,28]
    elif (test_set == 4):
        ids = [3,11,19,24,29]
    elif (test_set == 5):
        ids = [4,12,20,25,30]

    if (num_types == 4):
        del ids[3]

    isample = ids[id]

    gtp = gtp_all[isample:isample+1,0]-1    # 0 to 4(3)

    gtp_oh = one_hot(gtp)

    if (num_types == 4):
        gtp_oh = gtp_oh[:,[0,1,2,4]]

    npts = num_points[isample,0]
    npts_types = num_points_types[isample,:]        # (7,)
    npts_types_accu = np.array([np.sum(npts_types[:i]) for i in range(npts_types.shape[0]+1)])      # (8,)

    la = dat_all[isample,:num_points[isample,0],0]-1.0
    lc = dat_all[isample,:num_points[isample,0],1]-1.0
    sa = np.log(dat_all[isample,:num_points[isample,0],2]/1E3+1)/5
    sc = np.log(dat_all[isample,:num_points[isample,0],3]/1E3+1)/5

    sa = sa[:,None]
    sc = sc[:,None]

    # choose the data
    # version 1: 4 points on each curve
    num_x = choice_numx
    is_data = np.zeros_like(la,dtype=np.bool)
    for itype in range(num_points_types.shape[1]):
        istart = npts_types_accu[itype]
        num_points_itype = npts_types[itype]
        data_points_i = np.array([(j*(num_points_itype-1))//num_x for j in range(num_x+1)])+istart
        is_data[data_points_i] = True
        valid_a = np.logical_and(la>0.0,la<0.65)
        valid_c = np.logical_and(lc>0.0,lc<0.65)
        notvalid_a = np.logical_not(valid_a)
        notvalid_c = np.logical_not(valid_c)
        is_data[np.logical_or(notvalid_a, notvalid_c)] = False

    # choose the types
    is_type_i = np.zeros((la.shape[0],num_points_types.shape[1]),dtype=np.bool)
    for itype in range(num_points_types.shape[1]):
        istart = npts_types_accu[itype]
        iend = npts_types_accu[itype+1]
        is_type_i[istart:iend,itype] = True

    return (la, lc,     # (?, )
            sa, sc,     # (?, 2)
            is_data,            # (?,)
            gtp_oh,             # (1, num_types)
            is_type_i)    # (8,)


if __name__ == '__main__':
    import_data2(4, 5, 3)