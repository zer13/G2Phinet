import numpy as np
from scipy import io
import matplotlib.pyplot as plt

def mixup(x, y, num_new, alpha, num_mix):
    num_original = x.shape[0]
    x_new = np.zeros((num_new,x.shape[1],x.shape[2],x.shape[3]))
    y_new = np.zeros((num_new,y.shape[1]))
    mix_items = np.random.randint(num_original,size=(num_new,num_mix))
    mix_ratio = np.random.rand(num_new,num_mix)
    mix_ratio = mix_ratio/np.sum(mix_ratio,axis=1,keepdims=True)
    for inew in range(num_new):
        x_new[inew] = np.sum(mix_ratio[inew][:,None,None,None]*x[mix_items[inew]],axis=0)
        y_new[inew] = np.sum(mix_ratio[inew][:,None]*y[mix_items[inew]],axis=0)
    #print(mix_info_temp)
    w_new = alpha*np.ones((x_new.shape[0],1))
    return x_new, y_new, w_new

def oversampling(x, y, alpha):
    num_original = x.shape[0]
    num_types = y.shape[1]
    count = np.sum(y,axis=0)
    #print(count)
    maxnum = np.max(count)
    num_new_itype = np.int32(np.round(maxnum-count))
    num_new = np.sum(num_new_itype)
    x_new = np.zeros((num_new,x.shape[1],x.shape[2],x.shape[3]))
    y_new = np.zeros((num_new,y.shape[1]))
    w_new = alpha*np.ones((x_new.shape[0],1))
    temp = 0
    for itype in range(num_types):
        choice = y[:,itype] == 1
        x_i = x[choice]
        y_i = y[choice]
        num_i = num_new_itype[itype]
        newid = np.random.randint(x_i.shape[0],size=num_new_itype[itype])
        x_inew = x_i[newid]
        y_inew = y_i[newid]
        x_new[temp:temp+num_i] = x_inew
        y_new[temp:temp+num_i] = y_inew
        temp += num_i
    return x_new, y_new, w_new

def one_hot(x,num_labels = 5):
    # x: 1d array
    num_cases = x.shape[0]
    x_oh = np.zeros((num_cases,num_labels))
    for idata in range(num_cases):
        x_oh[idata,np.int(np.round(x[idata]))] = 1
    return x_oh

def import_data(num_types, test_set):

    y1, y2, xt, xz, gtp, num_cases_tot = read_biaxial_data()
    gtp_oh = one_hot(gtp)
    y = np.concatenate((y1[:,None,:,:],y2[:,None,:,:]),axis=1)

    allfalse = np.zeros(num_cases_tot,dtype=np.bool)
    istest = allfalse.copy()
    isnotdata = allfalse.copy()

    if (test_set == 1):
        istest[[0,8,16,21,26]] = True
    elif (test_set == 2):
        istest[[1,9,17,22,27]] = True
    elif (test_set == 3):
        istest[[2,10,18,23,28]] = True
    elif (test_set == 4):
        istest[[3,11,19,24,29]] = True
    elif (test_set == 5):
        istest[[4,12,20,25,30]] = True

    if (num_types == 5):
        pass
    elif (num_types == 4):
        isnotdata[21:26] = True
        gtp_oh = gtp_oh[:,[0,1,2,4]]

    isdata = np.logical_not(isnotdata)
    istrain = np.logical_and(isdata, np.logical_not(istest))
    istest = np.logical_and(istest, isdata)

    num_cases = y[istest].shape[0] + y[istrain].shape[0]
    num_test =  y[istest].shape[0]
    num_training = y[istrain].shape[0]
    num_trainplot = 6

    y_test = y[istest]
    y_train = y[istrain]
    y_trainplot = y_train[:num_trainplot]
    gtp_oh_test = gtp_oh[istest]
    gtp_oh_train = gtp_oh[istrain]
    gtp_oh_trainplot = gtp_oh_train[:num_trainplot]

    return xt, xz, y_train, y_trainplot, y_test, gtp_oh_train, gtp_oh_trainplot, gtp_oh_test, num_cases, num_training, num_trainplot, num_test

def read_biaxial_data():
    # read all original data from file
    # and normalize

    # read
    img_path = '../data/data.mat'
    data = io.loadmat(img_path)

    # extract and normalize
    saa = np.log(data['saa']+1.0)/5.0
    scc = np.log(data['scc']+1.0)/5.0
    lt = data['lt_all'][0,:]-1.0
    lz = data['lz_all'][0,:]-1.0
    LT, LZ = np.meshgrid(lt,lz)
    genetype = (data['genetype'][:,0]-1.0).astype('int32')

    num_cases_tot = saa.shape[0]

    # plot data 
    fig, axs = plt.subplots(nrows=num_cases_tot//4+1,ncols=4,num=123,figsize=(12.8,2.4*(num_cases_tot//4+1)))
    for idata in range(num_cases_tot):
        ax = axs[idata//4,idata%4]
        cs = ax.contour(LT,LZ,saa[idata,:,:])
        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_title('Data #'+str(idata+1)+' Label: '+str(genetype[idata]))
        ax.set_xlabel('lt')
        ax.set_ylabel('lz')
    plt.tight_layout()
    fig.savefig('contours_saa.png')
    plt.close(123)

    fig, axs = plt.subplots(nrows=num_cases_tot//4+1,ncols=4,num=123,figsize=(12.8,2.4*(num_cases_tot//4+1)))
    for idata in range(num_cases_tot):
        ax = axs[idata//4,idata%4]
        cs = ax.contour(LT,LZ,scc[idata,:,:])
        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_title('Data #'+str(idata+1)+' Label: '+str(genetype[idata]))
        ax.set_xlabel('lt')
        ax.set_ylabel('lz')
    plt.tight_layout()
    fig.savefig('contours_scc.png')
    plt.close(123)

    return saa, scc, lt, lz, genetype, num_cases_tot

if __name__ == "__main__":
    import_data(4,5)