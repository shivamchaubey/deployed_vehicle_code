# Tools
import scipy.io as sio
import numpy as np


# Constructor For Dictionary Like Data Structure 
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


# Import Functions (fun) for .mat Files (MATLAB)
def fun_EEFIG(path):
    
    # Load File
    EEFIG_info = sio.loadmat(path)['info']
    EEFIG_info = EEFIG_info.T # transpose numpy

    properties =   ['EEF.TIG_counter',
                    'EEFIG_C',
                    'EEFIG_m',
                    'EEFIG_esm',
                    'EEFIG_is_anomaly',
                    'EEFIG_ignore',
                    'EEFIG_anomaly_points',
                    'EEFIG_beta',
                    'EEFIG_alpha',
                    'EEFIG_L_it',
                    'EEFIG_R_it',
                    'EEFIG_m_it',
                    'EEFIG_a_it',
                    'EEFIG_b_it',
                    'EEFIG_a',
                    'EEFIG_b',
                    'EEFIG_L',
                    'EEFIG_R',
                    'EEFIG_k',
                    'EEFIG_w',
                    'EEFIG_idx',
                    'EEFIG_Q',
                    'EEFIG_Q_it',
                    'EEFIG_gsum',
                    'EEFIG_a1_it',
                    'EEFIG_b1_it',
                    'EEFIG_L1_it',
                    'EEFIG_R1_it',
                    'EEFIG_m1_it',
                    'EEFIG_is_improved',
                    'EEFIG_A',
                    'EEFIG_B']

    EEFIG = {}

    for i, prop in enumerate(properties):
        prop = '_'.join(prop.split('_')[1:]) # delete first element
        data = np.squeeze(np.array(EEFIG_info[i])[0][0])
        EEFIG[prop] = data

    for i, key in enumerate(EEFIG.keys()):
        for j in  range(EEFIG[key].shape[0]):
            if (EEFIG[key][j].shape[0] == 1 and EEFIG[key][j].shape[1] == 1):
                EEFIG[key][j] = np.squeeze(EEFIG[key][j])
    
    return DotDict(EEFIG)
        

def fun_settings(path):

    # Load File
    EEFIG_model_vehicle = sio.loadmat(path)['settings']

    a = (EEFIG_model_vehicle.dtype)
    keys = list(a.fields.keys())
    
    settings = {}
    for key in keys:
        if (EEFIG_model_vehicle[key][0][0].shape[0] == 1 and EEFIG_model_vehicle[key][0][0].shape[1] == 1):
            settings[key] = np.squeeze(EEFIG_model_vehicle[key][0][0])
        else:
            settings[key] = EEFIG_model_vehicle[key][0][0]

    return DotDict(settings)


# state + commands + BUF Data
def data_conv_EEFIG(path):
    
    # Load File
    mat_data = sio.loadmat(path)['data_learn_EEFIG']

    py_data = {}
    x   = []
    u   = []
    BUF = []

    for i in range(len(mat_data[0])):
        data = np.squeeze(mat_data[0][i])
        if i>0:
            x.append(data[:3])
            u.append(data[3:5])
            BUF.append(data[-1])

    py_data['x'] = x
    py_data['u'] = u
    py_data['BUF'] = BUF

    return DotDict(py_data)


def data_conv_fuzzy(path):
    
    # Load File
    mat_data = sio.loadmat(path)['data_fuzz_ws_V2']

    py_data = {}
    A_vec = []
    B_vec = []
    XX = []
    UU =  []
    CURV_ref = [] 
    Ts = []
    
    for i in range(len(mat_data[0])):
        if i>0:
            data = np.squeeze(mat_data[0][i])
            A_vec.append(data[0])
            B_vec.append(data[1])
            XX.append(data[2])
            UU.append(data[3])
            CURV_ref.append(data[4])
            Ts.append(data[5])

    py_data['A_vec'] = A_vec
    py_data['B_vec'] = B_vec
    py_data['XX']    = XX
    py_data['UU']    = UU
    py_data['CURV_ref'] = CURV_ref
    py_data['Ts'] = Ts

    return DotDict(py_data)
