import os
import scipy.io as sio

out_dir = 'out/Lymphocyte/pannuke_lymph_50_fix/'
mat_dir = out_dir + 'mat/'

for fn in os.listdir(mat_dir):
    file = os.path.join(mat_dir, fn)
    matfile = sio.loadmat(file)
    inst_uid = matfile['inst_uid']
    inst_map = matfile['inst_map']
    inst_type = matfile['inst_type']
    type_map = inst_map
    for uid in inst_uid:
        uid = uid[0] # this is the uid number, starting at 1
        mask = inst_map == uid
        type_map[mask] = inst_type[uid-1][0] # index inst_type by index, not uid number
    print('finished file: ', file)
    save_file = fn[:-5] + '_type_map.mat'
    sio.savemat(save_file, {'type_map': type_map})

    