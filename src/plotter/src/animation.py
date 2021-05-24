
import os
import numpy as np


path = '/home/auto/Desktop/autonomus_vehicle_project/project/deployement/vehicle_cpu/deployed_vehicle_code/src/plotter/data/camera/'
file_dir = 'd20_m05_y2021_hr20_min10_sec59/'
print (os.listdir(path+file_dir))

files = os.listdir(path+file_dir)

for filename in files:
#     data

    data = filename.split('.')[0]

    ## Note:: python 3 need encoding in latin1 
    globals()[data] = np.load(path+file_dir+filename,allow_pickle=True,encoding = 'latin1').item()
    
#     exec("%s = %{}" % (data,2))
#     data = np.load(path+file_dir+filename,allow_pickle=True)
#     print (filename.split('.')[0]) 



cam_pose_pure_x_hist    = pure_cam_pose['X']
cam_pose_pure_y_hist    = pure_cam_pose['Y']

cam_pose_fused_x_hist   = fused_cam_pose['X']
cam_pose_fused_y_hist   = fused_cam_pose['Y']

import matplotlib


import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation, rc
rc('animation')


rc_fonts = {
    "text.usetex": True,
    'text.latex.preview': True, # Gives correct legend alignment.
    'mathtext.default': 'regular',
    'text.latex.preamble': [r"""\usepackage{bm}"""],
}

# plt.rcParams.update({'pgf.preamble': r'\usepackage{amsmath}'})
plt.rcParams.update(rc_fonts)
fig, axs = plt.subplots(1, 1, figsize=(14,8))

# x,y = zip(cam_pose_pure_x_hist, cam_pose_pure_y_hist)

x = cam_pose_pure_x_hist 
y = cam_pose_pure_y_hist 

x1 = cam_pose_fused_x_hist 
y1 = cam_pose_fused_y_hist 


def animate(n):
    line, = plt.plot(x[:n], y[:n], color='g', label = r'Localization using visual sensor', linewidth =2 )
    line1, = plt.plot(x1[:n], y1[:n], color='b', label = r'Localization using visual-inertial sensor', linewidth =2 )
    
    plt.legend(loc = 'lower center')
    return line, line1

anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=10, blit=True, repeat = False)
plt.show()

print (path+file_dir+'myAnimation.gif')
# anim.save(path+file_dir+'myAnimation.gif', writer='imagemagick', fps=10)
