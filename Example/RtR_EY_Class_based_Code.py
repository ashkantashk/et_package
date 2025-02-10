import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

import pandas as pd
import numpy as np
import math

from datetime import timedelta, datetime
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from bokeh.layouts import column
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.models.tools import ZoomInTool, ZoomOutTool, UndoTool


from et_package import ET_Func

et_func = ET_Func(
    screen_size=27*25.4/(np.sqrt(2560**2+1440**2))*2560,          # in inches
    screen_ratio=16/9,
    viewing_distance=600,       # in mm
    screen_resolution=2560,    # width resolution in pixels
    freq=90                    # in Hz
)

# lone the repository to your local machine (or the same environment as your Python script):
# git clone https://lab.compute.dtu.dk/rtr/RtR-Data-Analysis.git

import sys
Directory = r"C:\Users\<user_name>\RtR-Data-Analysis-master\Tobii_data"
sys.path.append(Directory)
#%%
Pr = 1
fname = (Directory+rf"\sub-P00{Pr}_ses-S00{Pr}_task-Default_run-001_eeg.csv")
gazedata = pd.read_csv(fname)

events, proc_gazedata, all_options = et_func.find_fix(gazedata)
fixations = pd.DataFrame(events)

proc_gazedata = et_func.calc_velocity(proc_gazedata)

saccades = et_func.find_sac_from_fix(proc_gazedata, fixations)

forward, regress, newline = et_func.char_saccades(saccades)
#%%
from bokeh.plotting import output_notebook
output_notebook()
et_func.ETplot(proc_gazedata, fixations, 
                saccades=saccades[saccades.sclass=='S'], 
                nonsacm=saccades[(saccades.sclass!='S') & (saccades.sclass!='N')], 
                noise=saccades[saccades.sclass=='N'],
                title=("User %d all books" % (Pr)))
#%%
et_func.plot_saccades(saccades)
#%%
# Column to process
column_name = 'left_gaze_point_on_display_area_0'#'left_gaze_origin_in_user_coordinate_system_2'#'left_gaze_point_on_display_area_1'#


# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = gazedata[column_name].quantile(0.25)
Q3 = gazedata[column_name].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data to remove outliers
regularized_data = gazedata[(gazedata[column_name] >= lower_bound) & (gazedata[column_name] <= upper_bound)]
# regularized_data = gazedata.dropna(subset=[column_name])
# Plot the histogram
plt.subplot(1,2,1);gazedata[column_name].plot.hist(bins=100,edgecolor='blue',linewidth=0.5);plt.title(f'Hisogram of \n \"{column_name}\" \n with outliers');
plt.subplot(1,2,2);regularized_data[column_name].plot.hist(bins=100,edgecolor='blue',linewidth=0.5);plt.title(f'Histogram of \n \"{column_name}\" \n without outliers');
plt.subplots_adjust(wspace=0.5)  # wspace controls the width space between subplots
plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)  # Adjust padding and spacing

plt.show()

# Create the boxplot
plt.subplot(1,2,1);plt.boxplot(gazedata[column_name].dropna())
plt.title(f"Boxplot of \n \"{column_name}\" \n (with Outliers)")
plt.ylabel(column_name)
plt.subplot(1,2,2);plt.boxplot(regularized_data[column_name])
plt.title(f"Boxplot of \n \"{column_name}\" \n (Outliers Removed)")
plt.ylabel(column_name)
plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)  # Adjust padding and spacing
plt.show()
#%%

inside = []
outside = []
for dur in np.arange(0, 300):
    for amp in np.arange(0, 30, .1):
        if et_func.saccadic_prob(amp, dur)>=.33:
            inside.append((amp, dur))
        else:
            outside.append((amp, dur))

inside = pd.DataFrame(inside, columns=("amp", "dur"))
outside = pd.DataFrame(outside, columns=("amp", "dur"))

fig = plt.figure(figsize=(12,6))
plt.plot((0, 100), (21, 21+100*2.2), color='green')
plt.plot((0, 100), (21-11.1, 21-11.1*1.0+100*1.1), '--', color='green', linewidth=1)
plt.plot((0, 100), (21+11.1, 21+11.1*1.0+100*4.4), '--', color='green', linewidth=1)

plt.scatter(inside.amp, inside.dur, color='lime', marker='.', s=.1)
plt.scatter(outside.amp, outside.dur, color='orange', marker='.', s=.1)
plt.xlim((0, 30))
plt.ylim((0, 300))
plt.show()

print(inside)