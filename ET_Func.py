##  A Class for all ET functions
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
import I2MC
from bokeh.layouts import column
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, RangeTool, LinearAxis, Range1d
from bokeh.models.tools import ZoomInTool, ZoomOutTool, UndoTool, BoxZoomTool, ResetTool
from datetime import timedelta

output_notebook()

class ET_Func:

    """
    Example Usage:
    --------------
    from ET_Func import ET_Func
    import pandas as pd

    # Initialize ET_Func with screen parameters
    et_func = ET_Func(screen_size=24, screen_ratio=16/9, viewing_distance=60, screen_resolution=1920, freq=60)

    # Calculate px2deg
    px2deg = et_func.calculate_px2deg()
    print(f"Pixels per degree: {px2deg}")

    # Load gaze data
    gaze_data = pd.read_csv("gaze_data.csv")

    # Detect fixations
    fixations = et_func.find_fix(gaze_data)
    print(f"Detected fixations: {fixations}")
    """

    def __init__(self, screen_size, screen_ratio, viewing_distance, screen_resolution, freq):
        """
        Constructor to initialize screen properties and other essential variables.
        
        Parameters:
        screen_size (float): Physical screen width in inches.
        screen_ratio (float): Screen aspect ratio (e.g., 16/9).
        viewing_distance (float): Viewing distance in cm.
        screen_resolution (int): Screen width resolution in pixels (e.g., 1920 for a 1920x1080 screen).
        freq (float): Frequency in Hz.
        """
        self.SCREEN_SIZE = screen_size
        self.SCREEN_RATIO = screen_ratio
        self.VIEWING_DISTANCE = viewing_distance
        self.SCREEN_RESOLUTION = screen_resolution
        self.FREQ = freq
        self.px2deg = self.calculate_px2deg()
    
    def calculate_px2deg(self):
        """
        Calculate the px2deg (pixels per degree) conversion factor based on initialized variables.
        
        Returns:
        float: The calculated px2deg value.
        """
        # Calculate screen height in inches and convert to cm
        height_in_inches = self.SCREEN_SIZE / self.SCREEN_RATIO
        height_in_cm = height_in_inches * 2.54
        
        # Calculate pixel pitch (cm per pixel)
        pixel_pitch = height_in_cm / (self.SCREEN_RESOLUTION / self.SCREEN_RATIO)

        # Calculate px2deg using the visual angle formula
        self.px2deg = (math.atan2(.5 * self.SCREEN_SIZE, self.VIEWING_DISTANCE))/np.pi*180 / (.5 * self.SCREEN_RESOLUTION)
        return self.px2deg


    def find_fix(self, gazedata):
        """
        Preprocess gaze data and calculate fixations using I2MC.
        
        Parameters:
        gazedata (pd.DataFrame): A dataframe containing eye-tracking data with columns:
            - device_time_stamp
            - left_gaze_point_on_display_area_0
            - left_gaze_point_on_display_area_1
            - right_gaze_point_on_display_area_0
            - right_gaze_point_on_display_area_1
            - left_pupil_diameter
            - left_pupil_validity
            - right_pupil_diameter
            - right_pupil_validity
        
        Returns:
        dict: The output of the I2MC fixation detection.
        """
        options = {
            "xres": self.SCREEN_RESOLUTION,
            "yres": self.SCREEN_RESOLUTION / self.SCREEN_RATIO,
            "freq": self.FREQ,
            "missingx": np.nan,
            "missingy": np.nan,
            "scrSz": (self.SCREEN_SIZE, self.SCREEN_SIZE / self.SCREEN_RATIO),
            "disttoscreen": self.VIEWING_DISTANCE,
            "downsampFilter": False
        }

        
        renamed_gazedata = pd.DataFrame()

        renamed_gazedata["time"] = gazedata.device_time_stamp / 1e3 # must be msec. apparently...

        # I2MC likes pixels, it seems...
        renamed_gazedata["L_X"]  = ((gazedata.left_gaze_point_on_display_area_0 * self.SCREEN_RESOLUTION))
        renamed_gazedata["L_Y"]  = ((gazedata.left_gaze_point_on_display_area_1 * self.SCREEN_RESOLUTION/self.SCREEN_RATIO))

        renamed_gazedata["R_X"]  = ((gazedata.right_gaze_point_on_display_area_0 * self.SCREEN_RESOLUTION))
        renamed_gazedata["R_Y"]  = ((gazedata.right_gaze_point_on_display_area_1 * self.SCREEN_RESOLUTION/self.SCREEN_RATIO))

        renamed_gazedata["L_Pz"] = gazedata.left_pupil_diameter
        renamed_gazedata.loc[gazedata.left_pupil_validity<0.5, "L_Pz"] = np.nan
        renamed_gazedata["R_Pz"] = gazedata.right_pupil_diameter
        renamed_gazedata.loc[gazedata.right_pupil_validity<0.5, "R_Pz"] = np.nan

        return I2MC.I2MC(renamed_gazedata, options)

    def calculate_saccade_probability(self,amp, dur):
        # Parameters
        a = 21  # ms
        b_min = 1.1  # ms/deg
        b_max = 4.4  # ms/deg
        max_duration = 100  # ms

        # Expected duration
        expected_dur = a + 2.2 * amp

        # Tolerance range
        min_dur = a + b_min * amp
        max_dur = a + b_max * amp

        # Adjust for sampling frequency
        time_res = 1000 / self.FREQ
        min_dur_adj = min_dur - time_res
        max_dur_adj = max_dur + time_res

        # Check if duration is within acceptable range
        if dur > max_duration or dur < min_dur_adj or dur > max_dur_adj:
            return 0.0

        # Calculate deviation
        deviation = abs(dur - expected_dur) / expected_dur

        # Calculate sigma
        sigma = (max_dur_adj - min_dur_adj) / 6

        # Calculate goodness value
        goodness = np.exp(-((dur - expected_dur) ** 2) / (2 * sigma ** 2))

        return goodness

    def saccadic_prob(self, amp, dur):
        """
        Calculate the "probability" of the saccade being a real saccade.
        
        It uses a number of heuristics, including Carpenter's 1988 standard model of:
        
            duration = a + b * amp = 21 mS + amp * 2.2 mS/deg

        Returns a "goodness value" ("probability") that is close to 1 when the real duration is within and values approaching 0 for outliers.

        Given the large variability in saccadic peak velocities (which is inversely realated to the duration) in the normal population,
        we consider the range of b to be normal from 1.1 to 4.4 (corresponding to peak velocities above 700 deg/sec to approx half).

        Saccades above 100 mS are generally discarded.

        In addition we compensate for measurement errors from the sampling frequency.

        See e.g. 

        * Carpenter, R. H. (1988): "Movements of the eyes".
        * Duchowski, A., et al (2017): "An inverse-linear logistic model of the main sequence"
        * Guadron (2022): "Speed‐accuracy tradeoffs influence the main sequence of saccadic eye movements"
        * Hélène Devillez et al (2020): "The bimodality of saccade duration during the exploration of visual scenes"
        * Agostino Gibaldi and Silvio P. Sabatini (2020): "The saccade main sequence revised: A fast and repeatable tool for oculomotor analysis"

        amp is the amplitude in degrees
        dur is the duration in milliseconds
        freq is the sample frequency in Hz

        """
        edur = 21 + 2.2*amp

        mindur = 21 + 1.1*amp - 1000./self.FREQ*1.5
        maxdur = min((21 + 4.4*amp + 1000./self.FREQ*1.5), 100)

        if (dur >= mindur) and (dur <= maxdur):
            return 1 # - min(abs(dur-edur)/edur, 1)
        else:
            return 0

        # return (dur - (21 + 2.2*amp*px2deg))/(21 + 2.2*amp*px2deg)


    def find_sac_from_fix(self, gazedata, fixations):
        """
        Find saccades from fixation events in the gazedata

        Parameters
        ----------
        gazedata : pandas dataframe
            This dataframe holds the gazedata and must contain the following columns:
            * time : the relative time of the gaze data in mS
            * L_X  : the left eye X coordinate in arbitrary units
            * L_Y  : the left Y coordinate in arbitrary units
            * R_X  : the right X coordinate in arbitrary units
            * R_Y  : the right Y coordinate in arbitrary units
            * average_X : the average X coordinates in arbitrary units
            * average_Y : the average Y coordinates in arbitrary units
        fixations : pandas dataframe
            This dataframe holds the previously detected fixations and must contain the following columns:
            * start  : the start of the fixation as an index into the gazedata dataframe
            * end    : the end of the fixation as an index into the gazedata dataframe
            * startT : the start of a fixation in relative time on the same scale as above, in  mS
            * endT   : the end of the fixation on the same scale
            * xpos   : the central X coordinate in the same units as above
            * ypos   : the central Y coordinate in the same units as above
        """
        """
        Find saccades from fixation events in the gaze data.
        """
        if self.px2deg is None:
            raise ValueError("px2deg has not been calculated. Please call calculate_px2deg first.")
    
        # Loop through all fixation pairs and generate the corresponding saccades (or noise)
        saccades = []
        for i in range(len(fixations)-1):
            # first find the basic structure of the fixation
            start = (int )(fixations.iloc[i].end+1)
            end   = (int )(fixations.iloc[i+1].start-1)

            startT = fixations.iloc[i].endT
            endT   = fixations.iloc[i+1].startT

            xpos = fixations.iloc[i].xpos
            ypos = fixations.iloc[i].ypos

            xdel = fixations.iloc[i+1].xpos - fixations.iloc[i].xpos
            ydel = fixations.iloc[i+1].ypos - fixations.iloc[i].ypos

            ang = math.atan2(xdel, -ydel)/np.pi*180.

            dur = endT - startT
            amp = np.sqrt(xdel**2 + (ydel)**2)

            sprob = self.saccadic_prob(amp*self.px2deg, dur)

            # the following could also have been calculated from the average missing column
            xpts = gazedata.iloc[start:end+1].average_X.values
            ypts = gazedata.iloc[start:end+1].average_Y.values

            complete = not np.isnan(np.sum(xpts) + np.sum(ypts))

            if sprob >= .33: # TODO: magic number (i.e. we need to argue why this value(!))
                if complete:
                    sclass = 'S' # proper saccade
                else:
                    sclass = 'I' # likely a proper saccade but incomplete with a few missing points
            else:
                if complete:
                    sclass = 'P' # Non-saccadic movemement
                else:
                    sclass = 'N' # noise

            saccades.append((start, end, startT, endT, xpos, ypos, xdel, ydel, ang, dur, amp, sprob, sclass))

        saccades = pd.DataFrame(saccades, columns=('start', 'end', 'startT', 'endT', 'xpos', 'ypos', 'xdel', 'ydel', 'ang', 'dur', 'amp', 'sprob', 'sclass'))
        
        return saccades
    
    def plot_saccades(self, saccades,px2deg=None):
        if px2deg is None:
            if self.px2deg is None:
                raise ValueError("px2deg has not been calculated. Please call calculate_px2deg first.")
            px2deg = self.px2deg
        
        fig = plt.figure(figsize=(12,8))

        s = saccades[saccades.sclass=='S'] 
        plt.scatter(s.amp*px2deg, s.dur, 20, marker='^', c=np.clip(s.sprob, 0, 1), cmap='cool', label='Saccade')
        plt.colorbar(label='Estimated probability')

        s = saccades[saccades.sclass=='I'] 
        plt.scatter(s.amp*px2deg, s.dur, 20, marker='*', c='green', label='Incomplete Saccade')

        s = saccades[saccades.sclass=='P'] 
        plt.scatter(s.amp*px2deg, s.dur, 10, marker='.', c='lime', label='Non-saccadic movement')

        s = saccades[saccades.sclass=='N'] 
        plt.scatter(s.amp*px2deg, s.dur, 10, marker='.', c="orange", label='Noise/Data loss')

        plt.plot((0, 2500*px2deg), (21, 21+2500*px2deg*2.2), color='green')
        plt.plot((0, 2500*px2deg), (21-11, 21-11.1*1.5+2500*px2deg*1.1), '--', color='green', linewidth=1)
        plt.plot((0, 2500*px2deg), (21+11, 21+11.1*1.5+2500*px2deg*4.4), '--', color='green', linewidth=1)

        plt.xlabel("Amplitude (deg)")
        plt.ylabel("Duration (mS)")
        plt.ylim((0, 300))
        plt.xlim((0, 30))
        plt.legend(loc='upper right')
        plt.show()
        plt.close()


    def calc_velocity(self, gazedata):
        '''
        Estimate the velocity for the gazedata

        Parameters
        ----------
        gazedata : pandas dataframe
            This dataframe holds the gazedata and must contain the following columns:
            * time : the relative time of the gaze data in mS
            * L_X  : the left eye X coordinate in arbitrary units
            * L_Y  : the left Y coordinate in arbitrary units
            * R_X  : the right X coordinate in arbitrary units
            * R_Y  : the right Y coordinate in arbitrary units
            * average_X : the average X coordinates in arbitrary units
            * average_Y : the average Y coordinates in arbitrary units
        
        Returns a copy of the same dataframe with an added column 'velocity' which is an estimate of the 
        instantaneous velocity for each timepoint

        See:

            * Nystrom and Holmquist (2010): An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking data
            * Duchowski (2017): Eye Tracking Methodology (pp 261-262)
            * Raju (2023): FILTERING EYE-TRACKING DATA FROM AN EYELINK 1000: COMPARING HEURISTIC, SAVITZKY-GOLAY, IIR AND FIR DIGITAL FILTERS

        '''

        gazedata["L_vel"] = np.sqrt(np.square(savgol_filter(gazedata.L_X, 5, 2, 1, mode="nearest")) + np.square(savgol_filter(gazedata.L_Y, 5, 2, 1, mode="nearest")))*self.FREQ
        gazedata["R_vel"] = np.sqrt(np.square(savgol_filter(gazedata.R_X, 5, 2, 1, mode="nearest")) + np.square(savgol_filter(gazedata.R_Y, 5, 2, 1, mode="nearest")))*self.FREQ

        return gazedata


    def char_saccades(self, saccades, px2deg=None):
        
        if px2deg is None:
            if self.px2deg is None:
                raise ValueError("px2deg has not been calculated. Please call calculate_px2deg first.")
            px2deg = self.px2deg

        forward = np.sum([saccades.xdel*px2deg>=0])
        regress = np.sum([(saccades.xdel*px2deg<0) & (saccades.xdel*px2deg>=-6)])
        newline = np.sum([saccades.xdel*px2deg<-6])

        print(f"READ: There are {len(saccades)} saccades, made up of {forward} forward, {regress} regressions and {newline} newlines")

        return forward, regress, newline
    
    def ETplot(self, gazedata, fixations=None, saccades=None, nonsacm=None, noise=None, px2deg=None, title=None):
        """
        This function creates an interactive plot of eye tracking data.

        It can be useful for inspecting and analysing eye tracking data

        Parameters
        ----------
        gazedata : pandas dataframe
            This is a pandas dataframe that must contain the following columns:
            * time : the relative time of the gaze data in mS
            * L_X, L_Y  : the X and Y coordinate in arbitrary units (e.g. pixels) from the left eye
            * L_X, L_Y  : the X and Y coordinate in arbitrary units from the right eye
            * L_vel, R_vel : Derived velocity; plotted if given
            * L_Pz, R_Pz : Pupil size; plotted if given

        fixations : pandas dataframe (or None)
            This is a pandas dataframe that must contain the following columns:
            * startT : the start of a fixation in relative time on the same scale as above, in  mS
            * endT   : the end of the fixation on the same scale
            * xpos   : the central X coordinate in the same units as above
            * ypos   : the central Y coordinate in the same units as above

        saccades : A pandas dataframe containing the saccsdes, with the following attributes
            * startT : the start of a saccade in relative time on the same scale as above, in  mS
            * endT   : the end of the saccade on the same scale
            * xpos   : the starting X coordinate in the same units as above
            * ypos   : the starting Y coordinate in the same units as above

        nonsacm : A pandas dataframe containing the non-saccadic movements, with the following attributes
            * startT : the start of a saccade in relative time on the same scale as above, in  mS
            * endT   : the end of the saccade on the same scale

        noise : A pandas dataframe containing the noise periods, with the following attributes
            * startT : the start of a saccade in relative time on the same scale as above, in  mS
            * endT   : the end of the saccade on the same scale

        px2deg : a conversion from the arbitraty units (pixels?) to degree

        title : the title to go on the plot

        """
        if px2deg is None:
            if self.px2deg is None:
                raise ValueError("px2deg has not been calculated. Please call calculate_px2deg first.")
            px2deg = self.px2deg
        
        matplotlib.use('module://matplotlib_inline.backend_inline')
    
        # find the starting time and make the plot relative to this
        plotS = gazedata.time.min()

        # convert the gaze data to an easy-to-handle format (for bokeh)
        gaze = ColumnDataSource(pd.DataFrame({
            'time': pd.to_timedelta(gazedata.time-plotS, unit='ms'),
            'L_X': gazedata.L_X*px2deg,
            'L_Y': gazedata.L_Y*px2deg,
            'R_X': gazedata.R_X*px2deg,
            'R_Y': gazedata.R_Y*px2deg,
        }))

        plot_min = gazedata[["L_X", "L_Y", "R_X", "R_Y"]].min(skipna=True).min()*px2deg
        plot_max = gazedata[["L_X", "L_Y", "R_X", "R_Y"]].max(skipna=True).max()*px2deg

        # Plot the raw gaze data
        p = figure(x_axis_type='datetime', tools=["xpan", "ypan", "pan"],
                title=("%s (plot start at time h:m:s=%s (%.3f seconds))" % ((title if title else "Eye Tracking Data"), timedelta(milliseconds=plotS), plotS)), 
                x_axis_label="Time (s)", y_axis_label="Position (deg)", 
                height=800, width=1400,
                x_range=(0,30000), y_range=(plot_min, plot_max))
        p.line('time', 'L_X', source=gaze, legend_label="L_x", line_color="cyan")
        p.line('time', 'L_Y', source=gaze, legend_label="L_y", line_color="orange")
        p.line('time', 'R_X', source=gaze, legend_label="R_x", line_color="cyan", line_dash='dashed')
        p.line('time', 'R_Y', source=gaze, legend_label="R_y", line_color="orange", line_dash='dashed')

        # plot the detected events
        if fixations is not None:
            fixev = ColumnDataSource(pd.DataFrame({
                'startT' : pd.to_timedelta(fixations.startT-plotS, unit='ms'),
                'endT'   : pd.to_timedelta(fixations.endT-plotS, unit='ms'),
                'xpos'   : fixations.xpos*px2deg,
                'ypos'   : fixations.ypos*px2deg,
            }))

            p.segment(x0='startT', y0='xpos', x1='endT', y1='xpos', source=fixev, line_width=5, color='blue', line_alpha=.5, legend_label="x (fix)")
            p.segment(x0='startT', y0='ypos', x1='endT', y1='ypos', source=fixev, line_width=5, color='red',  line_alpha=.5, legend_label="y (fix)")

        if saccades is not None:
            sacev = ColumnDataSource(pd.DataFrame({
                'startT' : pd.to_timedelta(saccades.startT-plotS, unit='ms'),
                'endT'   : pd.to_timedelta(saccades.endT-plotS, unit='ms'),
                'xpos'   : saccades.xpos*px2deg,
                'ypos'   : saccades.ypos*px2deg,
            }))

            p.segment(x0='startT', y0='xpos', x1='endT', y1='xpos', source=sacev, line_width=30, color='green', line_alpha=.3, legend_label="saccade")
            p.segment(x0='startT', y0='ypos', x1='endT', y1='ypos', source=sacev, line_width=30, color='green', line_alpha=.3)

        if nonsacm is not None:
            nsmev = ColumnDataSource(pd.DataFrame({
                'startT' : pd.to_timedelta(nonsacm.startT-plotS, unit='ms'),
                'endT'   : pd.to_timedelta(nonsacm.endT-plotS, unit='ms'),
            }))

            p.segment(x0='startT', y0=0, x1='endT', y1=0, source=nsmev, line_width=20, color='brown', line_alpha=.3, legend_label="Non-saccadic movement")

        if noise is not None:
            noiev = ColumnDataSource(pd.DataFrame({
                'startT' : pd.to_timedelta(noise.startT-plotS, unit='ms'),
                'endT'   : pd.to_timedelta(noise.endT-plotS, unit='ms'),
            }))

            p.segment(x0='startT', y0=0, x1='endT', y1=0, source=noiev, line_width=20, color='grey', line_alpha=.3, legend_label="Noise")


        if 'L_Pz' in gazedata.columns:
            pup = ColumnDataSource(pd.DataFrame({
                'time': pd.to_timedelta(gazedata.time-plotS, unit='ms'),
                'L_Pz': gazedata.L_Pz,
                'R_Pz': gazedata.R_Pz,
            }))

            p.extra_y_ranges['pupil'] = Range1d(0, 10)
            p.line('time', 'L_Pz', source=pup, legend_label="L_Ps", line_color="black", y_range_name="pupil")
            p.line('time', 'R_Pz', source=pup, legend_label="R_Ps", line_color="black", line_dash="dashed", y_range_name="pupil")

            ax2 = LinearAxis(y_range_name="pupil", axis_label="Pupil Size (mm)")
            p.add_layout(ax2, 'left')

        if 'L_vel' in gazedata.columns:
            vel = ColumnDataSource(pd.DataFrame({
                'time': pd.to_timedelta(gazedata.time-plotS, unit='ms'),
                'L_vel': gazedata.L_vel*px2deg,
                'R_vel': gazedata.R_vel*px2deg
            }))

            # p.extra_y_ranges['velocity'] = Range1d(gazedata.L_vel.min(skipna=True)*px2deg, gazedata.L_vel.max(skipna=True)*px2deg)
            p.extra_y_ranges['velocity'] = Range1d(0, 1000)
            p.line('time', 'L_vel', source=vel, legend_label="L_vel", line_color="green", y_range_name="velocity")
            p.line('time', 'R_vel', source=vel, legend_label="R_vel", line_color="green", line_dash="dashed", y_range_name="velocity")

            ax3 = LinearAxis(y_range_name="velocity", axis_label="Velocity (deg/s)")
            p.add_layout(ax3, 'left')


        select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=80, width=1400, y_range=p.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
        
        range_tool = RangeTool(x_range=p.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

        select.line('time', 'L_X', source=gaze, legend_label="L_x", line_color="cyan")
        select.line('time', 'L_Y', source=gaze, legend_label="L_y", line_color="orange")

        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        p.add_tools(BoxZoomTool(), ZoomInTool(), ZoomOutTool(), UndoTool(), ResetTool())

        show(column(p, select))

        return None
