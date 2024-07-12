import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, TextBox, Slider, CheckButtons
from dataclasses import dataclass
from increments import obs_increment_eakf, obs_increment_enkf, obs_increment_rhf
import math

def hide_negative_numbers(x, pos):
    """Hide negative numbers on an axis."""
    if x < 0:
        return ''
    return f'{x:.2f}'

@dataclass
class PosteriorStyle:
    linestyle: str = '-.'
    marker: str = 'g+'
    linecolor: str = 'g'

@dataclass
class PriorStyle:
    linestyle: str = '-'
    marker: str = 'b.'
    linecolor: str = 'b'
        
@dataclass
class ObservationStyle:
    linestyle: str = '--'
    marker: str = 'r*'
    linecolor: str = 'r'
    markersize: int = 12

class DartLabPlot:

    def __init__(self, fig, x_limits):
        self.fig = fig # do you want the figure?
        self.x_limits = x_limits # Helen do you want this?
        self.current_filter_selection = 'EAKF'
        self.clicked_points = []
        self.mu = 0
        self.sigma = 1

    def plot_observation(self, ax, mu, sigma):
   
        # Generate x values
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

        # Compute y values for the normal distribution
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        #ax.set_xlim(self.x_limits)
        ax.yaxis.set_major_formatter(FuncFormatter(hide_negative_numbers))
        ax.plot(x,y, linestyle=ObservationStyle.linestyle, color=ObservationStyle.linecolor)
        ax.plot(mu, 0, ObservationStyle.marker, markersize=ObservationStyle.markersize)
        ax.set_xlabel('Observed Quantity')
        ax.set_ylabel('Observation Likelihood')
        self.obs_ax = ax

    def add_filter_options(self, position): # [left, bottom, width, height]
        # Check if position is a list and has exactly four elements
        if not isinstance(position, list) or len(position) != 4:
            raise ValueError("position must be a four-element list")

        self.radio_ax = plt.axes(position) 
        self.radio = RadioButtons(self.radio_ax, ('EAKF', 'EnKF', 'RHF'))

        #self.radio_conn_id = radio.on_clicked(self.handle_radio)
        self.radio_conn_id = self.radio.on_clicked(self.handle_radio)
   
    def handle_radio(self, label):
        self.current_filter_selection = label
        if label == 'EAKF':
            # Handle Option 1
            print("EAKF selected")
        elif label == 'EnKF':
            # Handle Option 2
            print("EnKF selected")
        elif label == 'RHF':
            # Handle Option 3
            print("RHF selected")
        plt.draw()  # Update the plot if necessary

    def add_update_button(self, position): # [left, bottom, width, height]
        # Check if position is a list and has exactly four elements
        if not isinstance(position, list) or len(position) != 4:
            raise ValueError("position must be a four-element list")

        self.button_ax = plt.axes(position)
        self.button = Button(self.button_ax, 'Update Ensemble', color='lightblue', hovercolor='0.975')
        self.add_update_button_conn_id = self.button.on_clicked(self.update_ensemble)

    def add_mu_observation_textbox(self, position):  # [left, bottom, width, height]
        # Adjust the position of the mu textbox if necessary
        self.mu_textbox_ax = plt.axes(position)
        self.mu_textbox = TextBox(self.mu_textbox_ax, 'Observation ', initial=str(self.mu))
        # Connect the update function to the mu TextBox
        self.mu_conn_id = self.mu_textbox.on_submit(self.update_mu)

    def add_sigma_observation_textbox(self, position):  # [left, bottom, width, height]
        # Adjust the position of the sigma textbox if necessary
        self.sigma_textbox_ax = plt.axes(position)
        self.sigma_textbox = TextBox(self.sigma_textbox_ax, 'Observation SD ', initial=str(self.sigma))
        # Connect the update function to the sigma TextBox
        self.sigma_conn_id = self.sigma_textbox.on_submit(self.update_sigma)

    # Update the update function to read from the TextBoxes
    def update_mu(self, val):
        self.mu = float(self.mu_textbox.text)  # Convert text input to float
        self.obs_ax.clear()  # Clear the previous plot
        self.plot_observation(self.obs_ax, self.mu, self.sigma)  # Re-plot the observation with the new mu
        self.update_ensemble(None)  # Update the ensemble with the new observation
        plt.draw()  # Redraw the figure to reflect changes

    def update_sigma(self, val):
        self.sigma = float(self.sigma_textbox.text)  # Convert text input to float
        self.obs_ax.clear()  # Clear the previous plot
        self.plot_observation(self.obs_ax, self.mu, self.sigma)  # Re-plot the observation with the new sigma
        self.update_ensemble(None)  # Update the ensemble with the new observation
        plt.draw()  # Redraw the figure to reflect changes

    def add_inflation_slider(self, position):
        self.inflation_slider_ax = plt.axes(position, facecolor='lightgoldenrodyellow')
        self.inflation_slider = Slider(ax=self.inflation_slider_ax, label='Inflation ', valmin=1.0, valmax=5.0, valinit=1.0, valfmt='%1.2f')
        self.inf_conn_id = self.inflation_slider.on_changed(self.update_inflation) # connect the slider to the update_inflation function

    def update_inflation(self, val):
        self.inflation_factor = self.inflation_slider.val
        # Here you would typically apply the inflation factor to your model or data
        print(f"Inflation Factor: {self.inflation_factor}")  # Example action

    def add_inflation_toggle(self, position):
        self.check_ax = plt.axes(position)  # Adjust these coordinates as needed
        for spine in self.check_ax.spines.values():
            spine.set_visible(False)
        self.check_button = CheckButtons(self.check_ax, [' Apply inflation'], [False], check_props={'s': [400]}, frame_props={'s': [400]})
        self.inf_tog_conn_id = self.check_button.on_clicked(self.toggle_inflation)

    def toggle_inflation(self, label):
        if label.strip() == 'Apply inflation':
            # Here, implement what happens when inflation is toggled
            # For example, adjust a parameter or update the plot
            # This is a placeholder for the actual logic
            self.inflation_enabled = self.check_button.get_status()[0]
            print(f"Inflation enabled: {self.inflation_enabled}")
            # Update your plot or parameters here based on the inflation_enabled state

    def update_ensemble(self, event):
        print("does nothing")

class TwodEnsemble(DartLabPlot):

    def __init__(self, fig, ax1, ax2, ax3, ax4, mu, sigma, x_limits):
        super().__init__(fig, x_limits)
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.sigma = sigma
        self.mu = mu
        self.x_limits = x_limits
        self.plot_observation(self.ax4, self.mu, self.sigma)

        
        self.ax1.grid(True)
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 10)
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])

        # Get the limits of ax1
        self.x_limits_ax1 = ax1.get_xlim()
        self.y_limits_ax1 = ax1.get_ylim()

        # Set the limits of ax2, ax3 to match those of ax1
        self.ax2.grid(True)
        self.ax2.set_xlim(self.x_limits_ax1)
        self.ax2.set_ylim(-0.1, 1)
        self.ax2.set_yticklabels([])
        self.ax2.set_xlabel('Observed quantity')

        self.ax3.grid(True)
        self.ax3.set_xlim(-0.1, 1)
        self.ax3.set_ylim(self.y_limits_ax1)
        self.ax3.set_xticklabels([])
        self.ax3.set_ylabel('Unobserved state quantity')
   
        # set ax4 limits
        #self.ax4.set_ylim(bottom=-0.2)
        self.ax4.set_xlim(self.x_limits_ax1)

        # Connect the clicky event handler to the figure
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def update_ensemble(self, event):

        if len(self.clicked_points) < 2:
            print("Need at least 2 points to update ensemble.")
            return  # Exit the function early


        # need to clear plots, but leave original clicked points, and observations or replot them each time.
        # Clear existing plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Reapply necessary settings to the axes
        self.ax1.grid(True)
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 10)
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax2.set_xlabel('Observed quantity')
        self.ax2.grid(True)
        self.ax2.set_xlim(self.x_limits_ax1)
        self.ax2.set_ylim(-0.1, 1)
        self.ax2.set_yticklabels([])
        self.ax3.set_ylabel('Unobserved state quantity')
        self.ax3.grid(True)
        self.ax3.set_xlim(-0.1, 1)
        self.ax3.set_ylim(self.y_limits_ax1)
        self.ax3.set_xticklabels([])
        #self.ax4.set_ylim(bottom=-0.2)
        self.ax4.set_xlim(self.x_limits_ax1)
        self.ax4.yaxis.set_major_formatter(FuncFormatter(hide_negative_numbers))

        # Use zip and the * operator to unzip the list of tuples into two lists
        x_list, y_list = zip(*self.clicked_points)

        # Convert the tuples to lists 
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        zeros_array = np.zeros(len(x_list))

        # plot the best fit line
        # Calculate the best fit line parameters: slope (m) and intercept (b)
        m, b = np.polyfit(x_list, y_list, 1)  # 1 means linear (first-degree polynomial)

        # Generate x values for the line: from the minimum to the maximum x value in clicked_points
        x_line = np.linspace(0, 10, 100)  # 100 points for a smooth line

        # Calculate corresponding y values based on the line equation y = mx + b
        y_line = m * x_line + b

        self.ax1.plot(x_list, y_list, 'go')
        self.ax2.plot(x_list, zeros_array, 'g*')
        self.ax3.plot(zeros_array, y_list, 'g*')
        self.ax4.plot(x_list, zeros_array, 'g*')
        # Plot the best fit line on ax1
        self.ax1.plot(x_line, y_line, 'g-', label='Best Fit Line')  

        self.plot_observation(self.ax4, self.mu, self.sigma)

        # Calculate the observation increments
        if self.current_filter_selection == 'EAKF':
            obs_increments = obs_increment_eakf(np.array(x_list), self.mu, self.sigma**2)
        elif self.current_filter_selection == 'EnKF':
            obs_increments = obs_increment_enkf(np.array(x_list), self.mu, self.sigma**2)
        elif self.current_filter_selection == 'RHF':
            obs_increments = obs_increment_rhf(np.array(x_list), self.mu, self.sigma**2)
        
        updated_x = x_list + obs_increments
        
        covar = np.cov(x_list, y_list)
        state_inc = obs_increments * covar[0, 1] / covar[0, 0]
        updated_y = y_list + state_inc   
    
        self.ax1.plot(updated_x, updated_y, 'b*')  # Plot the updated points
        z = [0.1] * len(updated_x) 
        dz = 1 / (len(updated_x) +1)
        z_plot = [element + dz*i for i, element in enumerate(z)]
        obs_z = [x * -0.1 for x in z_plot]
        
        # plot updated points on marginals
        self.ax2.plot(updated_x, z_plot, 'b*')
        self.ax3.plot(z_plot, updated_y, 'b*')
        self.ax4.plot(updated_x, obs_z, 'b*')
        

        # plot increment lines
        for i, _ in enumerate(updated_x):
            self.ax1.plot([x_list[i], updated_x[i]], [y_list[i], updated_y[i]], 'b-')
            self.ax2.plot([x_list[i], updated_x[i]], [z_plot[i], z_plot[i]], 'b-')
            self.ax3.plot([z_plot[i], z_plot[i]], [y_list[i], updated_y[i]], 'b-')
            self.ax4.plot([x_list[i], updated_x[i]], [obs_z[i], obs_z[i]], 'b-')

        plt.draw()  # Redraw the figure to reflect changes

    def on_click(self, event):
        # Check if the click was on ax1
        if event.inaxes == self.ax1:
            # Append the clicked point (xdata, ydata) to the list
            self.clicked_points.append((event.xdata, event.ydata))
            print(f"Clicked at x={event.xdata}, y={event.ydata}")
            # Plot the click coordinates on ax1
            self.ax1.plot(event.xdata, event.ydata, 'go')  # 'ro' plots a red dot
            self.ax2.plot(event.xdata, 0, 'g*')
            self.ax3.plot(0, event.ydata, 'g*')
            self.ax4.plot(event.xdata, 0, 'g*')
            plt.draw()  # Update the plot with the new point

class OnedEnsemble(DartLabPlot):
    def __init__(self, fig, ax1, mu, sigma, x_limits, y_limits):
        super().__init__(fig, x_limits)
        self.ax1 = ax1
        self.sigma = sigma
        self.mu = mu
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.ax1.set_xlim(*self.x_limits)     # these are reset in update mu & update sigma
        self.ax1.set_ylim(*self.y_limits)
        self.ax1.axhline(y=-0.2, color='k', linestyle='-')
        self.plot_observation(self.ax1, self.mu, self.sigma)
        self.inflation_enabled = False
        self.inflation_factor = 1.0

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        # Check if the click was on ax1
        if event.inaxes == self.ax1:
            # Append the clicked point (xdata) to the list
            self.clicked_points.append((event.xdata))
            print(f"Clicked at x={event.xdata}")
            # Plot the click coordinates on ax1
            self.ax1.plot(event.xdata, 0, 'go')
            plt.draw() # Update the plot with the new point

    def update_ensemble(self, event):

        if len(self.clicked_points) < 2:
            print("Need at least 2 points to update ensemble.")
            return  # Exit the function early

        x_list = np.array(list(self.clicked_points))
        zeros_array = np.zeros(len(x_list))

        self.ax1.clear()
        self.ax1.set_ylim(*self.y_limits)
        self.ax1.set_xlim(*self.x_limits)
        self.ax1.axhline(y=-0.2, color='k', linestyle='-')

        self.plot_observation(self.ax1, self.mu, self.sigma)
        self.ax1.plot(x_list, zeros_array, 'go')

        self.plot_observation(self.ax1, self.mu, self.sigma)

        # Calculate the observation increments
        if self.current_filter_selection == 'EAKF':
            obs_increments = obs_increment_eakf(np.array(x_list), self.mu, self.sigma**2)
        elif self.current_filter_selection == 'EnKF':
            obs_increments = obs_increment_enkf(np.array(x_list), self.mu, self.sigma**2)
        elif self.current_filter_selection == 'RHF':
            obs_increments = obs_increment_rhf(np.array(x_list), self.mu, self.sigma**2)
        
        updated_x = x_list + obs_increments
        
        z_plot = [element - 0.1 for i, element in enumerate(zeros_array)]
        self.ax1.plot(updated_x, z_plot, 'b*')

        if self.inflation_enabled:
            # Apply inflation to the ensemble
            inf_x = (x_list - x_list.mean())* math.sqrt(self.inflation_factor) + x_list.mean()
            z_plot_inf_x = [element - 0.2 for i, element in enumerate(zeros_array)]
            self.ax1.plot(inf_x, z_plot_inf_x, 'go')
            # Calculate the observation increments for the inflated ensemble
            if self.current_filter_selection == 'EAKF':
                obs_increments = obs_increment_eakf(np.array(inf_x), self.mu, self.sigma**2)
            elif self.current_filter_selection == 'EnKF':
                obs_increments = obs_increment_enkf(np.array(inf_x), self.mu, self.sigma**2)
            elif self.current_filter_selection == 'RHF':
                obs_increments = obs_increment_rhf(np.array(inf_x), self.mu, self.sigma**2)

            updated_x_inf = inf_x + obs_increments

            z_plot_inf = [element - 0.3 for i, element in enumerate(zeros_array)]
            self.ax1.plot(updated_x_inf, z_plot_inf, 'y*')
 
            # plot inflation increment lines
            for i, _ in enumerate(inf_x):
                self.ax1.plot([inf_x[i], updated_x_inf[i]], [zeros_array[i]-0.2, zeros_array[i]-0.3], 'y-')
            


        # plot increment lines
        for i, _ in enumerate(updated_x):
            self.ax1.plot([x_list[i], updated_x[i]], [zeros_array[i], zeros_array[i]-0.1], 'b-')

       

        plt.draw()  # Redraw the figure to reflect changes
    

class Lorenz96(DartLabPlot):
    def __init__(self, fig, ax1, ax2, ax3, ax4):
        x_limits = [0, 10]
        super().__init__(fig, x_limits)
    