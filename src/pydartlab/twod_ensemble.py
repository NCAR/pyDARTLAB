import matplotlib.pyplot as plt
import dartlabplot as dlp
import matplotlib.gridspec as gridspec

# Create figure
fig = plt.figure(figsize=(10, 8))
fig.canvas.header_visible = False

# Define GridSpec
gs = gridspec.GridSpec(5, 7, figure=fig)

# Create subplots of varying sizes and shapes
ax1 = fig.add_subplot(gs[0:3, 1:4]) # clicky main plot
ax2 = fig.add_subplot(gs[3, 1:4])   # Observered quantity
ax3 = fig.add_subplot(gs[0:3, 0])   # Unobserved state quantity
ax4 = fig.add_subplot(gs[2:4, 4:7]) # Marginal Distristribution of observation

ax4_position = ax4.get_position()
mu = 5
sigma = 2

plotter = dlp.TwodEnsemble(fig, ax1, ax2, ax3, ax4, mu, sigma, [0, 10])

plotter.add_filter_options([ax4_position.x0, 0.77, 0.2, 0.2])
plotter.add_update_button([0.8, 0.83, 0.15, 0.075])
plotter.add_mu_observation_textbox([ax4_position.x0+0.11, 0.7, 0.1, 0.03])
plotter.add_sigma_observation_textbox([ax4_position.x0+0.11, 0.65, 0.1, 0.03])

plt.tight_layout()
#plt.show() # this plots two another figure when used with %run twod_ensemble.py
