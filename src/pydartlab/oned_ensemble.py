import matplotlib.pyplot as plt
import dartlabplot as dlp
import matplotlib.gridspec as gridspec

# for notebook
# %matplotlib widget  

# Create figure
fig = plt.figure(figsize=(10, 8))
fig.canvas.header_visible = False

# Define GridSpec
gs = gridspec.GridSpec(4, 3, figure=fig)

# Create subplots of varying sizes and shapes
ax1 = fig.add_subplot(gs[0:3, 0:2]) 
ax1.set_xlim(-2, 4)
ax1.set_ylim(-0.4, 1)

mu = 1
sigma = 1

plotter = dlp.OnedEnsemble(fig, ax1, mu, sigma,[-2, 4], [-0.4, 0.8])

plotter.add_filter_options([0.65, 0.68, 0.2, 0.2])
plotter.add_update_button([0.65, 0.3, 0.2, 0.1])
plotter.add_mu_observation_textbox([0.75, 0.5, 0.1, 0.03])
plotter.add_sigma_observation_textbox([0.75, 0.45, 0.1, 0.03])
plotter.add_inflation_slider([0.70, 0.55, 0.12, 0.03])
plotter.add_inflation_toggle([0.65, 0.60, 0.15, 0.05])
