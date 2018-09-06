import matplotlib.pyplot as plt  
import numpy as np
import librosa

def plot_spec_sbs(specs, size, aspect, tick_scale = 1, vmin = None):
	fig, axis = plt.subplots(len(specs), 1, sharex=True, sharey=True)

	max_value = 0
	for spec in specs:
		local_max = np.amax(spec)
		if local_max > max_value:
			max_value = local_max

	if not vmin:
		vmin = 0

	for i,spec in enumerate(specs):
		last_plot = axis[i].imshow(spec.T, vmin=vmin, vmax=max_value, interpolation=None, cmap=plt.cm.magma)
		axis[i].set_aspect(aspect)
		axis[i].set_ylabel("Hop (t)")

	fig.set_size_inches(size[0],size[1])
	fig.colorbar(last_plot, ax=axis, pad=0.02, label="Amplitude  (log10)")

	print([str(item) for item in axis[-1].get_xticklabels()])
	print([])

	labels = [str(int(item * tick_scale)) for item in axis[-1].get_xticks()]
	print(labels)
	axis[-1].set_xticklabels(labels)
	axis[-1].set_xlabel("Hz")

	return fig

def plot_spec_sbs_vertical(specs, size, aspect, tick_scale = 1, vmin=None):
	fig, axis = plt.subplots(1, len(specs), sharex=True, sharey=True)

	max_value = 0
	for spec in specs:
		local_max = np.amax(spec)
		if local_max > max_value:
			max_value = local_max

	if not vmin:
		vmin = 0

	for i,spec in enumerate(specs):
		last_plot = axis[i].imshow(spec, vmin = vmin, vmax = max_value, interpolation = None, cmap=plt.cm.magma)
		axis[i].set_aspect(aspect)
		axis[i].set_xlabel("Hop (t)")
		axis[i].invert_yaxis()

	fig.set_size_inches(size[0],size[1])
	fig.colorbar(last_plot, ax=axis, pad=0.07, orientation="horizontal", label="Amplitude (log10)")

	labels = [str(int(item * tick_scale)) for item in axis[-1].get_yticks()]
	axis[0].set_yticklabels(labels)
	axis[0].set_ylabel("Hz")

	return fig

def plot_training_sbs(spec1, spec2, *args, **kwargs):
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	ax1.imshow(spec1[:,4:21].T, *args, **kwargs)
	ax2.imshow(spec2.T, *args, **kwargs)
	ax1.set_aspect(40)
	ax2.set_aspect(40)
	fig.set_size_inches(17,8)
	return fig


def plot_spectogram(spec, aspect, tick_scale=1, *args, **kwargs):
	fig, ax = plt.subplots(1, 1)

	last_plot = ax.imshow(spec.T, interpolation=None,
	                      cmap=plt.cm.magma, *args, **kwargs)
	ax.set_aspect(aspect)
	ax.set_ylabel("hop (t)")

	fig.colorbar(last_plot, ax=ax, pad=0.02)

	labels = [str(int(item * tick_scale)) for item in ax.get_xticks()]
	ax.set_xticklabels(labels)
	ax.set_xlabel("Hz")

	fig.set_size_inches(16,3)

	return fig
