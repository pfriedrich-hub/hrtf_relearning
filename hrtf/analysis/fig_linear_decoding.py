# improve noise solution (should be able to draw several points per ele)
# add outer function to get hist of accuracy dependent on TF width

import matplotlib.pyplot as plt
import numpy
import scipy
import sklearn.linear_model
import sklearn.metrics
import slab

def gaussian(x, mu=0, fwhm=2.355):
	'Returns a Gaussian distribution with mean=mu (default=0) and fwhm (default=2.355 [sigma=1])'
	sigma = fwhm / 2.355
	return numpy.exp(-numpy.power(x - mu, 2.) / (2 * numpy.power(sigma, 2.)))

def make_tfs(N_tfs=10, N_ele=100):
	ele = numpy.linspace(-70, 70, N_ele, endpoint=True)
	# make transfer functions (gaussians with random mu and sigma)
	tfs = numpy.zeros((N_ele, N_tfs))
	for i in range(N_tfs):
		fwhm = numpy.random.normal(loc=25, scale=15)
		tf = gaussian(ele, mu=numpy.random.randint(low=-90, high=90), fwhm=fwhm)
		tfs[:,i] = tf
		#tfs += 0.05*numpy.random.randn(N_ele,N_tfs) # add some noise
	return tfs

def train_regression(tfs, test_len=25, test_start=None):
	N_ele = tfs.shape[0]
	ele = numpy.linspace(-70, 70, N_ele, endpoint=True)
	if not test_start:
		test_start = -test_len / 2
	test_start = numpy.argmin(numpy.abs(ele-test_start))
	test_idx = numpy.full(N_ele, False, dtype=bool)
	test_idx[test_start:test_start+test_len] = True
	tfs_train, tfs_test, ele_train, ele_test = tfs[~test_idx,:], tfs[test_idx,:], ele[~test_idx], ele[test_idx]
	regressor = sklearn.linear_model.Lasso()
	regressor.fit(tfs_train, ele_train)
	ele_pred = regressor.predict(tfs_test)
	ele_train_pred = regressor.predict(tfs_train)
	error = numpy.sqrt(sklearn.metrics.mean_squared_error(ele_test, ele_pred))
	print('Root Mean Squared Error:', error)
	return regressor, ele_test, ele_pred, ele_train, ele_train_pred, error

def train_correlation(tfs, test_len=25, test_start=None):
	N_ele = tfs.shape[0]
	ele = numpy.linspace(-70, 70, N_ele, endpoint=True)
	if not test_start:
		test_start = -test_len / 2
	test_start = numpy.argmin(numpy.abs(ele-test_start))
	test_idx = numpy.full(N_ele, False, dtype=bool)
	test_idx[test_start:test_start+test_len] = True
	tfs_train, tfs_test, ele_train, ele_test = tfs[~test_idx,:], tfs[test_idx,:], ele[~test_idx], ele[test_idx]
	ele_pred = numpy.zeros_like(ele_test)
	for idx in range(len(ele_test)):
		pattern = tfs_test[idx]
		corrs = numpy.zeros_like(ele_train)
		for i in range(len(corrs)):
			corrs[i] = scipy.stats.pearsonr(pattern, tfs_train[i,:])[0]
		max_corr_idx = numpy.argmax(corrs)
		ele_pred[idx] = ele_train[max_corr_idx]
	return ele_test, ele_pred

def uncorr_hrtf_filterbank(length=5000, bandwidth=1/12, low_lim=6000, hi_lim=14000, samplerate=32000):
	freq_bins = numpy.fft.rfftfreq(length, d=1/samplerate)
	nfreqs = len(freq_bins)
	center_freqs, bandwidth, erb_spacing = slab.Filter._center_freqs(low_lim=low_lim, hi_lim=hi_lim, bandwidth=bandwidth)
	nfilters = len(center_freqs)
	filts = numpy.ones((nfreqs, nfilters))
	freqs_erb = slab.Filter._freq2erb(freq_bins)
	for i in range(nfilters):
		l = center_freqs[i] - erb_spacing/2
		h = center_freqs[i] + erb_spacing/2
		avg = center_freqs[i]  # center of filter
		rnge = erb_spacing  # width of filter
		filts[(freqs_erb > l) & (freqs_erb < h), i] = 1-numpy.cos((freqs_erb[(freqs_erb > l) & (freqs_erb < h)]- avg) / rnge * numpy.pi)
	return slab.Filter(data=filts, samplerate=samplerate, fir=False)

def synthetic_hrtf():
	fb = uncorr_hrtf_filterbank(length=10000, bandwidth=1/12, low_lim=5000, hi_lim=18000, samplerate=44100)
	ele = numpy.linspace(-65,50,fb.nchannels)
	sources = numpy.zeros((fb.nchannels,2))
	sources[:,1] = ele
	hrtf = slab.HRTF(fb, sources=sources)
	return hrtf

def figure(ax1=None):
	error = 100
	while error > 2:
		tfs = make_tfs(N_tfs=20, N_ele=100)
		regressor, ele_test, ele_pred, ele_train, ele_train_pred, error = train_regression(tfs, test_len=25)
	N_ele, N_tfs = tfs.shape
	ele = numpy.linspace(ele_train.min(), ele_train.max(), N_ele, endpoint=True)
	# plot TFs with color according to weight magnitude
	cmap = plt.get_cmap('coolwarm')
	weights = numpy.interp(regressor.coef_, [numpy.min(regressor.coef_), 0, numpy.max(regressor.coef_)], [0,127,255]).astype(int)

	for i in range(N_tfs):
		ax1.plot(ele, tfs[:,i], linewidth=0.5, color=cmap(weights[i]), alpha=0.7)
	ax1.set_ylim(ymin=0, ymax=1)
	ax1.set_xlabel('Elevation [˚]')
	ax1.set_ylabel('Activation')
	ax1.tick_params('both', length=2, pad=2)
	ax1.axvline(x=ele_test.min(), linewidth=1, alpha=0.7, color='0.0')
	ax1.axvline(x=ele_test.max(), linewidth=1, alpha=0.7, color='0.0')
	ax1.spines['top'].set_visible(False)
	ax1.text(-0.28, 1.15, 'D', transform=ax1.transAxes, fontweight='bold', horizontalalignment='left', verticalalignment='center')

	ax2 = ax1.twinx()
	#ax2.set_ylim(ymin=0, ymax=1)
	ax2.plot(ele_train, ele_train_pred, linewidth=1, color='0.0')
	ax2.plot(ele_test, ele_pred, linewidth=2.5, color='red')
	ax2.set_ylabel('predicted [˚]')
	ax2.tick_params('both', length=2, pad=2)
	ax2.spines['top'].set_visible(False)


	#ele_test, ele_pred = train_correlation(tfs, test_len=25, test_start=None)
	#ax2.plot(ele_test, ele_pred, linewidth=2, alpha=0.6, color='green')

if __name__ == '__main__':
	params = {
		'font.family': 'serif',
		'font.size': 10,
		'axes.labelsize': 9,
		'legend.fontsize': 7,
		'xtick.labelsize': 7,
		'ytick.labelsize': 7,
		}
	plt.rcParams.update(params)
	fig, ax = plt.subplots(nrows=1, ncols=1)
	figure(ax)
	fig.set_size_inches(2.05*0.3937, (2.05*0.3937)) # 8.2cm * conversion_to_inch, hight is width / golden ratio
	#fig.savefig('fig_lindecoding.pdf', format='pdf', bbox_inches='tight')
	fig.show()
