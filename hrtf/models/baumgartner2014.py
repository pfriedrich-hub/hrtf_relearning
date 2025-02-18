from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import numpy as numpy
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline # this is for the spline interpolation of step 7
from scipy.special import i0 # This is the von Mises PDF also for step 7

sofa_path_target = Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'hrtf_double_notch.sofa'
sofa_path_template = sofa_path_target
# sofa_path_template = Path.cwd() / 'data' / 'hrtf' / 'sofa' / 'kemar.sofa'

sig_path = Path.cwd() / 'data' / 'sounds' / '1s_pinknoise_44100.wav'
fig_savepath = Path.cwd() / 'data' / 'plots'

shutup = False
do_dtf = False
is_save = True

def baumgartner2014(sofa_path_target, sofa_path_template, sig_path, shutup, do_dtf, fig_savepath, is_save=False):

    # load sofa file
    try:
        target = pf.io.read_sofa(sofa_path_target)[0]
        coardinate = pf.io.read_sofa(sofa_path_target)[1]
        template = pf.io.read_sofa(sofa_path_template)[0]
    except ValueError:
        sofa = sf.read_sofa(sofa_path_target, verify=False)
        target = pf.io.convert_sofa(sf.read_sofa_as_netcdf(sofa_path_target))[0]
        coardinate = pf.io.convert_sofa(sf.read_sofa_as_netcdf(sofa_path_target))[1]
        template = pf.io.convert_sofa(sf.read_sofa_as_netcdf(sofa_path_target))[0]

    if sig_path:
        stim = pf.io.read_audio(sig_path)  # make sure to match the fs of the DTF and stimulus
    else:
        stim = []

    # zeropad inputs
    nfft_new = 2 ** 9
    if target.n_samples < nfft_new:
        target = zero_pad_in(target, nfft_new)
        template = zero_pad_in(template, nfft_new)

    # HRTF -> DTF
    if do_dtf:  # (diffuse field equalization)
        freq_range = [700, 18e3]
        target = hrtf2dtf(target, coardinate, freq_range, shutup)
        template = hrtf2dtf(template, coardinate, freq_range, shutup)

    # Get the DTF of the median plane (azimuth == 0 or 180)
    if 0 in coardinate.azimuth:
        az_mask = numpy.squeeze(numpy.logical_or(coardinate.azimuth == 0, coardinate.azimuth == numpy.pi))
    else:
        az_mask = numpy.squeeze(numpy.logical_or(coardinate.azimuth == numpy.pi*2, coardinate.azimuth == numpy.pi))

    # remove duplicate sources
    _, count = numpy.unique(coardinate[az_mask].polar, return_counts=True)
    az_mask[numpy.where(count>1)[0]] = False

    coardinate_median_plane = coardinate[az_mask]
    sort = numpy.argsort(coardinate_median_plane.polar)
    coardinate_median_plane = coardinate_median_plane[sort]
    tang = coardinate_median_plane
    target = target[az_mask, :]
    # target = target[sort]
    target = target[sort, :]
    template = template[az_mask, :]
    # template = template[sort]
    template = template[sort, :]
    # get some parameters from the input data
    m_sp_directions = target.cshape[0]
    ch = target.cshape[1]
    t_sample = target.n_samples
    fs = int(target.sampling_rate)
    """
    if not(shutup):
        #coardinate_median_plane.show()
        print("number of samples on the saggital (median) plane: ",m_sp_directions)
        print("time samples of HRIR: ",t_sample)
        print("fs = ",fs," [Hz]")

        print("\n\n---------------------")
        print("selected angels from the measured HRTF")
    """
    # step 1  - convolve signal with target DTF(s?)
    target = eq_1(target, stim, shutup )

    # step 2  - spectral analysis: rms of filters in a gamma tone filter bank convolved with target / templates
    target_filt_rmsdB = eq_2(target, fs, shutup)
    template_filt_rmsdB = eq_2(template, fs, shutup)

    # step 3 - gradient extraction across rms values
    target_grad = eq_3(target_filt_rmsdB, shutup)
    template_grad = eq_3(template_filt_rmsdB, shutup)

    # step 4 - distances between target and templates
    sigma = eq_4(template_grad, target_grad, shutup)

    # step 5 - response probabilities (gaussian)
    gamma = 6  # Example value for gamma
    S = 0.7  # Example value for S
    si = eq_5(sigma, S, gamma, shutup)

    # step 6 - binaural weighting (exponential)
    lat = 0
    bwcoef = 13
    sibin = eq_6(si, bwcoef, lat, shutup)

    # step 7 - sensorimotor mapping (smearing - gauss scatter)
    mrsmsp = 17  # This is the PDF parameter it controls the "smearing" effect
    kv_rangsamp = 5  # Example rangsamp value (distance of interpolated angles)
    flags_do_regular = True  # Example do_regular value
    ri, rang = eq_7(sibin, coardinate_median_plane, kv_rangsamp, lat, mrsmsp, flags_do_regular, shutup)

    # step 8 - normalization to probabilities
    pdf = eq_8(ri)
    if is_save:
        plot_pdf_matrix(pdf, rang, coardinate_median_plane, sofa_path_target.stem, fig_savepath, shutup)
    # Create a dictionary with the variables
    output_dict = {
        "pdf": pdf,
        "rang": rang,
        "tang": tang,
        "coardinate_median_plane": coardinate_median_plane
    }
    return output_dict

def eq_1(target,stim,shutup):
# Step 1: DTF filtering Eq.(1)
# The target is being time convolve with the stimuli, creating the directional target signal
# ----
    if stim:
        target = pf.dsp.convolve(target,stim)
    else:
        target = target

    if not(shutup):
        print("Step 1: DTF filtering Eq.(1)\n")
        print(target)
    return (target)

def eq_2(x,fs,shutup):
    # Step 2: Spectral Analysis Eq.(2)
    # in this step, both target and template are filtered using gammatone filterbanks
    # ----
    Gammatones     = pf.dsp.filter.GammatoneBands(freq_range=[700, 20e3],sampling_rate=fs) # Create the filter bank
    x_filt         = Gammatones.process(x)[0] # Apply the filter bank (T)
    x_filt_rms_dB  = do_rms(x_filt)

    if not(shutup):
        # Display using imshow
        print("\nStep 2: Spectral Analysis Eq.(2)")
        plt.figure()
        plt.imshow(x_filt_rms_dB[:,:,0], cmap='bone', origin='lower')
        plt.colorbar()  # Add colorbar
        plt.title("Gammatone filtering and time RMS")
        plt.show()
    return x_filt_rms_dB

def eq_3(x,shutup):
    # Step 3: Positive spectral gradient extraction Eq.(3)
    # in this step, both traget and template max diffrence over frequncy is calculated
    # ----
    x_pos_grad = numpy.zeros([x.shape[0]-1,x.shape[1],x.shape[2]])
    # Calculate the gradient
    for f_idx in range(x.shape[0]-1):
        x_pos_grad[f_idx,:,:] = x[f_idx+1,:,:] - x[f_idx,:,:]
    # Take only the positive values
    x_pos_grad[x_pos_grad < 0] = 0

    if not(shutup):
        print("\nStep 3: Positive spectral gradient extraction Eq.(3)")
        # Display using imshow
        plt.figure()
        plt.imshow(x_pos_grad[:,:,0], cmap='bone', origin='lower')
        plt.colorbar()  # Add colorbar
        plt.title("target_pos_grad")
        plt.show()
    return x_pos_grad

def eq_4(tem,tar,shutup):
    # Step 4: Comparison process Eq.(4)
    # in this step, calculate the L1 norm between the target and the template avarage over frequncy and for each direction
    # ----
    # Initialize sigma array
    sigma = numpy.zeros((tem.shape[1], tar.shape[1], tem.shape[2]))
    # Loop over time and angle indices
    for itang in range(tar.shape[1]):
        # replecate the target for each angle
        tmp = tar[:, itang, :]
        tmp = tmp[:, numpy.newaxis,:]
        tmp = numpy.tile(tmp,(1,tar.shape[1],1))
        isd = tmp - tem
        sigma[:, itang, :,] = numpy.mean(numpy.abs(isd), axis=0)
    
    if not(shutup):
        print("\nStep 4: Comparison process Eq.(4) - L1 error matrix")
        # Display using imshow
        plt.figure()
        plt.imshow(sigma[:,:,0], cmap='bone', origin='lower')
        plt.colorbar()  # Add colorbar
        plt.title("sigma (Left ear channel)")
        plt.show()
        
    return sigma


def eq_5(sigma,S,gamma,shutup):
     # Step 5: Similarity estimation Eq.(5)
    # in this step, The L1 comparison (distance metric) is translated into the response probability (si)
    # ----
    # Constants
    eps = numpy.finfo(float).eps  # Machine epsilon
    si = numpy.zeros(sigma.shape)
    # Loop over channels and iterations
    for ch in range(sigma.shape[2]):
        for it in range(sigma.shape[1]):
            si[:, it, ch] = 1 + eps - (1 + numpy.exp(-gamma * (sigma[:, it, ch] - S)))**-1
    if not(shutup):
        print("\nStep 5: Similarity estimation Eq.(5)")
        # Display using imshow
        plt.figure()
        plt.imshow(si[:,:,0], cmap='bone', origin='lower')
        plt.colorbar()  # Add colorbar
        plt.title("si")
        plt.show()
    return si
    

def eq_6(si,bwcoef,lat,shutup):
    # Step 6: Binaural weighting Eq.(6)
    # in this step, The left and right ear si matrices are weighted with one another
    # here we use the lat parameter to give "off median" results
    # ----
    # Compute the weight of left ear signal
    binw = 1 / (1 + numpy.exp(-lat / bwcoef))
    # Compute sibin
    sibin = binw * si[:, :, 0] + (1 - binw) * si[:, :, 1]
    if not(shutup):
        print("\nStep 6: Binaural weighting Eq.(6)")
        # Display using imshow
        plt.figure()
        plt.imshow(sibin, cmap='bone', origin='lower')
        plt.colorbar()  # Add colorbar
        plt.title("sibin")
        plt.show()

    return sibin

def eq_7(sibin,coardinate_median_plane,kv_rangsamp,lat,mrsmsp,flags_do_regular,shutup):
    # Step 7: Sensorimotor mapping Eq.(7)
    # In this step, the listener's auditory perception (captured by si) is translated into a motor response. 
    # This translation is modeled using circular convolution between the auditory perception and a
    # von Mises Probability Density Function (PDF).
    # Additionally, the si response is interpolated beforehand to more angles (with equal distances) using spline interpolation.
    # ----

    kv_polsamp = numpy.rad2deg(coardinate_median_plane.polar) # Polar angular sampling from the .sofa file
    rang, si_interpolated = interpolate_si(sibin, kv_polsamp, kv_rangsamp, flags_do_regular)

    mrs = mrsmsp/numpy.cos(numpy.deg2rad(lat)) # direction dependent scatter (derivation: const. length rel. to the circumferences of circles considered as cross sections of a unit sphere)
    x = numpy.arange(0, 2*numpy.pi, 2*numpy.pi/len(rang))
    kappa = 1/numpy.deg2rad(mrs)**2 # concentration parameter (~1/sigma^2 of normpdf)
    mrspdf = numpy.exp(kappa * numpy.cos(x)) / (2 * numpy.pi * i0(kappa)) # von Mises PDF
    
    # Perform the periodic convolution between the interpolated si and the PDF
    ri = si_interpolated.copy()
    for tt in range(si_interpolated.shape[1]):
        ri[:, tt] = pconv(si_interpolated[:, tt], mrspdf)
        
    if not(shutup):
        print("\nStep 7: Sensorimotor mapping Eq.(7)")
        # Display using imshow
        plt.figure()
        plt.imshow(si_interpolated, cmap='bone', origin='lower')
        plt.colorbar()  # Add colorbar
        plt.title("si_interpolated")
        plt.show()

        plt.figure()
        plt.imshow(ri, cmap='bone', origin='lower')
        plt.colorbar()  # Add colorbar
        plt.title("ri")
        plt.show()

        plt.figure()
        plt.plot(kv_polsamp, label="Measured polar angles")
        plt.plot(rang, label="Interpolated polar angles")
        plt.ylabel('Angle values (deg)')
        plt.title("Median plane angles from sofa file and the angle interpulation")
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.minorticks_on()
        plt.legend(loc="upper left")
        plt.show()

    return ri, rang

def eq_8(ri):
    # Step 8: Normalization to probabilities Eq.(8)
    # In this step, generate the probability mass vector (PMV) by normalizing the ri results so that each column is equal to one.
    # ----
    eps = numpy.finfo(float).eps  # Machine epsilon
    p = ri / (numpy.sum(ri, axis=0) + eps)
    p = numpy.expand_dims(p, axis=2)
    return p

def zero_pad_in(x,nfft):
    n_zeros = nfft - x.n_samples
    new_shape = (x.cshape[0],x.cshape[1],n_zeros)
    sig_zero = numpy.zeros(new_shape)
    x_data = x.time
    tmp = numpy.concatenate((x_data, sig_zero), axis=2)
    y = pf.Signal(tmp, x.sampling_rate)
    return y

def do_rms(sig):  # eq 2
    # Averaging over time (RMS)
    # Calculate root mean square (rms) along the last axis
    rms_values = numpy.sqrt(numpy.mean(numpy.square(sig.time), axis=-1))
    # Squeeze the array to remove axes with length 1
    rms_values_squeezed = numpy.squeeze(rms_values)
    # Convert to dB scale
    sig_rms_dB = 20 * numpy.log10(rms_values_squeezed)
    return sig_rms_dB

def pconv(x, h):  # eq 7
    # Perform periodic convolution of 1D arrays x and h.
    N = len(x)
    M = len(h)
    y = numpy.zeros(N)
    for n in range(N):
        for m in range(M):
            y[n] += h[m] * x[(n - m) % N]
    return y

def interpolate_si(si, polsamp, rangsamp, do_regular):  # eq 7
    if do_regular:
        rang0 = numpy.ceil(numpy.min(polsamp) * (1 / rangsamp)) * rangsamp
        rang = numpy.arange(rang0, numpy.max(polsamp) + rangsamp, rangsamp)
        siint = numpy.zeros((len(rang), si.shape[1]))
        for tt in range(si.shape[1]):
            interp_func = make_interp_spline(polsamp, si[:, tt], k=3)
            siint[:, tt] = interp_func(rang)
        si = siint
        si[si < 0] = 0  # Ensure SIs are positive (necessary due to spline interpolation)
    else:
        rang = polsamp
    return rang, si

def hrtf2dtf(hrtf, sampling, freq_range, shutup):
    weights = pf.samplings.calculate_sph_voronoi_weights(sampling, normalize=True, center=[0, 0, 0], round_decimals=12)
    weights = weights[..., None]
    ctf = pf.dsp.average(hrtf, caxis=0, mode='log_magnitude_zerophase', weights=weights, nan_policy='raise')
    # ctf = pf.dsp.average(hrtf, caxis=0,mode='log_magnitude_zerophase')
    # ctf_inverse = pf.dsp.regularized_spectrum_inversion(ctf, freq_range)
    ctf_inverse = 1 / ctf
    dtf = hrtf * ctf_inverse
    """
    if not(shutup):
        print("------------CTF--------")
        print(ctf)
        ax = pf.plots.time_freq(ctf, label="ctf")
        ax[0].set_title("Time domain")
        ax[1].set_title("Frequency domain")
        ax[1].legend(loc="lower center")
        plt.show()
        print("-----------------------")
    """
    return dtf

def plot_pdf_matrix(pdf, rang, coardinate_median_plane, title_name, fig_savepath, shutup):
    kv_polsamp = numpy.rad2deg(coardinate_median_plane.polar) # Polar angular sampling from the .sofa file

    target_angle = numpy.atleast_2d(kv_polsamp)
    response_angle = numpy.atleast_2d(rang)

    x_labels = numpy.arange(-60, target_angle[0][-1], 30).astype(int)
    y_labels = numpy.arange(-60, response_angle[0][-1], 30).astype(int)

    # Calculate absolute differences between each angle in a and a_label
    # and find the indices of the 10 closest angles in a to each angle in a_label
    absolute_diff = numpy.abs(target_angle[0] - x_labels.reshape(-1, 1))
    closest_indices_x = numpy.argmin(absolute_diff, axis=1)
    absolute_diff = numpy.abs(response_angle[0] - y_labels.reshape(-1, 1))
    closest_indices_y = numpy.argmin(absolute_diff, axis=1)

    plt.figure(figsize=(6, 6))  # Specify the size directly when creating the figure
    plt.imshow(pdf[:, :], cmap='bone', origin='lower', vmax=0.1)
    cbar = plt.colorbar()  # Add colorbar
    # Set x and y axis ticks
    plt.xticks(closest_indices_x, x_labels)
    plt.yticks(closest_indices_y, y_labels)
    # axis lables
    plt.xlabel('Target Angle (deg)')
    plt.ylabel('Response Angle (deg)')
    cbar.set_label(r'$Probability \,\, / \, 5\circ$')
    plt.title("PDF matrix: " + title_name)
    plt.savefig(fig_savepath / str("median_localization_" + title_name + ".png"))
    # if not (shutup):
    plt.show()
    # else:
    #     plt.close()

    return target_angle, response_angle


def circular_stats_from_pdf(pdf, rang, tang, shutup):
    """
    Compute descriptive statistics for modeled median plane localization.

    Parameters
    ----------
    pdf : numpy array
        Probability density function of the modeled localization performance of
        shape R x T x N, with R being the number of response angles, T the
        number of target angles and N the number of subjects.
    rang : numpy array
        Response polar angle in degree of shape R x 1. The polar angle is the
        angle along the median sagittal plane, with -90 being below, 0 to the
        front, 90 above, and 180 behind the subject.
    tang : numpy array
        Response target angle in degree of shape T x 1.

    Returns
    -------
    local_pe : numpy array
        RMS polar error in degree according to [1]_ Eq. (10) at each target
        angle. Array of shape T X S.
    local_qe : numpy array
        Quadrant error in percent according to [1]_ Eq. (9) at each target
        angle. Array of shape T X S.
    circ_mean : numpy array
        Circular mean in degree according to [2]_. Array of shape T x S.
    circ_var : numpy array
        Circular variance between 0 and 1 according to [2]_. Array of
        shape T x S.
    circ_std : numpy array
        Circular standard deviation between 0 and infinity according to [2]_.
        Array of shape T x S.

    References
    ----------
    .. [1] Baumgartner, R., Majdak, P., & Laback, B. (2014). Modeling sound-
           source localization in sagittal planes for human listeners. Journal
           of the Acoustical Society of America, 136(2), 791–802.
           https://doi.org/10.1121/1.4887447
    .. [2] Fisher, N. I. (1993). Statistical Analysis of Circular Data.
           Cambridge University Press.
    """

    kv_polsamp = numpy.rad2deg(tang.polar)  # Polar angular sampling from the .sofa file
    response = numpy.atleast_2d(rang).T
    target = numpy.atleast_2d(kv_polsamp)

    # append dimension for correct broadcasting
    if pdf.ndim == 3:
        response = response[..., None]
        target = target[..., None]

    # mean resultant vector R using complex numbers
    # 0 <= abs(R) <= 1
    mean_resultant_complex = numpy.exp(1j * numpy.radians(response))
    mean_resultant_complex = numpy.sum(pdf * mean_resultant_complex, axis=0)

    # mean angle in degree
    circ_mean = numpy.angle(mean_resultant_complex, deg=True) % 360

    # circular standard deviation, 0 <= var <= 1
    circ_var = 1 - numpy.abs(mean_resultant_complex)

    # circular standard deviation, 0 <= std <= inf
    circ_std = numpy.sqrt(-2 * numpy.log(numpy.abs(mean_resultant_complex)))

    # distance between two angles
    delta_angle = numpy.abs(response - target)
    delta_angle = numpy.minimum(delta_angle, 360 - delta_angle)

    # mask local response angles
    local = delta_angle < 90

    # allocate error measures
    local_pe = numpy.zeros((pdf.shape[1], pdf.shape[2]))
    local_qe = numpy.zeros((pdf.shape[1], pdf.shape[2]))

    # loop target angles
    for tt in range(pdf.shape[1]):
        # current local response angles
        rr = local[:, tt].flatten()

        # polar error at target angle
        local_pe[tt] = numpy.sum(delta_angle[rr, tt] ** 2 * pdf[rr, tt], axis=0)
        local_pe[tt] = numpy.sqrt(local_pe[tt] / numpy.sum(pdf[rr, tt], axis=0))
        # quadrant error at target angle
        local_qe[tt] = numpy.sum(pdf[~rr, tt], axis=0) * 100

    if not (shutup):
        print("Performance Predictions:")
        print("------------------------")
        print("Quadrant errors (%) ", f"{numpy.mean(local_qe):.1f}")
        print("Local polar RMS error (deg) ", f"{numpy.mean(local_pe):.1f}")

        print("circ_mean ", f"{numpy.mean(circ_mean):.1f}")
        print("circ_var ", f"{numpy.mean(circ_var):.1f}")
        print("circ_std ", f"{numpy.mean(circ_std):.1f}")

    return local_pe, local_qe, circ_mean, circ_var, circ_std

def calc_color_err(sofa_path_target, sofa_path_template, shutup):
    # load sofa file
    target = pf.io.read_sofa(sofa_path_target)[0]
    template = pf.io.read_sofa(sofa_path_template)[0]

    m_sp_directions = target.cshape[0]
    ch = target.cshape[1]
    t_sample = target.n_samples
    fs = int(target.sampling_rate)

    f_range_vec = [1.5e3, 20e3]

    # step 1 (filt with gammatone filter bank)
    Gammatones = pf.dsp.filter.GammatoneBands(freq_range=f_range_vec, sampling_rate=fs)  # Create the filter bank
    p_ref_fc = Gammatones.process(template)[0]  # Apply the filter bank (T)
    p_hat_fc = Gammatones.process(target)[0]  # Apply the filter bank (T)

    index = p_ref_fc.find_nearest_frequency(f_range_vec)

    p_ref_fc = p_ref_fc.freq[:, :, :, index[0]:index[1]]
    p_ref_fc = numpy.sum(p_ref_fc, axis=3)

    p_hat_fc = p_hat_fc.freq[:, :, :, index[0]:index[1]]
    p_hat_fc = numpy.sum(p_hat_fc, axis=3)

    color_err = numpy.abs(p_hat_fc) / numpy.abs(p_ref_fc)
    color_err = numpy.abs(10 * numpy.log10(color_err))

    color_err = numpy.mean(color_err[:, :, 0], axis=1)

    plt.figure()
    plt.plot(color_err, '--o', label='left ear')
    plt.grid()
    plt.xlabel("f index")
    plt.ylabel("Coloration error [dB]")
    plt.legend()
    plt.title("Coloration Error average over Space");
    plt.show()

if __name__ == "__main__":
    out = baumgartner2014(sofa_path_target,sofa_path_template,sig_path,shutup,do_dtf,fig_savepath,is_save)
    local_pe, local_qe, circ_mean, circ_var, circ_std = circular_stats_from_pdf(out['pdf'], out['rang'], out['tang'],shutup)
    local_qe = numpy.mean(local_qe)
    local_pe = numpy.mean(local_pe)
