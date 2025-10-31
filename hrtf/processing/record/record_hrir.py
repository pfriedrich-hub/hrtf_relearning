from hrtf.processing.record.in_ear_rec import *
from hrtf.processing.record.rec2ir import *
from hrtf.processing.make.add_interaural import *
import pyfar
# global parameters
fs = 48828  # 97656, 195312.5
slab.set_default_samplerate(fs)
hrtf_dir = Path.cwd() / 'data' / 'hrtf'

id = 'pf'
# reference_file = 'mems_long_full_15_10.pkl'
reference_dir = 'test'

def record_hrir(id, overwrite):
    """
    Record from in-ear microphones and estimate the head related impulse response function.
    Arguments:
        id (int): Subject ID
        overwrite (bool): Whether to overwrite existing recordings or load from existing.
            Automatically create new recordings if folder does not exist.
    """
    subject_dir = hrtf_dir / 'rec' / id
    if not subject_dir.exists() or overwrite:  # record and write
        subject_dir.mkdir(exist_ok=True, parents=True)
        (subject_dir / 'wav').mkdir(exist_ok=True)
        recordings_dict = record_dome(5, 20)  # record
        recordings2wav(recordings_dict, subject_dir / 'wav')  # write
    else:  # load existing recordings
        recordings_dict = wav2recordings(path=subject_dir / 'wav')
    # load reference dict
    reference_dict = wav2recordings(hrtf_dir / 'rec' / 'reference' / reference_dir)
    # compute Impulse Response
    ir_dict = recordings2ir(recordings_dict, reference_dict)
    # add azimuth sources
    ir_dict = add_az_sources(ir_dict, az_range=(-35, 35))
    ir_dict = dict(sorted(ir_dict.items(), key=lambda kv: (float(kv[0].rsplit("_", 2)[-1]),  # sort by elevation
                                                           float(kv[0].rsplit("_", 2)[-2]))))    # and azimuth
    # construct HRIR
    data = numpy.array([filt.data for filt in list(ir_dict.values())])
    hrir = slab.HRTF(data=data, sources=get_sources(ir_dict), samplerate=fs, datatype='FIR')
    # add itd / ild
    hrir = add_itd(hrir)  # todo indiv. head radius?
    hrir = add_ild(hrir)  # todo test db conversion
    slab.HRTF.write_sofa(hrir, subject_dir / f'{id}.sofa')     # write to sofa
    return hrir

def recordings2ir(recordings_dict, reference_dict):
    """
    Build a dictionary of Impulse Response slab.Filters from recordings and reference dictionaries
    Args:
        recordings_dict : Dictionary mapping In-Ear Recordings to loudspeaker indices and sound source locations.
        reference_dict : Dictionary mapping Reference Recordings to loudspeaker indices.
    Returns:
        ir_dict : Dictionary mapping directional impulse responses to sound source locations.
    """
    fs = list(recordings_dict.values())[0].samplerate
    ir_dict = dict()
    for key, recording in recordings_dict.items():
        # get the reference recording for the current speaker idx
        reference = reference_dict[[k for k in reference_dict.keys() if k[:2] == key[:2]][0]]
        # compute ir (see rec2ir.py for details)
        ir_dict[key] = rec2ir(recording, reference)
    return ir_dict

def get_sources(ir_dict, distance=1.2):
    """
    Extract azimuth and elevation (and fixed distance) from dict keys like '19_0.0_40.0'.
    Works only for central speaker array, assuming vertical polar coordinates.
    """
    coords = []
    for key in ir_dict.keys():
        parts = key.split("_")
        az = float(parts[1])
        el = float(parts[2])
        coords.append([az, el, distance])
    return numpy.array(coords).astype('float16')

def add_az_sources(ir_dict, az_range=(-35, 35)):
    """
    Extend an IR dictionary by duplicating each elevation across additional azimuths.
    """
    sources = get_sources(ir_dict)
    vertical_res = (sources[:,1].max() - sources[:,1].min()) / (len(sources) - 1)  # assuming equally spaced sources
    azimuths = numpy.arange(az_range[0], az_range[1]+1, vertical_res)
    new_entries = {}
    for k, filt in ir_dict.items():
        spk_id, _, el_str = k.split("_")  # keep original speaker index and elevation text
        for az in azimuths:
            az_str = f"{float(az):.1f}"  # match your one-decimal formatting
            new_key = f"{spk_id}_{az_str}_{el_str}"
            if new_key not in ir_dict and new_key not in new_entries:
                new_entries[new_key] = filt  # or copy.deepcopy(filt) if you want independent objects
    ir_dict.update(new_entries)
    return ir_dict