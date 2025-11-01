from pathlib import Path
import re
import numpy
from scipy.io import wavfile
import slab

def build_ir_dict(dirs, conflict="skip", up_correction=6.25, fmt="%.1f", sort_result=True):
    """
    Build a single IR dict from one or multiple folders of WAVs.

    Filenames expected:
        pf_(center|up)_{speaker}_az{az}_el{el}.wav[.wav]
    Example:
        pf_center_19_az0_el-25.wav.wav
        pf_up_20_az0_el-37.wav.wav   -> elevation += up_correction

    Parameters
    ----------
    dirs : str | Path | list[str | Path]
        One folder or a list of folders to scan (recursively).
    conflict : {'skip','overwrite','error'}, default 'skip'
        What to do if a generated key already exists.
    up_correction : float, default 6.25
        Degrees to add to elevation for files with 'pf_up_'.
    fmt : str, default '%.1f'
        Formatting for azimuth/elevation in keys.
    sort_result : bool, default True
        If True, sort by (elevation, azimuth) before returning.

    Returns
    -------
    dict[str, slab.Filter]
        Keys:  "{speaker}_{az_fmt}_{el_fmt}"
        Values: slab.Filter(data=<n_taps x n_ch>, samplerate=fs, fir='IR')
    """
    if not isinstance(dirs, (list, tuple)):
        dirs = [dirs]
    dirs = [Path(d) for d in dirs]

    pat = re.compile(
        r"pf_(?P<pos>center|up)_(?P<spk>\d+)_az(?P<az>-?\d+(?:\.\d+)?)_el(?P<el>-?\d+(?:\.\d+)?)\.wav(?:\.wav)?$"
    )

    ir_dict = {}
    seen_paths = set()

    for base_dir in dirs:
        for p in base_dir.rglob("*.wav"):
            if p in seen_paths:
                continue
            seen_paths.add(p)

            m = pat.search(p.name)
            if not m:
                continue

            pos = m.group("pos")            # 'center' or 'up'
            spk = int(m.group("spk"))
            az  = float(m.group("az"))
            el  = float(m.group("el"))
            if pos == "up":
                el += up_correction

            fs, data = wavfile.read(p)

            # Normalize dtype safely
            if numpy.issubdtype(data.dtype, numpy.integer):
                scale = numpy.iinfo(data.dtype).max
                data = data.astype(numpy.float32) / max(scale, 1)
            elif numpy.issubdtype(data.dtype, numpy.floating):
                data = data.astype(numpy.float32)
            else:
                raise TypeError(f"Unsupported WAV dtype {data.dtype} in {p}")

            if data.ndim == 1:
                data = data[:, None]  # ensure (n_taps, n_ch)

            key = f"{spk}_{fmt % az}_{fmt % el}"

            if key in ir_dict:
                if conflict == "skip":
                    continue
                elif conflict == "error":
                    raise ValueError(f"Duplicate key '{key}' from {p}")
                # 'overwrite' → fall through

            ir_dict[key] = slab.Filter(data=data, samplerate=fs, fir="IR")

    if sort_result:
        ir_dict = dict(sorted(
            ir_dict.items(),
            key=lambda kv: (float(kv[0].rsplit("_", 2)[-1]),  # elevation
                            float(kv[0].rsplit("_", 2)[-2]))  # azimuth
        ))
    return ir_dict

base1 = "/Users/paulfriedrich/projects/hrtf_relearning/data/hrtf/rec/pf_test_1/central cone"
base2 = "/Users/paulfriedrich/projects/hrtf_relearning/data/hrtf/rec/pf_test_1/up"

ir_dict = build_ir_dict([base1, base2], conflict="overwrite")  # or 'skip' / 'error'
print(len(ir_dict), "IRs loaded")