import freefield
import slab
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

freefield.initialize('dome', default='play_rec')  # initialize setup

speaker_id = 23
[speaker] = freefield.pick_speakers(speaker_id)
signal = slab.Sound.chirp(duration=3.0, level=90)


rec = freefield.play_and_record(speaker, signal, compensate_delay=True,
          compensate_attenuation=False, equalize=False)

rec.level = 30
rec.play()


rec.waveform()
rec.spectrum()

