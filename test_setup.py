import freefield
import slab

freefield.initialize('dome', default='play_birec')  # initialize setup

speaker_id = 23
[speaker] = freefield.pick_speakers(speaker_id)
signal = slab.Binaural.chirp(duration=1, level=70)
rec = freefield.play_and_record(speaker, signal, compensate_delay=True,
          compensate_attenuation=False, equalize=False)

rec.play()
rec.waveform()
rec.spectrum()

