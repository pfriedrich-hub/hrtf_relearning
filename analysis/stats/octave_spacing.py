# --- 5 1/2 non-overlapping octave bands
bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
# == numpy.logspace(1, 2, 5, base=4)

bands = [(3500, 5000), (5000, 7200), (7200, 10400), (10400, 15000)]
# == numpy.logspace(1, numpy.emath.logn(3.5, 15), 5, base=3.5)


# --- 5 overlapping octave bands
bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
# == numpy.logspace(1, 1.5, 5, base=4)
# == numpy.logspace(1, numpy.emath.logn(8, 16), 5, base=8)
# logarithm of 16 to base 8? numpy.emath.logn(8, 16) == 1.3333
# 2**3 = 8; therefore, 3 is the logarithm of 8 to base 2, or 3 = log2 8

bands = [(3500, 7000), (4200, 8500), (5100, 10200), (6200, 12400), (7500, 15000)]
# == numpy.logspace(1, numpy.emath.logn(3.5, 7.5), 5, base=3.5)
# == numpy.logspace(1, numpy.emath.logn(7, 15), 5, base=7)

bands = [(3500, 7000), (4200, 8300), (4900, 9900), (5900, 11800), (7000, 14000)]
# == numpy.logspace(1, numpy.emath.logn(3.5, 7), 5, base=3.5)
# == numpy.logspace(1, numpy.emath.logn(7, 14), 5, base=7)