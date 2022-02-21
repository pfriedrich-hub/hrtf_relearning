import sofar as sf

# print list of sofa 'spatially oriented format for acoustics' conventions
sf.list_conventions()

# create sofar.sofa object
sofa = sf.Sofa("SimpleFreeFieldHRTF")

# list all attributes inside a SOFA object
sofa.info("all")

sofa.info("Data_Real")

sofa.list_dimensions
# M denotes the number of source positions
# R is the number of ears
# N gives the lengths of the HRTFs in samples
# C is always three, because coordinates are either given by azimuth, elevation and radius in degree
#

sofa.get_dimension("N")
sofa.get_dimension('M')

sofa.Data_Real  # prints [0, 0]
sofa.Data_Real = [1, 1]
sofa.SourcePosition = [90, 0, 1.5]


sofa.Data_Real = [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]
sofa.Data_Imag = [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]
sofa.SourcePosition = [[90, 0, 1.5], [-90, 0, 1.5]]

sofa.info('SourcePosition')
sofa.verify()
sofa.list_dimensions
sofa.get_dimension("N")
