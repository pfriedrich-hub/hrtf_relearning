from mbientlab.metawear import MetaWear
from mbientlab.metawear.cbindings import *
from mbientlab.warble import *

device = MetaWear(address)
device.connect()