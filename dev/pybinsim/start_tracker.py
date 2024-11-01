import argparse
import sys
import time
# import msvcrt
import freefield
import slab
from pythonosc import udp_client

sources = slab.HRTF.kemar().sources.vertical_polar

def start_tracker():
    # Default values
    oscIdentifier = '/pyBinSim'
    ip = '127.0.0.1'
    port = 10000
    comPort = 'COM4'    # please choose the correct COM-Port
    baudRate = 57600
    print(['ComPort :', comPort])
    print(['Baudrate: ', baudRate])
    print(['IP: ', ip])
    print(['Using Port ', port])
    # Create OSC client
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
    args = parser.parse_args()
    client = udp_client.SimpleUDPClient(args.ip, args.port)


    # nSources = 1
    # minAngleDifference = 5
    # filterSet = 0

    # Internal settings:
    # positionVectorSubSampled = range(0, 360, minAngleDifference)



    # try:
    #     spark9dof = Spark9dof(com_port=comPort, baudrate=baudRate)
    # except RuntimeError as e:
    #     print(e)
    #     sys.exit(-1)

    for i in range(len(sources)):


        # # define current angles as "zero position"	when spacebar is hit
        # if msvcrt.kbhit():
        #     char = msvcrt.getch()
        #
        #     if ord(char) == 32:
        #         rollOffset = roll
        #         pitchOffset = pitch
        #         yawOffset = yaw

            # # Key '1' actives 1st filter set
            # if ord(char) == 49:
            #     filterSet = 0
            # # Key '2' actives 2nd filter set
            # if ord(char) == 50:
            #     filterSet = 1


            # print(filterSet)
        #
        # rpy = spark9dof.get_sensor_data()
        #
        # if rpy:
        #     roll, pitch, yaw = rpy
        #     yaw += 180
        # else:
        #     roll, pitch, yaw = 0, 0, 0



        # build OSC Message and send it
        # for n in range(0, nSources):
        #     # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
        #     yaw = min(positionVectorSubSampled, key=lambda x: abs(x - yaw))
        binSimParameters = [0, i, 0, 0, 0, 0, 0]
        # print(['Source ', n, ' Yaw: ', round(yaw), '  Filterset: ', filterSet])
        client.send_message(oscIdentifier, binSimParameters)
        print(f'sending parameters: az: {sources[i, 0]} ele: {sources[i, 1]}')

        time.sleep(0.3)


if __name__ == "__main__":
    start_tracker()
