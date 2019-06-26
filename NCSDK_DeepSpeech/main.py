from __future__ import print_function
import sys
import os
import soundfile as sf
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import logging as log
import ntpath

import mvnc.mvncapi as mvnc


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h','--help',action='help', default=SUPPRESS, help='Show this help message and exit')
    args.add_argument('-m','--model',help="Required. Path to graph file with a trained model", required=True, type = str)
    args.add_argument('-i','--input',help="Required. Path to speech file", required = True, type=str)
    return parser

def sound_decode(file):
    data,samplerate = sf.read(file,always_2d=True)
    data = data.sum(axis=1)/2
    #log.info("The sample rate is "+str(samplerate))
    return data

	

def open_ncs_device():
   # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:

        print( "No devices found" )

        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()
    return device
	
	


	
def load_graph(device):
    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()
    # Load the graph buffer into the NCS
    graph = device.AllocateGraph( blob )
    return graph

	
def infer_sound(graph,audio):
	graph.LoadTensor(img,'user object')
	
	output, userobj = graph.GetResult()
	
	print(order)

def close_ncs_device(device,graph):
	graph.DeallocateGraph()
	device.CloseDevice()

def main():
    log.basicConfig(format=" [ %(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    #log.info("Loading the network files model")
	device = open_ncs_device()
	graph = load_graph(device)
    voice = sound_decode(args.input)
	infer_sound(voice)
	close_ncs_device(device,graph)
if __name__ == "__main__":
    sys.exit(main() or 0)