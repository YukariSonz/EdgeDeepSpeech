import sys
import os
import soundfile as sf
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import logging as log
import ntpath
from edgetpu.basic.basic_engine import BasicEngine

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




def main():

    log.basicConfig(format=" [ %(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)

    args = build_argparser().parse_args()

    #log.info("Loading the network files model")

    voice = sound_decode(args.input)

    infer_engine = BasicEngine(args.model)
    latency, results = infer_engine.RunInference(voice)
    print(latency)
    print(infer_engine.get_inference_time())
    print(results)

if __name__ == "__main__":

    sys.exit(main() or 0)