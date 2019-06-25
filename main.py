from __future__ import print_function
import sys
import os
import soundfile as sf
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import logging as log

from armv7l.openvino.inference_engine import IENetwork, IEPlugin



def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h','--help',action='help', default=SUPPRESS, help='Show this help message and exit')
    args.add_argument('-m','--model',help="Required. Path to .xml file with a trained model", required=True, type = str)
    args.add_argument('-i','--input',help="Required. Path to speech file", required = True, type=str)
    args.add_argument('-l','--cpu_extension',help="Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.",
        type = str, default=None)
    args.add_argument('-pp','--plugin_dir',help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument('-d','--device',help="Optional. Specify the target device to infer on. Options: Intel CPU, GPU, FPGA, HDDL or MYRAID (Nerual Compute Stick) "
        "is acceptable. The program will look for a suitable plugin for device specified. Default Value: CPU", default="CPU",type=str)
    args.add_argument('--labels',help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument('-pt','--prob_threshold',help="Optional. Probability threshold for detections filtering", default = 0.5, type = float)
    return parser

def sound_decode(file):
    data,samplerate = sf.read(file,always_2d=True)
    data = data.sum(axis=1)/2
    #log.info("The sample rate is "+str(samplerate))
    return data


def main():
    log.basicConfig(format=" [ %(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model #File path for the model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    #log.info("Loading the network files model")


    plugin = IEPlugin(device = args.device, plugin_dirs = args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)

    net = IENetwork(model = model_xml, weights = model_bin)
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) !=0:
            log.error("Some layers are not supported by the plugin")
            sys.exit(1)

    #assert len(net.inputs.keys()) == 7904 # 2 inputs for DeepSpeech
    #assert len(net.outputs) == 2048 # [464,2048,2048]



    voice = sound_decode(args.input)




    exec_net = plugin.load(network = net, num_requests = 3)

    #exec_net.requests[0].inputs['data'][:] = voice
    res = exec_net.infer({'input_node': voice})
    #print("haha")



    #TODO implement the output for the inference
    

if __name__ == "__main__":
    sys.exit(main() or 0)
