"""
This is an example how to implement real time processing of the DTLN OpenVINO
model in python.
   

Author: Chungyeh Wang (chungyeh.wang@intel.com)
Version: 06.08.2021

This code is licensed under the terms of the MIT-license.
"""

import soundfile as sf
import numpy as np
import time
import argparse
from openvino.inference_engine import IECore


# arguement parser for running directly from the command line
parser = argparse.ArgumentParser(description='data evaluation')

parser.add_argument('-m1', '--model1', required=True, type=str,
                    help='Required. Path to an model 1 .xml file with a trained model.')
parser.add_argument('-m2', '--model2', required=True, type=str,
                    help='Required. Path to an model 2 .xml file with a trained model.')
parser.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')                    
parser.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an audio wav file.')

args = parser.parse_args()
    
##########################
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
block_len = 512
block_shift = 128

ie = IECore()
    
# load model 1
net1 = ie.read_network(model=args.model1)
input_blob_1_iter = iter(net1.input_info)
input_blob_1 = [next(input_blob_1_iter), next(input_blob_1_iter)]
out_blob_1_iter = iter(net1.outputs)
out_blob_1 = [next(out_blob_1_iter), next(out_blob_1_iter)]
exec_net1 = ie.load_network(network=net1, device_name=args.device)    

# load model 2
net2 = ie.read_network(model=args.model2)
input_blob_2_iter = iter(net2.input_info)
input_blob_2 = [next(input_blob_2_iter), next(input_blob_2_iter)]
out_blob_2_iter = iter(net2.outputs)
out_blob_2 = [next(out_blob_2_iter), next(out_blob_2_iter)]
exec_net2 = ie.load_network(network=net2, device_name=args.device)

states_1 = np.zeros((1,2,128,2)).astype('float32')
states_2 = np.zeros((1,2,128,2)).astype('float32')
    
# load audio file
audio,fs = sf.read(args.input)
# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio
out_file = np.zeros((len(audio)))
# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
# iterate over the number of blcoks  
time_array = []      
for idx in range(num_blocks):
    start_time = time.time()
    # shift values and write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
    # calculate fft of input block
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    # reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
    
    res1 = exec_net1.infer(inputs={input_blob_1[0]:in_mag, input_blob_1[1]:states_1})
    out_mask = res1[out_blob_1[0]]
    states_out_1 = res1[out_blob_1[1]]
    states_1 = states_out_1
 
    # calculate the ifft
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
    estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
    
    res2 = exec_net2.infer(inputs={input_blob_2[0]:estimated_block, input_blob_2[1]:states_2})
    out_block = res2[out_blob_2[0]]
    states_out_2 = res2[out_blob_2[1]]
    states_2 = states_out_2

    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # write block to output file
    out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    time_array.append(time.time()-start_time)
    
# write to .wav file 
sf.write('out.wav', out_file, fs) 
print('Processing Time [ms]:')
print(np.mean(np.stack(time_array))*1000)
print('Processing finished.')