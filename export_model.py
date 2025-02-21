import torch
from typing import Optional
from utils import *
import torch
from redefined_modules.modeling_clip import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from sd_portrait_ownpip import AutoencoderKLDecoder
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='描述程序的功能和用途')
    parser.add_argument('--export_quant_model', type=bool, default=False, help='where export the quant model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    batch_size = 1
    text_length = 77
    latend_h = 64
    unetc= 4
    text_dim = 768
    model_id = '/data1/chenjun/huf/portrait-finetuned'
    
    ## 加载文本编码器（text_encoder）
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder.config.return_dict = False
    onnx_file_path = "onnx/text_encoder.onnx"
    torch_inputs = torch.randint(100, (1, text_length), dtype=torch.int32)
    if not os.path.exists(onnx_file_path):
        torch.onnx.export(
                text_encoder,
                torch_inputs,
                onnx_file_path,
                input_names=['text'],
                output_names=["text_features"],
                opset_version=17,
            )
    if not args.export_quant_model:
        qnn_onnx_converter(onnx_file_path, folder='text_encoder_float', float_bit=32)
    else:
        qnn_onnx_converter(onnx_file_path, folder='text_encoder_quant', float_bit=32, quant_txt='./output/data/input_clip_b1.txt')


    # # unet
    model = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    onnx_file_path = "onnx/unet.onnx"
    input0 = torch.randn((batch_size, unetc, latend_h, latend_h))
    input1 = torch.tensor([999.0])
    input2 = torch.randn((batch_size, text_length, text_dim))
    inputs = tuple([input0, input1, input2])
    input_names = ['latent', 'time', 'text_emb']
    output_names = ['output_latend']
    if not os.path.exists(onnx_file_path):
        torch.onnx.export(
                model,
                inputs,
                onnx_file_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
            )
    param = "--input_layout text_emb NFC"
    if not args.export_quant_model:
        qnn_onnx_converter(onnx_file_path, folder='unet_float', float_bit=32, extra_param=param)
    else:
        qnn_onnx_converter(onnx_file_path, folder='unet_quant', float_bit=32, extra_param=param, quant_txt='./output/data/unet/unet_input_list.txt')


    ## VAE decoder
    model = AutoencoderKLDecoder.from_pretrained(model_id, subfolder="vae")
    onnx_file_path = "onnx/vae_decoder.onnx"
    input0 = torch.randn((1, unetc, latend_h, latend_h))
    inputs = tuple([input0,])
    input_names = ['latent']
    output_names = ['output_image']
    if not os.path.exists(onnx_file_path):
        torch.onnx.export(
                model,
                inputs,
                onnx_file_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
            )
    if not args.export_quant_model:
        qnn_onnx_converter(onnx_file_path, folder='vae_decoder_float', float_bit=32)
    else:
        qnn_onnx_converter(onnx_file_path, folder='vae_decoder_quant', float_bit=32, quant_txt='output/data/vae/vae_input_list.txt')

