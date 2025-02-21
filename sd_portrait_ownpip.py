import torch
from transformers import CLIPTokenizer
from redefined_modules.modeling_clip import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
from tqdm import tqdm
from utils import *




class AutoencoderKLDecoder(AutoencoderKL):
    def forward(self, z: torch.FloatTensor):
        z = 1 / 0.18215 * z
        z = self.post_quant_conv(z)
        image = self.decoder(z)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)

        return (image,)


if __name__ == '__main__':
    #
    model_portrait = '/data1/chenjun/huf/portrait-finetuned'
    device = 'cuda'
    dtype = torch.float

    print("Loading pre-trained TextEncoder model")
    text_encoder = CLIPTextModel.from_pretrained(model_portrait, subfolder="text_encoder", torch_dtype=dtype).to(device)
    text_encoder.config.return_dict = False

    tokenizer = CLIPTokenizer.from_pretrained(model_portrait, subfolder="tokenizer")

    print("Loading pre-trained UNET model")
    unet = UNet2DConditionModel.from_pretrained(model_portrait, subfolder="unet", torch_dtype=dtype).to(device)
    unet.config.return_dict = False


    print("Loading pre-trained VAE model")
    vae = AutoencoderKLDecoder.from_pretrained(model_portrait, subfolder="vae", torch_dtype=dtype).to(device)
    vae.config.return_dict = False


    #
    from stable_diff_pipeline import run_the_pipeline, save_image, replace_mha_with_sha_blocks
    # replace_mha_with_sha_blocks(unet)



    prompt = "Faceshot Portrait of pretty young (18-year-old) Caucasian wearing a high neck sweater, sharp focus, BREAK epicrealism"
    image = run_the_pipeline(prompt, unet, text_encoder, vae, tokenizer, test_name='fp32', seed=1.364777111e+14)
    save_image(image.squeeze(0), 'output/generated.png')
