from cgitb import text
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import CLIPTokenizer
import torch
from utils import *
import os
import random
from portrait_prompt import prompts, neg_prompt


model_id = '/data1/chenjun/huf/portrait-finetuned'

# create sub-directory per model
def create_directory(name, model_dir="output/data"):
    cur_dir = model_dir + "/" + name
    if not os.path.exists(cur_dir): os.mkdir(cur_dir)
    cur_input_dir = cur_dir + "/" + name + "_inputs"
    if not os.path.exists(cur_input_dir): os.mkdir(cur_input_dir)
    return cur_dir, cur_input_dir


def create_clip(name='input_clip.txt'):

    dir, input_dir = create_directory("clip")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    with open(f"{dir}/{name}", 'w') as outf:
        for i, prompt in enumerate(prompts):
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            image = text_input.input_ids
            path = os.path.realpath(os.path.join(input_dir, f'clip_input_b1_{i:04d}.bin'))
            save_data(image, path)
            outf.write(f'{path}\n')
            if i == 10:
                break


# load U-Net inputs
def create_unet_inputs():
    unet_dir, unet_input_dir = create_directory("unet")

    pickle_path = 'output/fp32.npy'
    src = np.load(pickle_path, allow_pickle=True)

    with open(unet_dir + "/unet_input_list.txt", "w") as f:
        for i, (latent, time_emb, hidden) in enumerate(src[0]["unet_inputs"]):
            latent_entry = os.path.realpath(f"{unet_input_dir}/unet_input_latent_{i+1}_1.bin")
            latent = latent.transpose(0,2,3,1)
            # latent = np.concatenate([latent, latent], axis=0)
            latent.tofile(latent_entry)                  # qnn中为NHWC排列
            timm_emb_entry = os.path.realpath(f"{unet_input_dir}/unet_input_time_embedding_{i+1}_2.bin")
            time_emb = time_emb.astype(np.float32).reshape([1,1])
            # time_emb = np.concatenate([time_emb, time_emb], axis=0)
            time_emb.tofile(timm_emb_entry)
            hidden_entry = os.path.realpath(f"{unet_input_dir}/unet_input_hidden_{i+1}_3.bin")
            # hidden = np.concatenate([hidden, hidden], axis=0)
            hidden.tofile(hidden_entry)
            # Write to input_list.txt; ensure 3 inputs in one line
            f.write(latent_entry + " " + timm_emb_entry + " " + hidden_entry + "\n")       


def create_vae():
    dir, input_dir = create_directory("vae")

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cpu")

    with open(dir + '/vae_input_list.txt', 'w') as f:
        for i in range(10):
            prompt = prompts[0]
            step = 10
            image = pipe(prompt=prompt,negative_prompt=neg_prompt, num_inference_steps=step, output_type='latent').images[0]  
            image = image.permute([1,2,0])
            path = os.path.realpath(os.path.join(input_dir, f'vae_{i:04d}.bin'))
            save_data(image, path)
            f.write(f'{path}\n')







if __name__ == '__main__':

    create_clip()
    create_unet_inputs()
    create_vae()

