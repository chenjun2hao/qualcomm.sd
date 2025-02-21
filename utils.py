import os
import torch
import numpy as np


def qnn_onnx_converter(onnx_model, folder='default', out_dir='qnn_models', float_bit=32, ACT_BITWIDTH=16, WEIGHTS_BITWIDTH=8, BIAS_BITWIDTH=32,
                       quant_txt='', **kwargv):
    dir = os.path.join(out_dir, folder)
    os.makedirs(dir, exist_ok=True)

    name = os.path.basename(onnx_model).split('.')[0]
    name = os.path.join(dir, name)
    qnn_sdk_root = os.environ["QNN_SDK_ROOT"]
    if not qnn_sdk_root:
        print("Please check QNN_SDK_ROOT in ~/.bashrc")
        exit(1)
    
    if os.name == 'nt':
        qnn_tools_target = 'x86_64-windows-msvc'
    else:
        qnn_tools_target = 'x86_64-linux-clang'

    ## 额外参数
    extra_param = kwargv.get('extra_param', None)

    ## 转cpp，bin文件 
    if not os.path.exists(f'{name}.cpp'):
        print("Converting and compiling QNN models...")
        converter_cmd = f"{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-onnx-converter -i {onnx_model} --float_bitwidth {float_bit} -o {name}.cpp"
        if quant_txt:
            # converter_cmd += f" --use_per_row_quantization --use_per_channel_quantization --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --bias_bitwidth {BIAS_BITWIDTH} --input_list {quant_txt}"
            converter_cmd += f" --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHTS_BITWIDTH} --bias_bitwidth {BIAS_BITWIDTH} --input_list {quant_txt}"
        if extra_param:
            converter_cmd += f" {extra_param}"
        print(converter_cmd)
        if os.name == 'nt':
            converter_cmd = "python " + converter_cmd
        os.system(converter_cmd)

    ## 转so文件
    if not os.path.exists(f'{name}.so'):
        print("Converting QNN so models...")
        converter_cmd = f'{qnn_sdk_root}/bin/{qnn_tools_target}/qnn-model-lib-generator -c {name}.cpp -b {name}.bin -o {dir}'
        print(converter_cmd)
        if os.name == 'nt':
            converter_cmd = "python " + converter_cmd
        os.system(converter_cmd)



def save_data(data, outname='output/torch.bin'):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    data = data.astype(np.float32)
    data.tofile(outname)
    
    
def save_data_int32(data, outname='output/torch.bin'):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    data = data.astype(np.int32)
    data.tofile(outname)
    

def save_data_int64(data, outname='output/torch.bin'):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    data = data.astype(np.int64)
    data.tofile(outname)
    

sd_prompts = [
    "A majestic mountain range with snow-capped peaks under a golden sunset sky.",
    "A fairytale castle surrounded by a moat and a lush green forest.",
    "A futuristic cityscape with flying cars and neon-lit skyscrapers.",
    "A cozy cabin in the woods with smoke rising from the chimney.",
    "A beautiful beach with crystal-clear water and palm trees swaying in the breeze.",
    "A mysterious ancient temple hidden deep in the jungle.",
    "A Victorian-era mansion with a large garden full of blooming flowers.",
    "A group of astronauts exploring an alien planet with strange rock formations.",
    "A steampunk airship sailing through the clouds.",
    "A modern art gallery filled with abstract paintings and sculptures.",
    "A enchanted forest with glowing mushrooms and fairies flitting about.",
    "A medieval knight on horseback charging into battle.",
    "A Japanese tea house in a serene zen garden.",
    "A field of sunflowers stretching as far as the eye can see.",
    "A post-apocalyptic wasteland with broken buildings and overgrown vegetation.",
    "A Venetian gondola gliding along the canals at dusk.",
    "A superhero flying through the city with a cape billowing behind.",
    "A high-tech laboratory with scientists working on cutting-edge experiments.",
    "A safari park with lions, giraffes and zebras roaming freely.",
    "A bakery filled with freshly baked bread and pastries.",
    "A underwater city with colorful coral reefs and schools of fish.",
    "A concert hall filled with people listening to a symphony orchestra.",
    "A ski resort with people skiing down snow-covered slopes.",
    "A haunted house with creaking doors and ghostly apparitions.",
    "A cyberpunk street with holographic advertisements and rain-soaked pavement.",
    "A vineyard with rows of grapevines and workers harvesting grapes.",
    "A motocross track with dirt bikes jumping over ramps.",
    "A lighthouse standing tall on a rocky cliff by the sea.",
    "A Renaissance painting come to life with characters in period costumes.",
    "A tropical rainforest with a waterfall cascading down.",
    "A gothic cathedral with stained glass windows and flying buttresses.",
    "A spaceship docked at a futuristic spaceport.",
    "A diner with a jukebox playing oldies and people eating burgers.",
    "A wildlife sanctuary with endangered animals being cared for.",
    "A fashion show with models walking down the runway in glamorous outfits.",
    "A library filled with antique books and cozy reading nooks.",
    "A hot air balloon floating over a picturesque countryside.",
    "A rock concert with a screaming crowd and a band on stage.",
    "A farm with cows grazing in the pasture and a red barn.",
    "A celestial nebula with swirling colors and stars being born.",
    "A ice cream parlor with a variety of delicious flavors on display.",
    "A Greek amphitheater with a play being performed.",
    "A motocross rider performing stunts in mid-air.",
    "A botanical garden with exotic plants and flowers.",
    "A train chugging through a mountain tunnel.",
    "A snowflake with intricate crystalline patterns.",
    "A surfing beach with big waves and surfers riding them.",
    "A baroque palace with gold decorations and frescoed ceilings.",
    "A space shuttle launching into space with a fiery trail.",
    "A koi pond with colorful fish swimming around.",
    "A cybernetic organism with a humanoid form and glowing implants.",
    "A mountain bike trail winding through a forest.",
    "A Oktoberfest celebration with people drinking beer and dancing.",
    "A cathedral choir singing hymns in a solemn atmosphere.",
    "A pirate ship sailing on the high seas with a Jolly Roger flag.",
    "A desert oasis with a palm tree-lined pool and fresh water.",
    "A roller coaster with twists and turns at a theme park.",
    "A sumptuous banquet hall filled with tables laden with food.",
    "A cowboy on horseback herding cattle on the prairie.",
    "A neon-lit nightclub with people dancing to electronic music.",
    "A Japanese sushi bar with a chef preparing fresh sushi.",
    "A Himalayan monastery perched on a cliff.",
    "A fireworks display over a city skyline.",
    "A marathon runner crossing the finish line with exhaustion.",
    "A woodland elf living in a treehouse in the forest.",
    "A glassblower working in a hot studio creating beautiful vases.",
    "A geodesic dome house in a desert landscape.",
    "A polo match with players on horseback hitting the ball.",
    "A quantum computer laboratory with complex circuitry.",
    "A canoe gliding through a calm lake in a national park.",
    "A Byzantine mosaic depicting religious scenes.",
    "A motocross race with dirt bikes speeding around a track.",
    "A chocolatier making handcrafted chocolates in a kitchen.",
    "A fjord with steep cliffs and a glacier in the background.",
    "A flamenco dancer performing in a Spanish tavern.",
    "A drone flying over a construction site taking aerial photos.",
    "A Victorian street with horse-drawn carriages and gas lamps.",
    "A rugby match with players tackling each other.",
    "A barbershop with a barber cutting hair and chatting with customers.",
    "A polar bear swimming in the Arctic Ocean.",
    "A digital art exhibition with interactive installations.",
    "A cider mill with apples being crushed to make cider.",
    "A paraglider soaring through the air above a valley.",
    "A Tibetan mastiff guarding a mountain pass.",
    "A Venetian carnival with people in elaborate masks.",
    "A trampoline park with kids bouncing around.",
    "A medieval market with merchants selling wares.",
    "A solar panel farm with rows of shiny panels.",
    "A taekwondo studio with students practicing kicks.",
    "A jazz club with a saxophonist playing a solo.",
    "A marble quarry with workers cutting huge blocks of marble.",
    "A synchronized swimming team performing in a pool.",
    "A geodesic dome observatory for stargazing.",
    "A llama farm with cute llamas grazing.",
    "A graffiti artist painting a mural on a wall.",
    "A Viking longship sailing on a rough sea.",
    "A bonsai garden with miniature trees carefully cultivated.",
    "A spelunker exploring a deep cave system.",
    "A quilting bee with women sewing quilts together.",
    "A windmill turning in a Dutch countryside."
]