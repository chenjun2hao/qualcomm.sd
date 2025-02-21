# 从零开始在高通手机上部署sd（一）

从零基础开始，在自己的高通手机(骁龙8 gen1+)上用NPU跑文生图stable diffusion模型。包含：
- 高通qnn下载安装
- sd模型浮点/量化导出
- 在高通手机上用cpu跑浮点模型，htp跑量化模型

## 1. python依赖安装
主要对齐transformers, diffusers的版本，其他参考requirements.txt
```
pip install transformers==4.40.0 diffusers==0.32.2
```


## 2. python浮点模型测试
```bash
# 1.下载sd 模型
git clone https://hf-mirror.com/segmind/portrait-finetuned

# 2.通过diffuser的pipe测试模型
cd $PROJECT_HOME
python sd_portrait_diffpip.py

# 3.通过自己实现的pipe测试模型
python sd_portrait_ownpip.py
```
结果可视化：
<div style="display: flex; justify-content: center;">
    <img src="./output/image.png" alt="diffusers pipe结果" style="width: 45%; margin: 0 2.5%; display: inline-block;">
    <img src="./output/generated.png" alt="own pipe结果" style="width: 45%; margin: 0 2.5%; display: inline-block;">
</div>


## 3. 导出浮点模型/x64 cpu模拟推理/高通cpu推理
高通libQnnCpu.so好像只支持浮点模型，不支持量化模型（可能后续的硬件，会支持）。

### 3.1 高通qnn安装
coming soon ......

### 3.2 导出浮点模型
```
cd $PROJECT_HOME
python export_model.py --export_quant_model false
```

导出的模型在`qnn_models/xxx`下. eg: `qnn_models/text_encoder_float/x86_64-linux-clang/libtext_encoder.so`是x64上模拟推理时需要的模型。`qnn_models/text_encoder_float/aarch64-android/libtext_encoder.so`是push到android手机上使用的模型

### 3.3 x64 cpu模拟推理
参考项目readme运行:[qualcomm.ai](https://github.com/chenjun2hao/qualcomm.ai)

### 3.4 高通cpu推理
将所有依赖的东西push到android手机上。参考项目readme运行:[qualcomm.ai](https://github.com/chenjun2hao/qualcomm.ai)



## 4. 导出量化模型/高通htp推理

### 4.1 生成量化用数据
```bash
cd $PROJECT_HOME
python make_calibration_data.py
```

### 4.2 导出量化模型

### 4.3 x64模拟htp推理
高通在x64平台上，有模拟HTP硬件执行的软件库，但是sd模型运行太慢了。

### 4.4 高通HTP推理
将所有依赖的东西push到android手机上。参考项目readme运行:[qualcomm.ai](https://github.com/chenjun2hao/qualcomm.ai)


## 5. reference
1. [StableDiffusionOnDevice](https://github.com/XiaoMi/StableDiffusionOnDevice)