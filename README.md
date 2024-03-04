## 一键创造你的虚拟人物对话机器人

Character-Chat 通过调用闭源大模型API生成合成数据，对开源大语言模型进行有监督微调（Supervised Fine Tuning），实现对任意角色对话模型的快速本地部署。最少**只需要输入角色名**，即可在个人PC上完成自动化的训练和部署，同时也支持自定义微调参数设置。

:tada:解决缺少SFT的数据痛点

:rocket:支持丰富的微调参数设置

:thumbsup:从训练到对话全流程自动化

:smile:可以本地运行，隐私无忧

:computer:最低只需要8G显存即可训练和部署

### 环境搭建

**Step 1: 安装cuda和pytorch**

确保`cuda`已安装，建议版本[CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)，请根据自己的操作系统选择下载安装。

在Torch官网上寻找合适的版本进行下载安装，请根据自己所安装的`cuda`版本和操作系统选择合适的版本进行安装。建议版本[Stable(2.2.1)](https://pytorch.org/get-started/locally/)。

**Step 2: 配置python环境**

Linux：

```bash
git clone https://github.com/WhiteBLANKN/character-chat.git
cd character-chat
conda create -n character-chat python=3.11
conda activate character-chat
pip install bitsandbytes
pip install -r requirements.txt
```

Windows：

```bash
git clone https://github.com/WhiteBLANKN/character-chat.git
cd character-chat
conda create -n character-chat python=3.11
conda activate character-chat
pip install -r requirements.txt
```

`BitsAndBytes`在Windows环境下运行报错，建议下载此项目下的安装包手动进行安装[bitsandbytes-windows-webui](https://github.com/jllllll/bitsandbytes-windows-webui)。下载完成后可以使用此命令手动安装，替换`/path_to_your_package`为文件实际下载地址：

```bash
pip install /path_to_your_package
```

### 开始训练

Windows 参考指令：

```bash
python main.py --output_dir ./output_dir --model_name_or_path meta-llama/Llama-2-7b-hf  --character 海绵宝宝 --download_from_modelscope False --ernie_api_key your_ernie_api_key --ernie_secret_key your_ernie_secret_key --dataset_length 1000 --overwrite_output_dir True --gradient_accumulation_steps 4 --per_device_train_batch_size 1 --learning_rate 2e-5 --warmup_steps 50 --save_steps 300 --num_train_epochs 3 --optim paged_adamw_8bit --logging_steps 1
```

Linux 参考指令：

```bash
python main.py \
--output_dir ./output_dir \
--model_name_or_path shakechen/Llama-2-7b-hf \
--character 海绵宝宝 \
--download_from_modelscope True \
--ernie_api_key your_ernie_api_key \
--dataset_length 1000 \
--overwrite_output_dir True \
--gradient_accumulation_steps 4 \
--per_device_train_batch_size 1 \
--learning_rate 2e-5 \
--warmup_steps 50 \
--save_steps 300 \
--num_train_epochs 3 \
--optim paged_adamw_8bit \
--logging_steps 1
```

:warning:**请将`--ernie_api_key`与`--ernie_secret_key`替换为你实际的开发者API密钥**

如果要使用OpenAI 的API进行数据生成，请将上面两个参数删除，新增`--openai_api_key`并传入你的API key，例如：

```bash
python main.py \
--openai_api_key sk-xxxxxxxxxxxxxxxxxxxxx \
...#同上
```

**关键参数说明：**

`--output_dir`：训练完成后的模型的保存路径。

`--model_name_or_path`：本地模型路径或模型ID。

<u>请查阅Hugging Face和魔塔社区的相关下载链接，例如llama2 7b的从Hugging Face下载是`meta-llama/Llama-2-7b-hf`，从魔塔社区进行下载则是`shakechen/Llama-2-7b-hf`</u>

`--character`：训练对象，虚拟角色人物名，可自定义。

`--download_from_modelscope`：是否从魔塔社区下载模型，部分模型在Hugging Face上需要认证，且国内存在网络问题。若设置为False，将从本地读取或从Hugging Face下载。

`--dataset_length`：调用API所生成的数据集长度，建议长度大于1000条。若数据较少，稍微增加训练轮次`--num_train_epochs`，让模型稍微过拟合一下，有助于提升对话表现。

`--per_device_train_batch_size`：如果你的显存比较大，可以用更大的batch size进行训练，可以让训练更快完成。在默认设置的qlora微调方法下，微调一个7b的模型所需要的显存约为8G。

其他参数可参考项目src/arg_parser中定义的dataclass类。

### 开始对话

使用以下指令可以自动加载模型及参数，并开始对话：

```bash
python chat_terminal.py
```

### TO DO

:white_large_square:：新增对齐算法，加强对合成数据集的利用效率。

:white_large_square:：测试更多的开源模型，提高兼容性。

:white_large_square:：支持手动更新数据，偏好标注。

:white_large_square:：一个对话图形界面
