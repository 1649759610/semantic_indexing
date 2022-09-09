# 无监督语义匹配模型 [SimCSE](https://aclanthology.org/2021.emnlp-main.552.pdf)

[SimCSE](https://aclanthology.org/2021.emnlp-main.552.pdf) 是基于对比学习方式建模的无监督语义索引模型，其基于Dropout策略构造样本对，同时使用对比学习方式进行训练，取得了SOTA的效果。利用SimCSE模型抽取的句向量，可被广泛应用于搜索引擎，智能问答等领域。

## 快速开始
### 代码结构说明

以下是本项目主要代码结构及说明：

```
DiffCSE/
├── model.py # DiffCSE 模型组网代码
├── data.py # 无监督语义匹配训练数据、测试数据的读取逻辑
├── run_simcse.py # 模型训练、评估、预测的主脚本
├── utils.py # 包括一些常用的工具式函数
├── run_train.sh # 模型训练的脚本
├── run_eval.sh # 模型评估的脚本
└── run_infer.sh # 模型预测的脚本
```

### 模型训练
默认使用无监督模式进行训练 SimCSE，模型训练数据的数据样例如下所示，每行表示一条训练样本：
```shell
全年地方财政总收入3686.81亿元，比上年增长12.3%。
“我对案情并不十分清楚，所以没办法提出批评，建议，只能希望通过质询，要求检察院对此做出说明。”他说。
据调查结果显示：2015年微商行业总体市场规模达到1819.5亿元，预计2016年将达到3607.3亿元，增长率为98.3%。
前往冈仁波齐需要办理目的地包含日喀则和阿里地区的边防证，外转沿途有一些补给点，可购买到干粮和饮料。
```

可以运行如下命令，开始模型训练并且进行模型测试。

```shell
gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
	run_simcse.py \
	--mode "train" \
	--model_name_or_path "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--train_set_file "your train_set path" \
	--eval_set_file "your dev_set path" \
	--save_dir "checkpoints" \
	--log_dir ${log_dir} \
	--save_steps "5000" \
	--eval_steps "100" \
	--batch_size "32" \
	--epochs "3" \
	--learning_rate "3e-5" \
	--weight_decay "0.01" \
	--warmup_proportion "0.01" \
	--dropout "0.1" \
	--dup_rate "0.0" \
	--seed "0" \
	--device "gpu"
```

可支持配置的参数：
* `mode`：可选，用于指明本次运行是模型训练、模型评估还是模型预测，仅支持[train, eval, infer]三种模式；默认为 train。
* `model_name_or_path`：可选，用于训练SimCSE的基座模型；默认为 rocketqa-zh-dureader-query-encoder。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `output_emb_size`：可选，向量抽取模型输出向量的维度；默认为32。
* `train_set_file`：可选，用于指定训练集的路径。
* `eval_set_file`：可选，用于指定验证集的路径。
* `save_dir`：可选，保存训练模型的目录；
* `log_dir`：可选，训练训练过程中日志的输出目录；
* `save_steps`：可选，用于指定模型训练过程中每隔多少 step 保存一次模型。
* `eval_steps`：可选，用于指定模型训练过程中每隔多少 step，使用验证集评估一次模型。
* `epochs`: 模型训练轮次，默认为3。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune 的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.01。
* `warmup_proportion`：可选，学习率 warmup 策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到 learning_rate, 而后再缓慢衰减，默认为0.01。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选 cpu 或 gpu。如使用 gpu 训练则参数 gpus 指定GPU卡号。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── best
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
└── ...
```

### 模型评估
在模型评估时，需要使用带有标签的数据，以下展示了几条模型评估数据样例，每行表示一条训练样本，每行共计包含3列，分别是query1， query2， label：
```shell
右键单击此电脑选择属性，如下图所示   右键单击此电脑选择属性，如下图所示   5
好医生解密||是什么，让美洲大蠊能美容还能救命    解密美洲大蠊巨大药用价值        1
蒜香蜜汁烤鸡翅的做法    外香里嫩一口爆汁蒜蓉蜜汁烤鸡翅的做法    3
项目计划书 篇2  简易项目计划书（参考模板）      2
夏天幼儿园如何正确使用空调？    老师们该如何正确使用空调，让孩子少生病呢？      3
```

可以运行如下命令，进行模型评估。
```shell
gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_eval"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
	run_simcse.py \
	--mode "eval" \
	--model_name_or_path "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--eval_set_file "your dev_set set" \
	--ckpt_dir "./checkpoints/best" \
	--batch_size "8" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.3" \
	--seed "0" \
	--device "gpu"

```
可支持配置的参数：
* `ckpt_dir`: 用于指定进行模型评估的checkpoint路径。

其他参数解释同上。

### 基于动态图模型预测
在模型预测时，需要给定待预测的两条文本，以下展示了几条模型预测的数据样例，每行表示一条训练样本，每行共计包含2列，分别是query1， query2：
```shell
韩国现代摩比斯2015招聘  韩国现代摩比斯2015校园招聘信息
《DNF》封号减刑方法 被封一年怎么办?     DNF封号减刑方法 封号一年怎么减刑
原神手鞠游戏三个刷新位置一览    手鞠游戏三个刷新位置一览
```

可以运行如下命令，进行模型预测：
```shell
gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_infer"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
    run_simcse.py \
    --mode "infer" \
    --model_name_or_path "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--infer_set_file "your infer_set data" \
	--ckpt_dir "./checkpoints/best" \
	--batch_size "16" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.3" \
	--seed "0" \
	--device "gpu"
```

可支持配置的参数：
* `infer_set_file`: 可选，用于指定测试集的路径。
* `save_infer_path`: 可选，用于保存模型预测结果的文件路径。

其他参数解释同上。 待模型预测结束后，会将结果保存至save_infer_path参数指定的文件中。


## Reference
[1] Gao T , Yao X , Chen D . SimCSE: Simple Contrastive Learning of Sentence Embeddings[C]// 2021.
