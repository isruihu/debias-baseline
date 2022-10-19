## 模型
网络采用 ResNet 结构，使用 torchvision 里已经定义好的 resnet18 模型


## 数据集
数据集包括部分训练数据集和测试数据集

其中训练测试集是有偏的，每个类别下主偏见属性图片 ：副偏见属性图片 约为 95 ：5，且主偏见属性种类为 1，副偏见属性种类为 9

对应 align : conflict 约为 95 ：5，align中每个类别只有一种主偏见属性，conflict中每个类别有9种副偏见属性

测试集是无偏的，即每个类别中每种偏见属性的图片数量相近


## 运行环境
通用 torch 环境即可，无硬性版本限制。用到的部分依赖包列表如下：
torch
torchvison
numpy
glob
argparse
pillow
tqdm


## 测试结果
测试结果为模型对测试数据的类别分类结果，所有结果保存在result.json文件中，文件中每一项为一张测试图片名称及其对应的类别


## 目录结构
|-- dataset                              # 数据集文件夹

|-- dataloader.py                        # 数据集加载文件

|-- README.md                            # 项目说明文件

|-- run.py                               # 模型训练加测试的脚本，可直接生成测试结果

|-- test.py                              # 模型测试文件

|-- train.py                      	     # 模型训练文件

|-- utils.py                      	     # 辅助功能文件


## 启动脚本
#### Train 
```python train.py```
#### Test 
```python test.py```
#### Train + Test 
```python run.py```