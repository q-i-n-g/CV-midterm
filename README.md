# CV-midterm
微调ResNet，Inception v3,，Alexnet等卷积神经网络用于鸟类分类，使用CUB-200-2011数据集

## 环境配置
python>=3.8，cuda=12.1

如果cuda版本不同需要下载相应的torch，torchvision版本
```
pip install -r /path/to/requirements.txt
```

## 数据准备
数据来源：

https://data.caltech.edu/records/65de6-vp158

下载后解压到当前根目录中，结构如下：
```
├───CUB_200_2011
│   ├───.idea
│   │   └───inspectionProfiles
│   ├───attributes
│   ├───images
│   └───parts
```



## 权重下载
权重地址：

https://drive.google.com/drive/folders/1PGs9KaQn7n5e5E64yWaJqPYDClflHkne?usp=sharing

下载后放入root/model/final_model中，结构如下：
```
├───models
│   └───final_model
│       │   config.json
│       │
│       └───inceptionv3_CUB
│           │   config.json
│           │   model_history.pkl
│           │   model_weights.pth
│           │
│           └───tensorboard_logs
│                   events.out.tfevents.1716489396.8f289ae79cff.2000575.0
```

## 测试
```
python test.py
```
测试其他的模型修改test.py文件中的路径即可

## 训练
训练最终模型：
```
python main.py
```
