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
