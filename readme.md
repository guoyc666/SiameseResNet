# 小样本分类

本代码适用于小样本分类。原先`train.py`使用孪生网络的方式训练，但效果反而比预训练模型还差。于是就`fine_tuning.py`使用普通的分类任务进行微调。

预训练模型是resnet50或101，替换最后一层分类头。初始效果在0.9左右，训练后能达到0.99。

`split_dataset.py` 用于划分训练集、验证集、测试集。

`make_test_data.py` 从测试文件夹中随机生成若干个task，每个task选择若干类中的一些图片构成支持集和查询集。

`myrun.py` 运行在task根目录，由 `test.py` 调用。

训练集形式：

```txt
train/
    class0/
    class1/
    ...
```

测试集形式

```txt
test/
    task1/
        support/
        query/
    task2/
        ...
    ...
```
