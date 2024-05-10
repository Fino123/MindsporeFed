## clientX and clientXtest
前者是第X个客户端训练用的数据，后者是第X个客户端测试用的数据。
其中每个文件夹包含三个文件：
1. label-int-5000.txt，包含数据的标签相关的数据，一行对应一个样本
2. mask-int-5000.txt，包含已经安装的APP的序号，用于对测试结果进行mask（忽略没有安装的app的预测概率），一行对应一个样本。
3. test-data-int-5000.txt，包含用户的行为数据，一行对应一个样本。

## 其它数据文件
id2app_500_with_minor.json：一个映射app名称和id的json文件。