  main.py	               主程序入口，支持参数选择编码方式和数据集
  
  model.py	              CNN-BiLSTM-Attention模型定义
  
  Biodata.py	            数据加载与编码
  
  encode.py	             三种编码实现：DNA2Vec、NCP、DPP
  
  train.py	              训练、测试、指标计算
  
  plot_linechart.py	     性能对比折线图


实验针对人类TATA、人类非TATA、小鼠TATA、小鼠非TATA以及植物共五个启动子数据集，分别采用DNA2Vec、NCP和DPP三种编码方式，依次执行模型训练与测试。每条命令对应一种“数据集 + 编码方式”的组合，共15组实验。运行后会自动输出ACC、SN、SP、PRE、F1、MCC、Err七个分类性能指标，用于后续对比分析。

命令格式：

python main.py --encoding <编码方式> --dataset <数据集名称>

参数说明：

<编码方式>：dna2vec / ncp / dpp

<数据集名称>：H_TATA / H_NonTATA / M_TATA / M_NonTATA / Plants

示例命令（使用DPP编码，训练并测试人类TATA数据集）：

python main.py --encoding dpp --dataset H_TATA

消融实验，采用多种模型：

CNN.py：只保留 CNN

LSTM.py：只保留 BiLSTM

CNN-LSTM.py：CNN + BiLSTM

CNN-ATT.py：CNN + Attention

LSTM-ATT.py：BiLSTM + Attention

model.py：CNN + BiLSTM + Attention（完整版）
