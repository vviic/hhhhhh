#负责把原始 DNA 序列变成“样本矩阵 + 标签”的训练数据

import encode
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
from functools import partial
import torch
from torch.utils.data import Dataset
import re


class BioDataset(Dataset):
    def __init__(self, features, labels):
        """
        :param features: (n_samples, seq_len, num_features)
        :param labels: (n_samples,)
        """
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Returns a sample at the given index idx.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a sample at the given index idx.
        :param idx: index of the sample
        """
        return self.features[idx], self.labels[idx]


class Biodata:
    # 增加encoding参数
    def __init__(self, pos_fasta_files, neg_fasta_files, k=6, encoding='dna2vec'):
        self.feature = None
        self.dna_seq = {}
        self.label = []

        self.encoding = encoding # 存储encoding

        for pos_fasta_file in pos_fasta_files:
            for seq_record in SeqIO.parse(pos_fasta_file, "fasta"):
                sequence = str(seq_record.seq).upper()
                sequence = re.sub(r'[^ATCG]', 'C', sequence)
                if sequence:
                    if seq_record.description not in self.dna_seq:
                        self.dna_seq[seq_record.description] = sequence
                        self.label.append(1)
        for neg_fasta_file in neg_fasta_files:
            for seq_record in SeqIO.parse(neg_fasta_file, "fasta"):
                sequence = str(seq_record.seq).upper()
                sequence = re.sub(r'[^ATCG]', 'T', sequence)
                if sequence:
                    if seq_record.description not in self.dna_seq:
                        self.dna_seq[seq_record.description] = sequence
                        self.label.append(0)
        self.k = k
        self.label = np.array(self.label)

    # 根据self.encoding命令传参选择编码函数
    def encode_seq(self, thread):
      print(f"encoding sequences with {self.encoding} method...")
      seq_list = list(self.dna_seq.values())
      
      # 多进程：使用 with 语句管理进程池
      try:
          if self.encoding == 'dna2vec':
              partial_encode_seq = partial(encode.matrix_encoding, k=self.k)
          elif self.encoding == 'ncp':
              partial_encode_seq = encode.ncp_encoding
          elif self.encoding == 'dpp':
              partial_encode_seq = encode.dpp_encoding
          else:
              raise ValueError(f"Unsupported encoding type: {self.encoding}")
          
          # 使用 with 语句确保进程池正确关闭
          with Pool(thread) as pool:
              self.feature = np.array(pool.map(partial_encode_seq, seq_list))
          
      except Exception as e:
          print(f"Error in encoding: {e}")
          # 多进程失败就回退到单进程处理
          print("Falling back to single-threaded processing...")
          self.feature = []
          for seq in seq_list:
              if self.encoding == 'dna2vec':
                  encoded = encode.matrix_encoding(seq, k=self.k)
              elif self.encoding == 'ncp':
                  encoded = encode.ncp_encoding(seq)
              elif self.encoding == 'dpp':
                  encoded = encode.dpp_encoding(seq)
              self.feature.append(encoded)
          self.feature = np.array(self.feature)
      
      dataset = BioDataset(self.feature, self.label)
      return dataset