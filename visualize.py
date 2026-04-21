# 折线图 - Y轴从0.7开始

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import math
import os

import Biodata
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 测试函数 ============
def test_model_and_return_metrics(dataset, model_name, val_split=0.5):
    random.seed(42)
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    testset = [dataset[i] for i in data_list if i in test_list]
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=True, drop_last=True)
    
    loaded_model = torch.load(model_name, map_location=device, weights_only=False)
    loaded_model.eval()
    loaded_model.to(device)

    TP, FN, FP, TN = 0, 0, 0, 0
    for data in test_loader:
        with torch.no_grad():
            inputs = data[0].to(device), data[1].to(device)
            labels = data[1].to(device)
            pred = loaded_model(inputs)
            pred = pred.argmax(dim=1)
            
            for idx, label in enumerate(labels):
                if label == 1:
                    if label == pred[idx]:
                        TP += 1
                    else:
                        FN += 1
                elif label == pred[idx]:
                    TN += 1
                else:
                    FP += 1
    
    SN = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    ACC = (TP + TN) / (TP + FN + FP + TN) if (TP + FN + FP + TN) > 0 else 0
    MCC_num = (TP * TN - FP * FN)
    MCC_den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) > 0 else 1
    MCC = MCC_num / MCC_den
    F1 = (2 * TP) / (2 * TP + FN + FP) if (2 * TP + FN + FP) > 0 else 0
    
    return {'ACC': ACC, 'SN': SN, 'SP': SP, 'MCC': MCC, 'F1': F1}


# ============ 配置所有模型 ============
models_to_test = [
    ('H_TATA_dna2vec.pt', 'Human', 'TATA', 'dna2vec', 6),
    ('H_TATA_ncp.pt', 'Human', 'TATA', 'ncp', 1),
    ('H_TATA_dpp.pt', 'Human', 'TATA', 'dpp', 1),
    ('H_NonTATA_dna2vec.pt', 'Human', 'NonTATA', 'dna2vec', 6),
    ('H_NonTATA_ncp.pt', 'Human', 'NonTATA', 'ncp', 1),
    ('H_NonTATA_dpp.pt', 'Human', 'NonTATA', 'dpp', 1),
    ('M_TATA_dna2vec.pt', 'Mouse', 'TATA', 'dna2vec', 6),
    ('M_TATA_ncp.pt', 'Mouse', 'TATA', 'ncp', 1),
    ('M_TATA_dpp.pt', 'Mouse', 'TATA', 'dpp', 1),
    ('M_NonTATA_dna2vec.pt', 'Mouse', 'NonTATA', 'dna2vec', 6),
    ('M_NonTATA_ncp.pt', 'Mouse', 'NonTATA', 'ncp', 1),
    ('M_NonTATA_dpp.pt', 'Mouse', 'NonTATA', 'dpp', 1),
    ('Plants_dna2vec.pt', 'Plants', 'All', 'dna2vec', 6),
    ('Plants_ncp.pt', 'Plants', 'All', 'ncp', 1),
    ('Plants_dpp.pt', 'Plants', 'All', 'dpp', 1),
]

dataset_config = {
    'H_TATA': {'pos_test': ['datasets/H.spanies_TATA/Pos_test.fasta'], 'neg_test': ['datasets/H.spanies_TATA/Neg_test.fasta']},
    'H_NonTATA': {'pos_test': ['datasets/H.spanies_Non_TATA/Pos_test.fasta'], 'neg_test': ['datasets/H.spanies_Non_TATA/Neg_test.fasta']},
    'M_TATA': {'pos_test': ['datasets/M.musculus_TATA/Pos_test.fasta'], 'neg_test': ['datasets/M.musculus_TATA/Neg_test.fasta']},
    'M_NonTATA': {'pos_test': ['datasets/M.musculus_Non_TATA/Pos_test.fasta'], 'neg_test': ['datasets/M.musculus_Non_TATA/Neg_test.fasta']},
    'Plants': {'pos_test': ['datasets/Plants/Pos_test.fasta'], 'neg_test': ['datasets/Plants/Neg_test.fasta']},
}

# ============ 评估所有模型 ============
results = []
print("评估所有模型...\n")

for model_file, species, ptype, encoding, k_val in models_to_test:
    if not os.path.exists(model_file):
        print(f"❌ {model_file} 不存在")
        continue
    
    if species == 'Human':
        dataset_key = f'H_{ptype}'
    elif species == 'Mouse':
        dataset_key = f'M_{ptype}'
    else:
        dataset_key = 'Plants'
    
    try:
        cfg = dataset_config[dataset_key]
        data_test = Biodata.Biodata(cfg['pos_test'], cfg['neg_test'], k=k_val, encoding=encoding)
        test_set = data_test.encode_seq(thread=4)
        metrics = test_model_and_return_metrics(test_set, model_file, val_split=0.5)
        
        results.append({
            'species': species, 'type': ptype, 'encoding': encoding,
            'ACC': metrics['ACC'], 'SN': metrics['SN'], 'SP': metrics['SP'],
            'MCC': metrics['MCC'], 'F1': metrics['F1']
        })
        print(f"✓ {model_file}: ACC={metrics['ACC']:.4f}, MCC={metrics['MCC']:.4f}")
    except Exception as e:
        print(f"❌ {model_file} 失败: {e}")

df = pd.DataFrame(results)
print("\n" + "="*60)
print(df.to_string(index=False))
df.to_csv('evaluation_results.csv', index=False)

# ============ 折线图（Y轴从0.7开始） ============
metrics_to_plot = ['ACC', 'SN', 'SP', 'MCC', 'F1']
colors = {'dna2vec': '#4472C4', 'ncp': '#E07257', 'dpp': '#70AD47'}
markers = {'dna2vec': 'o', 'ncp': 's', 'dpp': '^'}

groups = [
    ('Human', 'TATA'), ('Human', 'NonTATA'),
    ('Mouse', 'TATA'), ('Mouse', 'NonTATA'),
    ('Plants', 'All')
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (species, ptype) in enumerate(groups):
    ax = axes[idx]
    group_data = df[(df['species'] == species) & (df['type'] == ptype)]
    
    if len(group_data) == 0:
        ax.set_title(f'{species} - {ptype} (No Data)')
        ax.axis('off')
        continue
    
    for enc in ['dna2vec', 'ncp', 'dpp']:
        enc_data = group_data[group_data['encoding'] == enc]
        if len(enc_data) == 0:
            continue
        values = [enc_data[m].values[0] for m in metrics_to_plot]
        ax.plot(metrics_to_plot, values, 
                marker=markers[enc], label=enc.upper(), 
                color=colors[enc], linewidth=2, markersize=8)
        
        # 添加数值标签
        for i, (m, v) in enumerate(zip(metrics_to_plot, values)):
            ax.annotate(f'{v:.3f}', (m, v), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=8)
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0.7, 1.01)  # 关键修改：Y轴从0.7开始
    ax.set_yticks(np.arange(0.70, 1.01, 0.05))  # 设置更细的刻度
    title = f'{species}' + (f' - {ptype}' if ptype != 'All' else '')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

axes[5].axis('off')

plt.suptitle('Performance Comparison of Different Encoding Methods\n(CNN-BiLSTM-Attention Model)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('encoding_comparison_linechart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n✓ 折线图已保存: encoding_comparison_linechart.png")