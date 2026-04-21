#数据加载—模型构建—训练—测试

import torch
import model
from multiprocessing import Process
import Biodata
import train
import argparse
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def foo(i):
    print(" This is Process ", i)

def main():
    # 命令行参数解析（去掉mode参数）
    parser = argparse.ArgumentParser(description='DNA Promoter Prediction')
    parser.add_argument('--encoding', type=str, default='dna2vec',
                        choices=['dna2vec', 'ncp', 'dpp'],
                        help='Encoding method: dna2vec, ncp, dpp')
    parser.add_argument('--dataset', type=str, default='H_TATA',
                        choices=['H_TATA', 'H_NonTATA', 'M_TATA', 'M_NonTATA', 'Plants'],
                        help='Dataset to use')
    args = parser.parse_args()
    
    print(f"Using encoding: {args.encoding}")
    print(f"Dataset: {args.dataset}")
    
    # 根据dataset参数设置文件路径
    if args.dataset == 'H_TATA':
        pos_train_files = ['datasets/H.spanies_TATA/Pos_train.fasta']
        neg_train_files = ['datasets/H.spanies_TATA/Neg_train.fasta']
        pos_test_files = ['datasets/H.spanies_TATA/Pos_test.fasta']
        neg_test_files = ['datasets/H.spanies_TATA/Neg_test.fasta']
        model_name = f'H_TATA_{args.encoding}.pt'
    elif args.dataset == 'H_NonTATA':
        pos_train_files = ['datasets/H.spanies_Non_TATA/Pos_train.fasta']
        neg_train_files = ['datasets/H.spanies_Non_TATA/Neg_train.fasta']
        pos_test_files = ['datasets/H.spanies_Non_TATA/Pos_test.fasta']
        neg_test_files = ['datasets/H.spanies_Non_TATA/Neg_test.fasta']
        model_name = f'H_NonTATA_{args.encoding}.pt'
    elif args.dataset == 'M_TATA':
        pos_train_files = ['datasets/M.musculus_TATA/Pos_train.fasta']
        neg_train_files = ['datasets/M.musculus_TATA/Neg_train.fasta']
        pos_test_files = ['datasets/M.musculus_TATA/Pos_test.fasta']
        neg_test_files = ['datasets/M.musculus_TATA/Neg_test.fasta']
        model_name = f'M_TATA_{args.encoding}.pt'
    elif args.dataset == 'M_NonTATA':
        pos_train_files = ['datasets/M.musculus_Non_TATA/Pos_train.fasta']
        neg_train_files = ['datasets/M.musculus_Non_TATA/Neg_train.fasta']
        pos_test_files = ['datasets/M.musculus_Non_TATA/Pos_test.fasta']
        neg_test_files = ['datasets/M.musculus_Non_TATA/Neg_test.fasta']
        model_name = f'M_NonTATA_{args.encoding}.pt'
    elif args.dataset == 'Plants':
        pos_train_files = ['datasets/Plants/Pos_train.fasta']
        neg_train_files = ['datasets/Plants/Neg_train.fasta']
        pos_test_files = ['datasets/Plants/Pos_test.fasta']
        neg_test_files = ['datasets/Plants/Neg_test.fasta']
        model_name = f'Plants_{args.encoding}.pt'
    
    # 根据encoding设置embedding_dim输入维度
    if args.encoding == 'dna2vec':
        embedding_dim = 100
        k_value = 6  # DNA2Vec需要k-mer
    elif args.encoding == 'ncp':
        embedding_dim = 3
        k_value = 1  # NCP不需要k-mer，但Biodata需要k参数
    elif args.encoding == 'dpp':
        embedding_dim = 6
        k_value = 1  # DPP不需要k-mer
    
    # 第一步：训练模型
    print("\n" + "="*50)
    print("STEP 1: Training Model")
    print("="*50)
    
   

    print("Loading training data...")
    data_train = Biodata.Biodata(pos_train_files, neg_train_files, k=k_value, encoding=args.encoding)
    
    train_set = data_train.encode_seq(thread=4)
    
    # 创建模型
    train_model = model.CNNBiLSTMAttention(embedding_dim=embedding_dim).to(device)
    
    print(f"Starting training... Model will be saved as {model_name}")
    train.train_model(train_set, train_model, model_name=model_name)
    
    # 第二步：测试模型
    print("\n" + "="*50)
    print("STEP 2: Testing Model")
    print("="*50)
    
    print("Loading test data...")
    data_test = Biodata.Biodata(pos_test_files, neg_test_files, k=k_value, encoding=args.encoding)
    test_set = data_test.encode_seq(thread=4)
    
    print(f"Testing model: {model_name}")
    train.test_model(test_set, model_name=model_name)
    
    print("\n" + "="*50)
    print("Training and testing completed!")
    print("="*50)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()