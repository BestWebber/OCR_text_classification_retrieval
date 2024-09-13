import pymysql
import mysql.connector
import os
os.environ["TORCH_HOME"] = "./torch"
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import mysql.connector
from transformers import AutoModel, BertTokenizerFast
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
# specify GPU
device = torch.device("cuda")

# 定义模型架构
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.5)
        self.fc2 = nn.Linear(512, 3)  # 3 LABELS (根据任务调整)
        self.bn2 = nn.BatchNorm1d(3, momentum=0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.bn1(self.fc1(cls_hs))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = self.softmax(x)
        return x


# 数据库连接函数
def build_connect():
    try:
        cnx = mysql.connector.connect(
            host='49.235.181.131',
            user='ccb',
            password='Abcd@1234',
            database='CCB',
            port=3306
        )
        cursor = cnx.cursor()
        return cnx, cursor
    except mysql.connector.Error as err:
        print(f'Error: {err}')
        return None, None


# 同数据库建立连接并创建一张表存储分类得到的结果
def create_table(table_name):
    try:
        cnx, cursor = build_connect()
        if cnx is None or cursor is None:
            print("Failed to connect to the database!")
            return
        #创建表
        create_table_query = f'''
        CREATE TABLE IF NOT EXISTS {table_name}(
            id INT AUTO_INCREMENT PRIMARY KEY,
            Text TEXT)
        '''
        #执行创建表的语句
        cursor.execute(create_table_query)
        cnx.commit()
        print(f"Table {table_name} has been created!")
    except mysql.connect.Error as err:
        print(f'Error: {err}')


# 插入文本到数据库
def insert_text(table_name, text_value):
    cnx, cursor = build_connect()
    if not cnx or not cursor:
        print("Failed to connect to the database.")
        return
    try:
        insert_query = f"INSERT INTO {table_name} (Text) VALUES (%s)"
        cursor.execute(insert_query, (text_value,))
        cnx.commit()
        print("Text inserted successfully!")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        cnx.close()


# 加载模型并进行预测
def load_test_file(file, model_save_path, table_name):
    # 加载模型
    bert = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext', return_dict=False)
    tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext', return_dict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BERT_Arch(bert)
    model = model.to(device)

    # 加载预训练模型权重
    pretrained_model_state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(pretrained_model_state_dict)
    model.eval()

    # 读取测试数据
    test_file = pd.read_csv(file, encoding_errors='ignore')
    test_text = test_file['Text']

    # Tokenize 输入文本
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=190,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )

    test_seq = torch.tensor(tokens_test['input_ids']).to(device)
    test_mask = torch.tensor(tokens_test['attention_mask']).to(device)

    # 分批处理数据
    batch_size = 16  # 根据 GPU 内存情况调整
    num_batches = (len(test_seq) + batch_size - 1) // batch_size

    all_preds = []

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_seq))

            batch_seq = test_seq[start_idx:end_idx]
            batch_mask = test_mask[start_idx:end_idx]

            preds = model(batch_seq, batch_mask)
            preds = preds.cpu().numpy()

            all_preds.append(preds)

        all_preds = np.concatenate(all_preds, axis=0)

    preds = np.argmax(all_preds, axis=1)

    # 创建表并插入数据
    create_table(table_name)  # 创建 MySQL 表
    for i in range(len(preds)):
        if preds[i] == 1:
            text = test_text.iloc[i]  # 预测为积极的文本
            insert_text(table_name, text)

    # 清空文件内容，一方面节省空间，另一方面保证不会重复添加
    try:
        with open(file, 'w') as f:
            f.truncate(0)   # 清空文件内容
        print(f"{file} has been successfully cleared.")
    except Exception as e:
        print(f"Failed to clear {file}: {e}.")


file = '/marketing_story_generation/data/output.csv'
model_save_path = '/marketing_story_generation/config/saved_weights.ckpt'
table_name = 'CCB_marketing_story'


"""
file由用户选择的预测的文档
"""
def main():
    load_test_file(file, model_save_path, table_name)


if __name__ == "__main__":
    main()
