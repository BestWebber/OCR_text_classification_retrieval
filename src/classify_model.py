import os
os.environ["TORCH_HOME"] = "./torch"
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, \
    average_precision_score
import numpy as np

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

# specify GPU
device = torch.device("cuda")

df = pd.read_csv("/marketing_story_generation/data/input.csv", encoding_errors='ignore')
df.head()

print("Number of sentences =", df.shape[0])
print("Label 0 Num :", (df["Review"] == 0).sum())  # netural
print("Label 1 Num :", (df["Review"] == 1).sum())  # positive
print("Label 2 Num :", (df["Review"] == 2).sum())  # negative
# check class distribution
df['Review'].value_counts(normalize=True)

train_text, temp_text, train_labels, temp_labels = train_test_split(df['Text'], df['Review'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=df['Review'])

# we will use temp_text and temp_labels to create validation and test set
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext', return_dict=False)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext', return_dict=False)

max_seq_len = 190

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

# define a batch size
batch_size = 32
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = True


class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.5)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 3)  # HAM vs SPAM (3 LABELS)
        self.bn2 = nn.BatchNorm1d(3, momentum=0.5)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.bn1(self.fc1(cls_hs))
        # x = self.fc1(cls_hs)
        # x dim 512
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.bn2(self.fc2(x))
        # x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# compute the class weights
class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

# convert class weights to tensor
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)

# loss function
cross_entropy = nn.NLLLoss(weight=weights)

# number of training epochs
epochs = 20

batch = next(iter(train_dataloader))
batch = [r.to(device) for r in batch]
sent_id, mask, labels = batch

out = model.bert(sent_id, mask)


# function to train the model
def train():
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()  # GRADIENT

        # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate():
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()  # DROP OUT

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def model_train():
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # 设定保存模型的路径
    model_save_path = '/marketing_story_generation/config/saved_weights.ckpt'  # 请替换为你指定的路径

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train()

        # evaluate model
        valid_loss, _ = evaluate()

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"SAVING MODEL to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')


def test():
    # get predictions for train data
    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    # 假设 y_true 是真实标签，preds 是预测概率
    n_classes = 3  # 你的分类数
    y_true_bin = label_binarize(test_y, classes=[0, 1, 2])  # 将标签转换为一对多格式
    fpr = {}
    tpr = {}
    roc_auc = {}

    # 计算每个类别的 FPR 和 TPR
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制所有类别的 ROC 曲线
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class {} (area = {:.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
    # 假设 y_true 是真实标签，preds 是预测概率
    y_pred = np.argmax(preds, axis=1)  # 获取预测的类别标签
    y_true_bin = label_binarize(test_y, classes=[0, 1, 2])  # 将标签转换为一对多格式

    # 准确率
    accuracy = accuracy_score(test_y, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # 精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(test_y, y_pred, average='weighted')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # 混淆矩阵
    conf_matrix = confusion_matrix(test_y, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)

    # 分类报告
    class_report = classification_report(test_y, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
    print('Classification Report:')
    print(class_report)

    # 平均精确率
    average_precision = average_precision_score(y_true_bin, preds, average='weighted')
    print(f'Average Precision: {average_precision:.2f}')


def main():
    model_train()
    print("模型训练完成！")
    test()
    print("")


if __name__ == "__main__":
    main()
