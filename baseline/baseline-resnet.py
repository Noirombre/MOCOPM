import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import re
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.models import resnet18
import torchvision
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.optim as optim

parser = argparse.ArgumentParser()

parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1,help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=0, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=5, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default='./split', help='manually specify the set of splits to use, instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--data_dir', default='./data', help = 'the path to the dataset' )
parser.add_argument('--label_path', default='./label.csv', help = 'Path to the label file')
parser.add_argument('--batchsize',type = int, default = 64)

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)


def load_labels(slide_data):

    slide_ids = slide_data.iloc[:, 1].astype(str).values  
    labels = slide_data.iloc[:, 2].values                 
    slide_label_dict = dict(zip(slide_ids, labels))
    return slide_label_dict



class BagDataset(Dataset):
    def __init__(self, data_dir, selected_data, slide_ids, slide_label_dict):

        self.data_dir = data_dir  
        self.selected_data = selected_data 
        self.slide_ids = slide_ids  
        self.labels = slide_label_dict

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[slide_id]
        folder_path = os.path.join(self.data_dir, slide_id)
        image_names = os.listdir(folder_path)
        images = []

        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                
                img = np.array(img)
                img = torch.tensor(img)
                img = img.permute(2, 0, 1)
                img = img.float()
                images.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

        if len(images) > 0:
            images = torch.stack(images)  
        else:
            images = torch.empty(0, 3, 224, 224)

        return images, label, slide_id


def create_datasets(data_dir, split_path, label_path, args):

    slide_data = pd.read_csv(label_path)
    label_dict = {'yuanfa': 0, 'zhuanyi': 1}

    for i in slide_data.index:
        key = slide_data.loc[i, 'label']
        slide_data.at[i, 'label'] = label_dict[key]

    ids_in_data = os.listdir(data_dir)
    selected_data = slide_data[slide_data['slide_id'].isin(ids_in_data)]

    slide_label_dict = load_labels(selected_data)

    split_df = pd.read_csv(split_path, dtype=slide_data['slide_id'].dtype)

    train_slide_ids = split_df['train'][split_df['train'].isin(selected_data['slide_id'])].dropna().astype(str).tolist()
    val_slide_ids = split_df['val'][split_df['val'].isin(selected_data['slide_id'])].dropna().astype(str).tolist()
    test_slide_ids = split_df['test'][split_df['test'].isin(selected_data['slide_id'])].dropna().astype(str).tolist()

    train_dataset = BagDataset(data_dir, selected_data, train_slide_ids, slide_label_dict)
    val_dataset = BagDataset(data_dir, selected_data, val_slide_ids, slide_label_dict)
    test_dataset = BagDataset(data_dir, selected_data, test_slide_ids, slide_label_dict)

    return train_dataset, val_dataset, test_dataset


def collate_bag_batch(batch):

    images_batch = [item[0] for item in batch]
    labels_batch = torch.tensor([item[1] for item in batch], dtype=torch.long)
    slide_ids_batch = [item[2] for item in batch]
    return images_batch, labels_batch, slide_ids_batch


class MILModel(nn.Module):
    def __init__(self, resnet_model, num_classes = 2):
        super(MILModel, self).__init__()
        self.resnet_model = nn.DataParallel(resnet_model.to(device), output_device = 'cuda:0')
        self.classifier = nn.Linear(512, num_classes).to(device)
    
    def forward(self, images_batch):
        bag_embeddings = []
        for images in images_batch:
            if images.size(0) == 0:
                embedding = torch.zeros(512).to(device)
            else:
                images = images.to(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                features = self.resnet_model(images) 
                features = features.view(features.size(0), -1)  
                embedding, _ = torch.max(features, dim=0)
            bag_embeddings.append(embedding)
        bag_embeddings = torch.stack(bag_embeddings)  
        outputs = self.classifier(bag_embeddings)    

        return outputs

def resnet_model():
    resnet_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    for param in resnet_model.parameters():
        param.requires_grad = False

    for param in resnet_model.fc.parameters():
        param.requires_grad = True


    resnet_model.fc = nn.Identity() 

    return resnet_model

def train(train_dataset, cur, args):
    print(f'the length of train dataset:{len(train_dataset)}')
    print(f'batch size:{args.batchsize}')
    print(f'fold:{args.k_start}')
        
    num_epochs = args.max_epochs
    
    torch.cuda.empty_cache()
    
    model_res = resnet_model()  
    
    model = MILModel(model_res)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()


    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, 
                            num_workers=32, collate_fn=collate_bag_batch)
    
    print('Fold{cur} start train')

    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        train_len = 0
        torch.cuda.empty_cache()
        
        for image, label, slide_id in train_loader:
            
            label = label.to(device)
            optimizer.zero_grad()

            torch.cuda.synchronize()  
            torch.backends.cudnn.enabled = False  

            outputs = model(image)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_len += 1

        train_loss /= train_len

        print(f'Fold:{cur}, epoch:{epoch}, the length of train is:{length_train} ,the loss of train is:{train_loss}')
    
    print(f'Fold{cur} train finished')
    torch.save(model.state_dict(), os.path.join(args.results_dir, "fold_{}_checkpoint.pt".format(cur)))

    return model


def val(val_dataset, model, cur, args):

    test_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=32,
                            collate_fn=collate_bag_batch)
    
    criterion = nn.CrossEntropyLoss()

    print(f'Fold{cur} start val')

    model.eval()
    
    val_loss = 0.0
    val_len = 0
    
    with torch.no_grad():
        torch.cuda.empty_cache()

        for image, label, slide_id in val_loader:

            label = label.to(device)

            torch.cuda.synchronize()
            torch.backends.cudnn.enabled = False
                
            outputs = model(image)
            loss = criterion(output, label)

            val_loss += loss.item()
            val_len += 1
        
        val_loss /= val_len


    print(f'Fold{cur} : the length of val is:{val_len}, the loss of val is:{val_loss}')




def test(test_dataset, model, cur, args):

    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=32,
                            collate_fn=collate_bag_batch)
    
    print(f'Fold{cur} start test')

    model.eval()
    

    all_predictions = []
    all_slide_ids = []
    all_labels = []
    all_pros = []
    test_len = 0
    
    with torch.no_grad():
        torch.cuda.empty_cache()

        for image, label, slide_id in test_loader:

            label = label.to(device)

            torch.cuda.synchronize()
            torch.backends.cudnn.enabled = False
                
            outputs = model(image)

            probabilities = nn.Softmax(dim=1)(outputs)
            predicted = torch.argmax(probabilities, dim=1)
            all_predictions.append(predicted.cpu().numpy())
            all_slide_ids.append(slide_id)
            all_labels.append(label.cpu().numpy())
            all_pros.append(probabilities.cpu().numpy())

            test_len += 1

    print(f'Fold{cur} : the length of test is:{test_len}')

    all_predictions = np.concatenate(all_predictions)  
    all_labels = np.concatenate(all_labels)  
    all_slide_ids = np.concatenate(all_slide_ids)  
    all_pros = np.concatenate(all_pros)

    result_df = pd.DataFrame({'slide_id': all_slide_ids, 'true_label':all_labels, 'pre_label':all_predictions,
                            'p_0':all_pros[:, 0], 'p_1':all_pros[:, 1]})

    test_acc = accuracy_score(all_labels, all_predictions)
    test_f1 = f1_score(all_labels, all_predictions, average='binary')
    test_precision = precision_score(all_labels, all_predictions, average='binary')
    test_recall = recall_score(all_labels, all_predictions, average='binary')    
    test_auc = roc_auc_score(all_labels, all_pros[:, 1])

    return test_acc, test_f1, test_precision, test_recall, test_auc, result_df



def main(args):

    seed_torch(args.seed)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end


    all_test_f1 = []
    all_test_acc = []
    all_test_precision = []
    all_test_recall = []
    all_test_auc = []

    folds = np.arange(start, end)

    for i in folds:
        train_dataset, val_dataset, test_dataset = create_datasets(data_dir = args.data_dir,
                                                                split_path = '{}/splits_{}.csv'.format(args.split_dir, i),
                                                                label_path = args.label_path, args = args) 


        model = train(train_dataset, i, args) 
        val(val_dataset, model, i, args)
        test_acc, test_f1, test_precision, test_recall, test_auc, result_df = test(test_dataset, model, i, args)
        result_name = 'result_partial_{}.csv'.format(i)
        result_df.to_csv(os.path.join(args.results_dir, result_name),index = False)
        final_df = pd.DataFrame({'folds': [i], 'test_acc': [test_acc], 'test_f1': [test_f1], 
                                'test_precision': [test_precision], 'test_recall': [test_recall], 'test_auc':[test_auc]})
        final_name = 'summary_partial_{}.csv'.format(i)
        final_df.to_csv(os.path.join(args.results_dir, final_name),index = False)
        all_test_acc.append(test_acc)
        all_test_f1.append(test_f1) 
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_auc.append(test_auc)

    all_result = pd.DataFrame({'folds': folds, 'test_acc': all_test_acc, 'test_f1': all_test_f1,
                            'test_precision':all_test_precision,  'test_recall': all_test_recall,
                            'test_auc':all_test_auc})
    mean_row = pd.DataFrame({'folds': ['mean'], 'test_acc': [np.mean(all_test_acc)], 
                    'test_f1': [np.mean(all_test_f1)],
                    'test_precision':[np.mean(all_test_precision)],
                    'test_recall': [np.mean(all_test_recall)],
                    'test_auc':[np.mean(all_test_auc)]})
    all_result = pd.concat([all_result, mean_row], ignore_index=True)
    all_result.to_csv(os.path.join(args.results_dir, 'all_result.csv'))



if __name__ == "__main__":
    results = main(args)
