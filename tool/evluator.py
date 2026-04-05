import numpy as np
import torch
import tqdm
from imblearn import metrics
from sklearn.metrics import roc_curve, auc


def auc_metric(y_true,y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc=auc(fpr,tpr)
    index=np.argmax(tpr-fpr)
    min_fpr,max_tpr,best_th=fpr[index],tpr[index],thresholds[index]
    predictions = (y_pred >= best_th).astype(int)
    acc = np.mean(predictions == y_true)
    return {
        "auc": roc_auc,
        "min_fpr": min_fpr,
        "max_tpr": max_tpr,
        "best_threshold": best_th,
        "acc": acc,
        "fpr": fpr,
        "tpr": tpr
    }

METRIC={
    "ROC_AUC":auc_metric,
}
class evluator:
    def __init__(self,metric_list):
        self.metric_list = []
        self.metric_names = metric_list
        for metric in metric_list:
            self.metric_list.append(METRIC[metric])

    @torch.no_grad()
    def evaluate(self,model,test_dataloader,device):
        model.eval()
        all_similarities = []
        all_labels = []
        for data,labels in tqdm.tqdm(test_dataloader):
            #这里是最后结果的计算方式
            result=model(data)
            all_similarities.append(result.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())
        all_similarities = np.concatenate(all_similarities, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        result_dict={}
        for i,metric in enumerate(self.metric_list):
            tmp_list=metric(all_labels,all_similarities)
            result_dict[self.metric_names[i]]=tmp_list
        return result_dict









