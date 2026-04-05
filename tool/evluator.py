import numpy as np
import torch
import tqdm
from imblearn import metrics
from matplotlib import pyplot as plt
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
def plot_roc(fpr, tpr, auc, save_path="roc_result.png",**kwargs):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')  # 对角线虚线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"ROC曲线已保存为: {save_path}")
METRIC={
    "ROC_AUC":[auc_metric,plot_roc],
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
            tmp_dict=metric[0](all_labels,all_similarities)
            result_dict[self.metric_names[i]]=tmp_dict
            metric[1](**tmp_dict)
        return result_dict











