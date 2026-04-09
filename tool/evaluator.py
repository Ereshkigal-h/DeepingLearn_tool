import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import evaluate
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
def nlp_bleu_metric(references, predictions):
    bleu = evaluate.load("bleu")
    # BLEU 要求 references 是嵌套列表 [[ref1], [ref2]]
    formatted_refs = [[ref] for ref in references]
    return bleu.compute(predictions=predictions, references=formatted_refs)

def nlp_rouge_metric(references, predictions):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)

METRIC = {
    "ROC_AUC": [auc_metric, plot_roc, "classification"],
    "BLEU":    [nlp_bleu_metric, None, "generation"],
    "ROUGE":   [nlp_rouge_metric, None, "generation"],
}
class evaluator:
    def __init__(self,metric_list):
        self.metric_names = metric_list
        self.metric_list = []
        self.task_type = None

        for metric in metric_list:
            if metric not in METRIC:
                raise ValueError(f"不支持的指标: {metric}")
            metric_info = METRIC[metric]
            self.metric_list.append(metric_info)

            if self.task_type is None:
                self.task_type = metric_info[2]
            elif self.task_type != metric_info[2]:
                raise ValueError("不能在一个 evaluator 中混用分类和生成指标！")

    @torch.no_grad()
    def evaluate_classification(self,model,test_dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        for data,labels in tqdm.tqdm(test_dataloader):
            result=model(data)
            all_preds.append(result.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())
        all_similarities = np.concatenate(all_preds, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        result_dict={}
        for i,metric_info in enumerate(self.metric_list):
            compute_func, plot_func, _ = metric_info
            tmp_dict = compute_func(all_labels, all_preds)
            result_dict[self.metric_names[i]] = tmp_dict
            # 如果有画图函数则调用
            if plot_func:
                plot_func(**tmp_dict)
        return result_dict











