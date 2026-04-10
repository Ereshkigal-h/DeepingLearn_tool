import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
import evaluate


# ================= 二分类指标 =================
def auc_metric(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    index = np.argmax(tpr - fpr)
    min_fpr, max_tpr, best_th = fpr[index], tpr[index], thresholds[index]
    predictions = (y_pred >= best_th).astype(int)
    acc = np.mean(predictions == y_true)
    return {
        "auc": roc_auc, "min_fpr": min_fpr, "max_tpr": max_tpr,
        "best_threshold": best_th, "acc": acc, "fpr": fpr, "tpr": tpr
    }


def plot_roc(fpr, tpr, auc, save_path="roc_result.png", **kwargs):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
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


# ================= 多分类指标 =================
def multi_class_metric(y_true, y_pred):
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        predictions = np.argmax(y_pred, axis=1)
    else:
        predictions = np.round(y_pred)
    acc = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(y_true, predictions, average='macro')
    return {"acc": acc, "macro_f1": macro_f1}


# ================= NLP 文本生成指标 =================
def nlp_bleu_metric(references, predictions):
    bleu = evaluate.load("bleu")
    formatted_refs = [[ref] for ref in references]
    return bleu.compute(predictions=predictions, references=formatted_refs)
def nlp_rouge_metric(references, predictions):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


# ================= 指标注册表 =================
METRIC = {
    # 格式: "指标名": [计算函数, 画图函数, "任务类型"]
    "ROC_AUC": [auc_metric, plot_roc, "classification"],
    "MULTI_ACC": [multi_class_metric, None, "classification"],
    "BLEU": [nlp_bleu_metric, None, "generation"],
    "ROUGE": [nlp_rouge_metric, None, "generation"],
}


class evaluator:
    def __init__(self, metric_list):
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
    def evaluate(self, model, test_dataloader, tokenizer=None, device="cuda"):
        model.eval()
        if self.task_type == "classification":
            return self._evaluate_classification(model, test_dataloader, device)
        elif self.task_type == "generation":
            if tokenizer is None:
                raise ValueError("生成任务必须传入 tokenizer 用于解码文本！")
            return self._evaluate_generation(model, test_dataloader, tokenizer, device)

    def _evaluate_classification(self, model, test_dataloader, device):
        all_preds = []
        all_labels = []
        for data, labels in tqdm.tqdm(test_dataloader, desc="分类评估中"):
            data = data.to(device)
            result = model(data)
            all_preds.append(result.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())


        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0).flatten()


        if all_preds.ndim == 2 and all_preds.shape[1] == 1:
            all_preds = all_preds.flatten()

        result_dict = {}
        for i, metric_info in enumerate(self.metric_list):
            compute_func, plot_func, _ = metric_info
            tmp_dict = compute_func(all_labels, all_preds)
            result_dict[self.metric_names[i]] = tmp_dict
            if plot_func:
                plot_func(**tmp_dict)
        return result_dict

    def _evaluate_generation(self, model, test_dataloader, tokenizer, device, max_length=50):
        all_preds = []
        all_labels = []
        for batch in tqdm.tqdm(test_dataloader, desc="生成评估中"):
            input_ids = batch["src"].to(device)
            target_ids = batch["tar_label"]
            src_mask = batch["src_mask"]
            generated_ids = model.generate(input_ids,src_mask=src_mask,max_length=max_length
                                           ,start_token_id=tokenizer.convert_tokens_to_ids("[CLS]")
                                           ,end_token_id=tokenizer.convert_tokens_to_ids("[SEP]"))
            preds_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels_text = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            all_preds.extend(preds_text)
            all_labels.extend(labels_text)

        result_dict = {}
        for i, metric_info in enumerate(self.metric_list):
            compute_func = metric_info[0]
            tmp_dict = compute_func(references=all_labels, predictions=all_preds)
            result_dict[self.metric_names[i]] = tmp_dict
        return result_dict