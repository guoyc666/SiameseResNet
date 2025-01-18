import os, sys

from numpy import average
import myrun as run
from make_test_data import generate_test_data
from modelS import SiameseResNet
from tqdm import tqdm

def score_task(result_file, gt_file):
    with open(result_file, 'r') as f:
        res = f.readlines()[1:]
    with open(gt_file, 'r') as f:
        gt = f.readlines()[1:]
    res = [i.strip().split(',') for i in res]
    gt = [i.strip().split(',') for i in gt]
    if len(res) != len(gt):
        return 0
    score = 0
    res.sort(key=lambda x: x[0])
    gt.sort(key=lambda x: x[0])
    for i in range(1, len(res)):
        if res[i] == gt[i]:
            score += 1
    return score / len(res)

def eval(model, generate=False):
    if generate:  
        generate_test_data()

    to_pred_dir = '../data'
    result_file = './out/result.csv'
    gt_file = './out/labels.csv'
    run.main(to_pred_dir, result_file, model)
    score = score_task(result_file, gt_file)
    # os.remove(result_file)
    return score
    
if __name__ == '__main__':
    model_list = []
    scores = []
    model_dir = './archive'
    model_names = []
    for model_name in os.listdir(model_dir):
        if not model_name.endswith('.pth'):
            continue
        model_path = os.path.join(model_dir, model_name)
        model = SiameseResNet(model_path)
        model_list.append(model)
        model_names.append(model_name.rstrip('.pth'))
        scores.append([])

    for j in range(1):
        generate_test_data(70)
        for i, m in tqdm(enumerate(model_list)):
            score = eval(m)
            scores[i].append(score)

    for i, s in enumerate(scores):
        print(f"[{model_names[i]}] Average score: {average(s)} {s}")


