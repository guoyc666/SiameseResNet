import os, sys
import torch
from utils import load_support, get_image

def main(to_pred_dir, result_save_path, model):
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py) # 当前文件夹路径

    dirpath = os.path.abspath(to_pred_dir)
    filepath = os.path.join(dirpath, 'testB') # 测试集A文件夹路径
    task_lst = os.listdir(filepath) 

    res = ['img_name,label']  # 初始化结果文件，定义表头
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # weight_path = os.path.join(model_dir, 'resnet50_0_0.27.pth')
    # model = SiameseResNet(weight_path)
    # model = SiameseResNet()
    model.to(device)
    model.eval()
    with torch.no_grad():
        for task_name in task_lst:  # 循环task文件夹
            support_path = os.path.join(filepath, task_name, 'support')  # 支持集路径（文件夹名即为标签）
            query_path = os.path.join(filepath, task_name, 'query')  # 查询集路径（无标签，待预测图片）
            # 计算支持特征
            support_features = []
            for image, label in load_support(support_path):
                image = image.unsqueeze(0).to(device)
                feature = model.forward_one(image)
                support_features.append((feature, label)) 

            # 预测
            test_img_lst = [name for name in os.listdir(query_path) if name.endswith('.png')]
            for pathi in test_img_lst:
                if not pathi.endswith('.png'):
                    continue
                name_img = os.path.join(query_path, pathi)
                image = get_image(name_img).to(device)

                # pred_class = model.predict_prototypical(image, support_features)
                pred_class = model.predict(image, support_features)
                res.append(pathi + ',' + pred_class)

    # 将预测结果保存到result_save_path
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(res))

if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
