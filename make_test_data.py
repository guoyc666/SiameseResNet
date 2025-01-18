from itertools import cycle
import os
import shutil
import random

def create_folders(dst_folder, n, class_list):
    task_folder = os.path.join(dst_folder, 'task'+str(n))
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    query_folder = os.path.join(task_folder, 'query')
    if not os.path.exists(query_folder):
        os.makedirs(query_folder)
    support_folder = os.path.join(task_folder, 'support')
    if not os.path.exists(support_folder):
        os.makedirs(support_folder)

    for path in class_list:
        image_files = [f for f in os.listdir(path)]
        random.shuffle(image_files)
        # 查询集
        for j in range(query_num):
            src_image_path = os.path.join(path, image_files[j])
            dst_image_path = os.path.join(query_folder, image_files[j])
            shutil.copy2(src_image_path, dst_image_path)
        # 支持集
        class_folder = os.path.join(support_folder, os.path.basename(path).lstrip('class_'))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        for j in range(query_num, query_num+support_num):
            src_image_path = os.path.join(path, image_files[j])
            dst_image_path = os.path.join(class_folder, f'{j-query_num}.png')
            shutil.copy2(src_image_path, dst_image_path)

        # 垃圾主办方
        empty_file_path = os.path.join(class_folder, '.DS_Store')
        with open(empty_file_path, 'w') as f:
            pass

    empty_file_path = os.path.join(query_folder, '.DS_Store')
    with open(empty_file_path, 'w') as f:
        pass
    empty_file_path = os.path.join(support_folder, '.DS_Store')
    with open(empty_file_path, 'w') as f:
        pass

def split_test_images(src_folder, dst_folder, n):
    class_files = [os.path.join(src_folder, path) for path in os.listdir(src_folder)]

    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)

    class_groups = split_group(class_files, classnum_per_task)

    for i in range(n):
        try:
            class_list = next(class_groups)
        except StopIteration:
            class_groups = split_group(class_files, classnum_per_task)
            class_list = next(class_groups)

        create_folders(dst_folder, i, class_list)

def split_group(class_files, m):
    random.shuffle(class_files)
    class_groups = [class_files[i:i+m] for i in range(0, len(class_files), m)]
    groups_iter = cycle(class_groups)
    return groups_iter

def labels_to_csv(test_folder):
    res = ['img_name,label']
    for task_name in os.listdir(test_folder):
        query_path = os.path.join(test_folder, task_name, 'query')
        test_img_lst = [name for name in os.listdir(query_path) if name.endswith('.png')]
        for idx, pathi in enumerate(test_img_lst):
            imgname = f'{task_name}_{idx}.png'
            os.rename(os.path.join(query_path, pathi), os.path.join(query_path, imgname))
            res.append(imgname + ',' + pathi.split('_')[1])

    with open('./out/labels.csv', 'w') as f:
        f.write('\n'.join(res))

classnum_per_task = 10
support_num = 5
query_num = 2
src_folder = '../data/test'
dst_folder = '../data/testB'
label_path = './out/labels.csv'

def generate_test_data(n=5):

    split_test_images(src_folder, dst_folder, n)
    labels_to_csv(dst_folder)
    # print(f'{n} tasks prepared')

if __name__ == '__main__':
    generate_test_data()
