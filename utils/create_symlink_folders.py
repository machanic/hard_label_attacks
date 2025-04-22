import os
from collections import defaultdict

source_root_folder_1 = "F:/logs/hard_label_attacks"
source_root_folder_2 = "F:/logs/cluster_logs"
target_root_folder = "F:/logs/hard_label_attack_complete"

all_source_folders = defaultdict(int)
for dir_name in os.listdir(source_root_folder_1):
    all_source_folders[dir_name] += 1
for dir_name in os.listdir(source_root_folder_2):
    all_source_folders[dir_name] += 1
os.makedirs(target_root_folder, exist_ok=True)
for dir_name, count in all_source_folders.items():
    if count > 1:
        source_sub_files_1 =[dir_name+"/"+file for file in os.listdir(source_root_folder_1+"/"+dir_name)]
        source_sub_files_2 = [dir_name + "/" + file for file in
                              os.listdir(source_root_folder_2 + "/" + dir_name)]
        a = set(source_sub_files_1) & set(source_sub_files_2)
        rest_files = set(source_sub_files_1).union(set(source_sub_files_2)) - a
        if len(a) > 0:
            for sub_file in a:
                source_file_path = source_root_folder_1 + "/" + sub_file
                target_file_path = target_root_folder + "/" + sub_file
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                if not os.path.exists(target_file_path):
                    os.symlink(source_file_path, target_file_path, False)
            for rest_file in rest_files:
                target_file_path = target_root_folder + "/" + rest_file
                if os.path.exists(source_root_folder_1 + "/" + rest_file) and not os.path.exists(target_file_path):
                    os.symlink(source_root_folder_1 + "/" + rest_file, target_file_path, False)
                if os.path.exists(source_root_folder_2 + "/" + rest_file) and not os.path.exists(target_file_path):
                    os.symlink(source_root_folder_2 + "/" + rest_file, target_file_path, False)
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            for rest_file in rest_files:
                target_file_path = target_root_folder + "/" + rest_file
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                if os.path.exists(source_root_folder_1 + "/" + rest_file) and not os.path.exists(target_file_path):
                    os.symlink(source_root_folder_1 + "/" + rest_file, target_file_path, False)
                if os.path.exists(source_root_folder_2 + "/" + rest_file) and not os.path.exists(target_file_path):
                    os.symlink(source_root_folder_2 + "/" + rest_file, target_file_path, False)
    else:
        if os.path.exists(source_root_folder_1 + "/" + dir_name) and not os.path.exists(target_root_folder + "/" + dir_name):
            os.symlink(source_root_folder_1 + "/" + dir_name, target_root_folder + "/" + dir_name, target_is_directory=True)
        if os.path.exists(source_root_folder_2 + "/" + dir_name) and not os.path.exists(target_root_folder + "/" + dir_name):
            os.symlink(source_root_folder_2 + "/" + dir_name, target_root_folder + "/" + dir_name,
                       target_is_directory=True)
