import math
from collections import defaultdict

import numpy as np
import json
import os
from sklearn.metrics import auc

from models.standard_model import StandardModel

method_name_to_paper = {"tangent_attack":"TA",
                        "ellipsoid_tangent_attack":"G-TA", "GeoDA":"GeoDA",
                        "HSJA":"HSJA",  "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                         "Evolutionary":"Evolutionary", "SurFree":"SurFree",
                        "TriangleAttack":"Triangle Attack", "PriorSignOPT":"Prior-Sign-OPT",
                        "PriorOPT":"Prior-OPT", "RayS":"RayS"
                        #"QEBA":"QEBA", "CGBA_H":"CGBA-H"
                        }
surrogate_arch_name_to_paper = {"inceptionresnetv2":"IncResV2", "xception":"Xception", "resnet50":"ResNet50","convit_base":"ConViT",
                                "jx_vit":"ViT", "resnet-110":"ResNet110", "senet154":"SENet154", "densenet-bc-100-12":"DenseNetBC100"}

def longest_common_subsequence(x: str, y: str):
    """
    Finds the longest common subsequence between two strings. Also returns the
    The subsequence found

    Parameters
    ----------

    x: str, one of the strings
    y: str, the other string

    Returns
    -------
    L[m][n]: int, the length of the longest subsequence. Also equal to len(seq)
    Seq: str, the subsequence found

    """
    # find the length of strings

    assert x is not None
    assert y is not None

    m = len(x)
    n = len(y)

    # declaring the array for storing the dp values
    l = [[0] * (n + 1) for _ in range(m + 1)]  # noqa: E741

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = 1 if x[i - 1] == y[j - 1] else 0

            l[i][j] = max(l[i - 1][j], l[i][j - 1], l[i - 1][j - 1] + match)

    seq = ""
    i, j = m, n
    while i > 0 and j > 0:
        match = 1 if x[i - 1] == y[j - 1] else 0

        if l[i][j] == l[i - 1][j - 1] + match:
            if match == 1:
                seq = x[i - 1] + seq
            i -= 1
            j -= 1
        elif l[i][j] == l[i - 1][j]:
            i -= 1
        else:
            j -= 1

    return l[m][n], seq

def from_method_to_denfensive_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack" or method == "ellipsoid_tangent_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA" or method == "RayS" or method == "SurFree" or method == "Evolutionary" or method == "TriangleAttack" or method == "PriorOPT" or method == "PriorSignOPT":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        if targeted:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        if targeted:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    return path


def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "ellipsoid_tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                               norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    # elif method == "RayS":
    #     path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "AHA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "Evolutionary":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "TriangleAttack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "CGBA_H":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SurFree":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "QEBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    return path

def read_json_data(json_path):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    # print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read()
        data_json = json.loads(data_txt)
        distortion_dict = data_json["distortion"]
        surrogate_archs = []
        if data_json["args"]["targeted"]:
            assert data_json["args"]["load_random_class_image"] is True
        if "surrogate_arch" in data_json["args"] and data_json["args"]["surrogate_arch"] is not None:
            surrogate_archs.append(data_json["args"]["surrogate_arch"])
        elif "surrogate_archs" in data_json["args"] and data_json["args"]["surrogate_archs"] is not None:
            surrogate_archs.extend(data_json["args"]["surrogate_archs"])
    return distortion_dict, surrogate_archs

def get_all_exists_folder(dataset, methods, norm, targeted, is_defense_model):
    root_dir = "F:/logs/hard_label_attack_complete/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    for method in methods:
        if is_defense_model:
            file_name = from_method_to_denfensive_dir_path(dataset, method, norm, targeted)
        else:
            file_name = from_method_to_dir_path(dataset, method, norm, targeted)
        file_path = root_dir + file_name
        if os.path.exists(file_path):
            dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method])] = file_path
        else:
            print("{} does not exist!!!".format(file_path))
    return dataset_path_dict

def read_query_distortion_data(dataset_path_dict, arch, query_budgets, success_distortion_threshold, is_defense):
    # dataset_path_dict {("CIFAR-10","l2","untargeted"): "/.../"， }
    data_info = {}
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            condition = longest_common_subsequence(arch, file_path)[1] == arch if is_defense else file_path.startswith(arch)
            if condition and file_path.endswith(".json"):
                file_path = dir_path + "/" + file_path
                print("read file_path {}".format(file_path))
                distortion_dict, surrogate_archs = read_json_data(file_path)
                if "resnet50" in surrogate_archs and "jx_vit" in surrogate_archs:
                    continue
                x = []
                y_distortions = []
                y_sucess_rates = []
                for query_budget in query_budgets:
                    distortion_list = []
                    for image_id, query_distortion_dict in distortion_dict.items():
                        query_distortion_dict = {int(float(query)): float(dist) for query, dist in query_distortion_dict.items()}
                        queries = np.array(list(query_distortion_dict.keys()))
                        queries = np.sort(queries)
                        find_index = np.searchsorted(queries, query_budget, side='right') - 1
                        if query_budget < queries[find_index]:
                            print(
                                "query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(
                                    query_budget, queries[find_index], np.min(queries).item(),
                                    len(query_distortion_dict)))
                            continue
                        distortion_list.append(query_distortion_dict[queries[find_index]])
                    distortion_list = np.array(distortion_list)
                    distortion_list = distortion_list[~np.isnan(distortion_list)]  # 去掉nan的值
                    mean_distortion = np.mean(distortion_list)
                    x.append(query_budget)
                    y_distortions.append(mean_distortion)

                    success_list = distortion_list <= success_distortion_threshold
                    success_list = success_list.astype(np.float32)
                    success_rate = np.mean(success_list) * 100.0
                    y_sucess_rates.append(success_rate)

                x = np.array(x)
                y_distortions = np.array(y_distortions)
                y_sucess_rates = np.array(y_sucess_rates)
                # print("final ASR:{}".format(y_sucess_rates[-1]))
                surrogate_archs_new = [surrogate_arch_name_to_paper[surrogate_arch] for surrogate_arch in surrogate_archs]
                data_info[(dataset, arch, norm, targeted, method, "\&".join(surrogate_archs_new))] = (x, y_distortions, y_sucess_rates)
    return data_info


def draw_wide_table(table_data):
    for method, arch_data_info_dict in table_data.items():
        arch_new_dict = {}
        for (dataset, arch, norm, targeted), (ASR, AUC, mean_l2) in arch_data_info_dict.items():
            arch_new_dict[arch] = (mean_l2,AUC,ASR)

        print("{0} & {1:.3f} & {2:.1f} & {3:.1f}\% & {4:.3f} & {5:.1f} & {6:.1f}\% & {7:.3f} & {8:.1f} & {9:.1f}\% \\\\".format(method,
                arch_new_dict["jx_vit"][0],arch_new_dict["jx_vit"][1],arch_new_dict["jx_vit"][2],
                arch_new_dict["gcvit_base"][0],arch_new_dict["gcvit_base"][1],arch_new_dict["gcvit_base"][2],
                arch_new_dict["swin_base_patch4_window7_224"][0],arch_new_dict["swin_base_patch4_window7_224"][1],arch_new_dict["swin_base_patch4_window7_224"][2]))
def draw_narrow_table(table_data):
    for method, arch_data_info_dict in table_data.items():
        arch_new_dict = {}
        for (dataset, arch, norm, targeted), (ASR, AUC, mean_l2) in arch_data_info_dict.items():
            arch_new_dict[arch] = (mean_l2,AUC,ASR)

        # print(" & \\footnotesize {0} & {1:.3f} & {2:.3f} & {3:.3f} \\\\".format(method,arch_new_dict["resnet-50_adv_train"][0], arch_new_dict["resnet-50_TRADES"][0],arch_new_dict["resnet-50_feature_scatter"][0]))
        print(" & \\footnotesize {0} & {1:.1f}\% & {2:.1f}\% \\\\".format(method,
                                                                                arch_new_dict["resnet-50_adv_train"][1],
                                                                                arch_new_dict["resnet-50_TRADES"][1]))

def collect_AUC_ASR_table(table_data, dataset, norm, targeted, arch, is_defense_model, success_distortion_threshold):

    # fig_type can be [query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query]
    methods = list(method_name_to_paper.keys())
    if targeted:
        if "TriangleAttack" in methods:
            methods.remove("TriangleAttack")
        if "RayS" in methods:
            methods.remove("RayS")
        if "GeoDA" in methods:
            methods.remove("GeoDA")
    if norm == "l2":
        if "RayS" in methods:
            methods.remove("RayS")
    # if dataset == "CIFAR-10":
    #     if "TriangleAttack" in methods:
    #         methods.remove("TriangleAttack")
    #     if "HSJA" in methods:
    #         methods.remove("HSJA")
    #     if "tangent_attack" in methods:
    #         methods.remove("tangent_attack")
    #     if "ellipsoid_tangent_attack" in methods:
    #         methods.remove("ellipsoid_tangent_attack")

    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted, is_defense_model)
    max_query = 10000
    if dataset=="ImageNet" and targeted:
        max_query = 20000
    if dataset == "ImageNet" and is_defense_model:
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    data_info = read_query_distortion_data(dataset_path_dict, arch, query_budgets, success_distortion_threshold, is_defense_model)  # fig_type can be mean_distortion or median_distortion
    for idx, ((dataset, arch, norm, targeted, method, surrogate_archs), (x,y_distortions, y_sucess_rates)) in enumerate(data_info.items()):
        AUC = round(auc(x, y_distortions),1)
        final_success_rate = round(y_sucess_rates[np.where(x==max_query)[0].item()],1)
        final_mean_l2_distortion = round(y_distortions[np.where(x==max_query)[0].item()], 3)
        if surrogate_archs:
            method = method + "\\textsubscript{\\tiny "+ surrogate_archs + "}"
        table_data[method][(dataset, arch, norm, targeted)] = (final_success_rate, AUC, final_mean_l2_distortion)
        # if arch == "gcvit_base" or arch == "swin_base_patch4_window7_224":
        #     if method == "Prior-Sign-OPT\\textsubscript{\\tiny ResNet50\&ConViT}":
        #         print(method, arch, final_success_rate)
        # print("{dataset}, {arch}, {norm} {targeted} {method}  mean_l2_distortion: {mean_l2}, ASR: {ASR}, AUC: {AUC}".format(dataset=dataset, arch=arch,
        #                             norm=norm, method=method,
        #                                 targeted="untargeted" if not targeted else "targeted",ASR=final_success_rate, AUC=AUC,
        #                                 mean_l2=final_mean_l2_distortion))



if __name__ == "__main__":
    is_defense_model = True
    for dataset in ["CIFAR-10"]:
        if "CIFAR" in dataset:
            archs = ["pyramidnet272","WRN-28-10-drop","WRN-40-10-drop","densenet-bc-L190-k40","gdas"]
        else:
            archs = [#"inceptionv4","senet154","resnet101","inceptionv3","resnext101_64x4d",
                     "jx_vit","gcvit_base","swin_base_patch4_window7_224"]
        if is_defense_model:
            archs = ['resnet-50_TRADES', "resnet-50_feature_scatter", "resnet-50_adv_train"]
        targeted_list = [False]
        norm = "l2"
        table_data = defaultdict(dict)
        for targeted in targeted_list:
            for arch in archs:
                if dataset == "CIFAR-10":
                    dim = 32 * 32 * 3
                else:
                    model = StandardModel(dataset=dataset, arch=arch, no_grad=True, load_pretrained=True)
                    dim = np.prod(np.array([each_dim for each_dim in model.input_size])).item()
                success_distortion_threshold = math.sqrt(0.001 * dim)
                print("arch:{}, dim:{}, Success distortion threshold:{}".format(arch, dim, success_distortion_threshold))
                collect_AUC_ASR_table(table_data, dataset, norm, targeted, arch, is_defense_model, success_distortion_threshold)
        if is_defense_model:
            draw_narrow_table(table_data)
        else:
            draw_wide_table(table_data)
    print("======================================================================================================")
    # is_defense_model = True
    # for dataset in ["ImageNet"]:
    #     if "CIFAR" in dataset:
    #         archs = ['resnet-50_TRADES', "resnet-50_jpeg", "resnet-50_feature_scatter",
    #                  "resnet-50_feature_distillation", "resnet-50_com_defend", "resnet-50_adv_train"]
    #     else:
    #         archs = ["resnet50_adv_train_on_ImageNet_linf_4_div_255", "resnet50_adv_train_on_ImageNet_l2_3",
    #                  "resnet50_adv_train_on_ImageNet_linf_8_div_255"]
    #     norm = "l2"
    #     for arch in archs:
    #         if dataset == "CIFAR-10":
    #             dim = 32 * 32 * 3
    #         elif dataset == "ImageNet":
    #             dim = 224 * 224 * 3
    #         if dataset == "ImageNet":
    #             success_distortion_threshold = math.sqrt(0.001 * dim)
    #         elif dataset == "CIFAR-10":
    #             success_distortion_threshold = 1.0
    #         print(
    #             "arch:{}, dim:{}, Success distortion threshold:{}".format(arch, dim, success_distortion_threshold))
    #         print_AUC_ASR_table(dataset, norm, False, arch, is_defense_model, success_distortion_threshold)

