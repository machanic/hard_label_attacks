import math
import os
import random
import sys

from models.standard_model import StandardModel

sys.path.append(os.getcwd())
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ("dashdot","dashdot"),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


def read_json_data(json_path):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    print("begin read {}".format(json_path))
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

surrogate_arch_name_to_paper = {"inceptionresnetv2":"IncResV2", "xception":"Xception", "resnet50":"ResNet50","convit_base":"ConViT",
                                "jx_vit":"ViT", "resnet-110":"ResNet110"}

def read_all_data(dataset_path_dict, arch, query_budgets, success_distortion_threshold):
    # dataset_path_dict {("CIFAR-10","l2","untargeted"): "/.../"， }
    data_info = {}
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            if file_path.startswith(arch) and file_path.endswith(".json"):
                file_path = dir_path + "/" + file_path
                distortion_dict, surrogate_archs = read_json_data(file_path)
                if "resnet50" in surrogate_archs and "jx_vit" in surrogate_archs:
                    continue
                x = []
                y = []
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
                    success_list = distortion_list <= success_distortion_threshold
                    success_list = success_list.astype(np.float32)
                    success_rate = np.mean(success_list) * 100.0
                    x.append(query_budget)
                    y.append(success_rate)

                x = np.array(x)
                y = np.array(y)
                surrogate_archs_new = [surrogate_arch_name_to_paper[surrogate_arch] for surrogate_arch in surrogate_archs]
                data_info[(dataset, norm, targeted, method, "&".join(surrogate_archs_new))] = (x, y)
    return data_info


method_name_to_paper = {"tangent_attack":"TA",
                        "ellipsoid_tangent_attack":"G-TA", "GeoDA":"GeoDA",
                        "HSJA":"HSJA",  "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                         "Evolutionary":"Evolutionary", "SurFree":"SurFree",
                        "TriangleAttack":"Triangle Attack", "PriorSignOPT":"Prior-Sign-OPT",
                        "PriorOPT":"Prior-OPT", "RayS":"RayS"
                        #"QEBA":"QEBA", "CGBA_H":"CGBA-H"
                        }

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


def get_all_exists_folder(dataset, methods, norm, targeted):
    root_dir = "F:/logs/hard_label_attack_complete/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    for method in methods:
        file_name = from_method_to_dir_path(dataset, method, norm, targeted)
        file_path = root_dir + file_name
        if os.path.exists(file_path):
            dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method])] = file_path
        else:
            print("{} does not exist!!!".format(file_path))
    return dataset_path_dict



method_linestyle_mark_dict = {}

def draw_query_success_rate_figure(dataset, norm, targeted, arch, success_distortion_threshold, dump_file_path, xlabel, ylabel):

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
    if dataset == "CIFAR-10":
        # if "TriangleAttack" in methods:
        #     methods.remove("TriangleAttack")
        if "HSJA" in methods:
            methods.remove("HSJA")
        if "tangent_attack" in methods:
            methods.remove("tangent_attack")
        if "ellipsoid_tangent_attack" in methods:
            methods.remove("ellipsoid_tangent_attack")

    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted)
    max_query = 10000
    if dataset=="ImageNet" and targeted:
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    data_info = read_all_data(dataset_path_dict, arch, query_budgets, success_distortion_threshold)  # fig_type can be mean_distortion or median_distortion
    plt.style.use('bmh')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'orange', "pink", "brown", "slategrey", "cornflowerblue",
              "greenyellow", "darkgoldenrod", "r", "slategrey", "navy", "darkseagreen", "xkcd:blueberry", "grey",
              "indigo", "olivedrab"]
    markers = ['o', '>', '*', 's', "P", "p", "X", "h", "D", "H", "^", "<", "d", ".", "+", "x", "v", "1", "2", "3", "4"]
    linestyles = ["solid", "dashed", "densely dotted", "dashdotdotted", "densely dashed", "densely dashdotdotted"]

    xtick = np.array([0,1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    if max_query == 20000:
        xtick = np.array([0,1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])

    for idx, ((dataset, norm, targeted, method, surrogate_archs), (x,y)) in enumerate(data_info.items()):
        x = np.asarray(x)
        y = np.asarray(y)
        if surrogate_archs:
            method = method + "$_{\mathrm{"+ surrogate_archs + "}}$"
        if method not in method_linestyle_mark_dict:
            linestyle = linestyle_dict[linestyles[idx%len(linestyles)]]
            mark = markers[idx]
            color = colors[idx]

            for_loop_count = 0
            all_use_color = False
            all_use_mark = False
            while color in [tuple_value[2] for tuple_value in
                            method_linestyle_mark_dict.values()]:
                color = colors[random.randint(0, len(colors) - 1)]
                for_loop_count += 1
                if for_loop_count > 1000:
                    all_use_color = True
                    break
            for_loop_count = 0
            while mark in [tuple_value[1] for tuple_value in
                           method_linestyle_mark_dict.values()]:
                mark = markers[random.randint(0, len(colors) - 1)]
                for_loop_count += 1
                if for_loop_count > 1000:
                    all_use_mark = True
                    break
            if all_use_color or all_use_mark:
                while (mark, color) in [(tuple_value[1], tuple_value[2]) for tuple_value in
                                        method_linestyle_mark_dict.values()]:
                    color = colors[random.randint(0, len(colors) - 1)]
                    mark = markers[random.randint(0, len(markers) - 1)]

            method_linestyle_mark_dict[method] = (linestyle, mark, color)
        plt.plot(x, y, label=method, color=method_linestyle_mark_dict[method][2],
                         linestyle=method_linestyle_mark_dict[method][0], linewidth=1.5,
                         marker=method_linestyle_mark_dict[method][1], markersize=6)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    if max_query > 10000:
        plt.xlim(0, max_query+1000)
    else:
        plt.xlim(0, max_query)
    plt.ylim(0, 101)
    plt.gcf().subplots_adjust(bottom=0.15)
    # xtick = [0, 5000, 10000]
    if dataset == "ImageNet" and targeted:
        x_ticks = xtick[0::2]
        x_ticks = x_ticks.tolist()
        x_ticks_label = ["0"]  + ["{}K".format(x_tick // 1000) for x_tick in x_ticks[1:]]
        plt.xticks(x_ticks, x_ticks_label, fontsize=20)
    else:
        x_ticks_label =["0"]  + ["{}K".format(x_tick // 1000) for x_tick in xtick[1:]]
        plt.xticks(xtick, x_ticks_label, fontsize=20)
    yticks = np.arange(0, 101, 10)
    plt.yticks(yticks, fontsize=20)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(loc='lower right' if not  targeted else 'upper left', prop={'size': 15},handlelength=4,framealpha=0.5,fancybox=True,frameon=True)
    plt.savefig(dump_file_path, dpi=200)
    plt.close()
    print("save to {}".format(dump_file_path))



def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Attacking Normal Models')
    parser.add_argument("--dataset", type=str,  help="the dataset to train")
    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "D:/黑盒攻击论文/hard-label attacks/Prior-OPT/icml2024/figures/query_vs_success_rate/"
    os.makedirs(dump_folder, exist_ok=True)

    for dataset in [ "CIFAR-10"]:
        args.dataset = dataset
        if "CIFAR" in args.dataset:
            archs = ["pyramidnet272","WRN-28-10-drop","WRN-40-10-drop","densenet-bc-L190-k40","gdas"]
        else:
            archs = ["senet154", "resnet101", "inceptionv3", "resnext101_64x4d", "inceptionv4",
                     "jx_vit", "gcvit_base", "swin_base_patch4_window7_224"]
        targeted_list = [False,True]
        for targeted in targeted_list:
            args.targeted = targeted
            for arch in archs:
                file_path  = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack.pdf".format(dataset=args.dataset,
                              model=arch, norm=args.norm, target_str="untargeted" if not args.targeted else "targeted")
                x_label = "Number of Queries"
                y_label = "Attack Success Rate"
                model = StandardModel(dataset=args.dataset, arch=arch, no_grad=True, load_pretrained=True)
                dim = np.prod(np.array([each_dim for each_dim in model.input_size])).item()
                if dataset == "CIFAR-10":
                    success_distortion_threshold = 1.0
                else:
                    success_distortion_threshold = math.sqrt(0.001 * dim)
                print("arch:{}, dim:{}, Success distortion threshold:{}".format(arch, dim, success_distortion_threshold))
                draw_query_success_rate_figure(args.dataset, args.norm, args.targeted, arch, success_distortion_threshold, file_path, x_label, y_label)
