import math
from collections import defaultdict

import bisect
import numpy as np
import json
import os

from sklearn.metrics import auc


def new_round(_float, _len):
    """
    Parameters
    ----------
    _float: float
    _len: int, 指定四舍五入需要保留的小数点后几位数为_len

    Returns
    -------
    type ==> float, 返回四舍五入后的值
    """
    if isinstance(_float, float):
        if str(_float)[::-1].find('.') <= _len:
            return (_float)
        if str(_float)[-1] == '5':
            return (round(float(str(_float)[:-1] + '6'), _len))
        else:
            return (round(_float, _len))
    else:
        return (round(_float, _len))


method_name_to_paper = {"PriorOPT": "Prior-OPT",
                        "PriorSignOPT": "Prior-Sign-OPT", "PurePriorSignOPT": "Pure-Prior-Sign-OPT",
                        "PurePriorOPT": "Pure-Prior-OPT",
                        "SignOPT": "Sign-OPT"}


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
    elif method == "BBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
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
    elif method == "SQBA_prior1":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SQBA_prior2":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "BBA_prior1":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "BBA_prior2":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SurFree":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT_prior1":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT_prior1":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT_PGD_init_theta":
        path = "PriorSignOPT-{dataset}-{norm}-{target_str}_with_PGD_init_theta".format( dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT_2priors":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT_2priors":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method.split("_")[0], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT_PGD_init_theta":
        path = "PriorOPT-{dataset}-{norm}-{target_str}_with_PGD_init_theta".format( dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PurePriorOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PurePriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "QEBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    return path


def read_json_and_extract(json_path):
    with open(json_path, "r") as file_obj:
        json_content = json.load(file_obj)
        distortion = json_content["distortion"]
        return distortion

def get_file_name_list(dataset, method_name_to_paper, norm, targeted):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        file_path = "G:/logs/hard_label_attacks/" + from_method_to_dir_path(dataset, method, norm, targeted)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def bin_search(arr, target):
    if target not in arr:
        return None
    arr.sort()
    return arr[arr.index(target)-1], arr.index(target)-1

def get_mean_distortion_ASR(distortion_dict, query_budgets, success_distortion_threshold, want_key):
    mean_and_median_distortions = defaultdict(lambda : "-")
    query_to_ASR = defaultdict(lambda : "-")
    x = []
    y_distortions = []
    for query_budget in query_budgets:
        distortion_list = []
        for idx, (image_index, query_distortion) in enumerate(distortion_dict.items()):
            if idx==0:
                assert int(image_index) == 0
            query_distortion = {int(float(query)):float(dist) for query,dist in query_distortion.items()}
            queries = np.array(list(query_distortion.keys()))
            queries = np.sort(queries)
            find_index = np.searchsorted(queries, query_budget, side='right') - 1
            if query_budget < queries[find_index]:
                print("query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(query_budget, queries[find_index], np.min(queries).item(), len(query_distortion)))
                continue
            distortion_list.append(query_distortion[queries[find_index]])
        distortion_list = np.array(distortion_list)
        distortion_list = distortion_list[~np.isnan(distortion_list)]  # 去掉nan的值
        mean_distortion = np.mean(distortion_list)
        median_distortion = np.median(distortion_list)

        success_list = distortion_list <= success_distortion_threshold
        success_list = success_list.astype(np.float32)
        success_rate = np.mean(success_list) * 100.0
        query_to_ASR[query_budget] = round(success_rate,1)
        x.append(query_budget)
        y_distortions.append(mean_distortion)

        if want_key == "mean_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(mean_distortion.item(),3))
        elif want_key =="median_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(median_distortion.item(),3))
    x = np.array(x)
    y_distortions = np.array(y_distortions)
    AUC = round(auc(x, y_distortions), 1)
    return mean_and_median_distortions, query_to_ASR


def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, query_budgets,
                                           wanted_one_prior1_list, wanted_one_prior2_list, wanted_two_priors_list,
                                            success_distortion_threshold, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result_distortion = defaultdict(lambda : defaultdict(lambda : "-"))
    result_ASR = defaultdict(lambda : defaultdict(lambda : "-"))
    for method, folder in folder_list.items():
        if not os.path.exists(folder):
            continue
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if file_name.startswith(arch) and file_name.endswith(".json"):
                if method in ["RayS","GeoDA"] and targeted:
                    print("{} does not exist!".format(file_path))
                    result_distortion[method] = defaultdict(lambda : "-")
                    result_ASR[method] = defaultdict(lambda : "-")
                    continue
                if not os.path.exists(file_path):
                    distortion_dict = {}
                else:
                    if method.endswith("prior1"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_arch"]:
                                json_surrogate_arch.append(json_content["args"]["surrogate_arch"])
                            elif json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            distortion_dict = json_content["distortion"]
                        if len(json_surrogate_arch) != 1 or json_surrogate_arch[0] not in wanted_one_prior1_list:
                            distortion_dict.clear()
                            continue
                        print("Read prior1 : " + file_path)
                    elif method.endswith("prior2"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_arch"]:
                                json_surrogate_arch.append(json_content["args"]["surrogate_arch"])
                            elif json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            distortion_dict = json_content["distortion"]
                        if len(json_surrogate_arch) != 1 or json_surrogate_arch[0] not in wanted_one_prior2_list:
                            distortion_dict.clear()
                            continue
                        print("Read prior2  : " + file_path)
                    elif method.endswith("2priors"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            distortion_dict = json_content["distortion"]
                        two_prior_found = False
                        if len(json_surrogate_arch) != 2:
                            distortion_dict.clear()
                            continue
                        for two_prior in wanted_two_priors_list:
                            if len(set(two_prior) & set(json_surrogate_arch)) == 2:
                                two_prior_found = True
                        if not two_prior_found:
                            distortion_dict.clear()
                            continue
                        print("Read  : " + file_path)
                    elif method.endswith("_PGD_init_theta"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_arch"]:
                                json_surrogate_arch.append(json_content["args"]["surrogate_arch"])
                            elif json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            distortion_dict = json_content["distortion"]
                        if len(json_surrogate_arch) != 1 or json_surrogate_arch[0] not in wanted_one_prior1_list:
                            distortion_dict.clear()
                            continue
                        print("Read PGD : " + file_path)
                    else:
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            distortion_dict = json_content["distortion"]
                            print("Read  : " + file_path)

                mean_and_median_distortions, query_to_ASR = get_mean_distortion_ASR(distortion_dict, query_budgets,
                                                                                    success_distortion_threshold, want_key)
                result_distortion[method] = mean_and_median_distortions
                result_ASR[method] = query_to_ASR
    return result_distortion





def fetch_json_content_for_multiple_priors(dataset, norm, targeted, arch, query_budgets, success_distortion_threshold, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result_distortion = defaultdict(lambda : defaultdict(lambda : "-"))
    result_ASR = defaultdict(lambda :defaultdict(lambda : '-'))
    for method, folder in folder_list.items():
        if not os.path.exists(folder):
            continue
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if file_name.startswith(arch) and file_name.endswith(".json"):
                if method in ["RayS","GeoDA"] and targeted:
                    print("{} does not exist!".format(file_path))
                    result_distortion[method] = defaultdict(lambda : "-")
                    result_ASR[method] = defaultdict(lambda : "-")
                    continue
                if not os.path.exists(file_path):
                    distortion_dict = {}
                else:
                    if method.startswith("Prior") or method.startswith("Pure"):
                        json_surrogate_arch = []
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            if json_content["args"]["surrogate_archs"]:
                                json_surrogate_arch.extend(json_content["args"]["surrogate_archs"])
                            else:
                                json_surrogate_arch.append(json_content["args"]["surrogate_arch"])
                            distortion_dict = json_content["distortion"]
                        new_method = method+"-{}prior".format(len(json_surrogate_arch))
                        if len(json_surrogate_arch) == 1 and json_surrogate_arch[0] == "convit_base":
                            continue
                        if len(json_surrogate_arch) > 2:
                            continue
                        print("{}  : ".format(new_method) + file_path + " Archs:{}".format( ",".join(json_surrogate_arch)))
                    else:
                        with open(file_path, "r") as file_obj:
                            json_content = json.load(file_obj)
                            distortion_dict = json_content["distortion"]
                            print("Read  : " + file_path)
                        new_method = method

                mean_and_median_distortions, query_to_ASR = get_mean_distortion_ASR(distortion_dict, query_budgets, success_distortion_threshold, want_key)
                result_distortion[new_method] = mean_and_median_distortions
                result_ASR[new_method] = query_to_ASR
    return result_distortion, result_ASR


def draw_tables(result, surrogate_1, surrogate_2, surrogate_3, surrogate_4, surrogate_5):
    print("""
        Sign-OPT \\cite{{cheng2019sign}}& no prior & {SignOPT_1000} & {SignOPT_2000} & {SignOPT_5000} & {SignOPT_8000} & {SignOPT_10000}  & {SignOPT_12000} & {SignOPT_15000} &  {SignOPT_18000}  &  {SignOPT_20000}  \\\\
        Pure-Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}}} & 1 prior & {PurePriorSignOPT_prior1_1000} & {PurePriorSignOPT_prior1_2000} & {PurePriorSignOPT_prior1_5000} & {PurePriorSignOPT_prior1_8000} & {PurePriorSignOPT_prior1_10000}  & {PurePriorSignOPT_prior1_12000}  &  {PurePriorSignOPT_prior1_15000} &  {PurePriorSignOPT_prior1_18000} &  {PurePriorSignOPT_prior1_20000} \\\\
        Pure-Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & 2 priors  & {PurePriorSignOPT_prior2_1000} & {PurePriorSignOPT_prior2_2000} & {PurePriorSignOPT_prior2_5000} & {PurePriorSignOPT_prior2_8000} & {PurePriorSignOPT_prior2_10000}  & {PurePriorSignOPT_prior2_12000}  &  {PurePriorSignOPT_prior2_15000} &  {PurePriorSignOPT_prior2_18000} &  {PurePriorSignOPT_prior2_20000} \\\\
        Pure-Prior-OPT\\textsubscript{{\\tiny {surrogate_1}}} & 1 prior & {PurePriorOPT_prior1_1000} & {PurePriorOPT_prior1_2000} & {PurePriorOPT_prior1_5000} & {PurePriorOPT_prior1_8000} & {PurePriorOPT_prior1_10000}  & {PurePriorOPT_prior1_12000}  &  {PurePriorOPT_prior1_15000} &  {PurePriorOPT_prior1_18000} &  {PurePriorOPT_prior1_20000} \\\\
        Pure-Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & 2 priors  & {PurePriorOPT_prior2_1000} & {PurePriorOPT_prior2_2000} & {PurePriorOPT_prior2_5000} & {PurePriorOPT_prior2_8000} & {PurePriorOPT_prior2_10000}  & {PurePriorOPT_prior2_12000}  &  {PurePriorOPT_prior2_15000} &  {PurePriorOPT_prior2_18000} &  {PurePriorOPT_prior2_20000} \\\\
        Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}}} & 1 prior & {PriorSignOPT_prior1_1000} & {PriorSignOPT_prior1_2000} & {PriorSignOPT_prior1_5000} & {PriorSignOPT_prior1_8000} & {PriorSignOPT_prior1_10000}  & {PriorSignOPT_prior1_12000}  &  {PriorSignOPT_prior1_15000} &  {PriorSignOPT_prior1_18000} &  {PriorSignOPT_prior1_20000} \\\\
        Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & 2 priors  & {PriorSignOPT_prior2_1000} & {PriorSignOPT_prior2_2000} & {PriorSignOPT_prior2_5000} & {PriorSignOPT_prior2_8000} & {PriorSignOPT_prior2_10000}  & {PriorSignOPT_prior2_12000}  &  {PriorSignOPT_prior2_15000} &  {PriorSignOPT_prior2_18000} &  {PriorSignOPT_prior2_20000} \\\\
        Prior-OPT\\textsubscript{{\\tiny {surrogate_1}}} & 1 prior & {PriorOPT_prior1_1000} & {PriorOPT_prior1_2000} & {PriorOPT_prior1_5000} & {PriorOPT_prior1_8000} & {PriorOPT_prior1_10000}  & {PriorOPT_prior1_12000}  &  {PriorOPT_prior1_15000} &  {PriorOPT_prior1_18000} &  {PriorOPT_prior1_20000} \\\\
        Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & 2 priors  & {PriorOPT_prior2_1000} & {PriorOPT_prior2_2000} & {PriorOPT_prior2_5000} & {PriorOPT_prior2_8000} & {PriorOPT_prior2_10000}  & {PriorOPT_prior2_12000}  &  {PriorOPT_prior2_15000} &  {PriorOPT_prior2_18000} &  {PriorOPT_prior2_20000} \\\\
                """.format(
        surrogate_1=surrogate_1,
        surrogate_2=surrogate_2,

        SignOPT_1000=result["Sign-OPT"][1000],
        SignOPT_2000=result["Sign-OPT"][2000],
        SignOPT_5000=result["Sign-OPT"][5000],
        SignOPT_8000=result["Sign-OPT"][8000],
        SignOPT_10000=result["Sign-OPT"][10000],
        SignOPT_12000=result["Sign-OPT"][12000],
        SignOPT_15000=result["Sign-OPT"][15000],
        SignOPT_18000=result["Sign-OPT"][18000],
        SignOPT_20000=result["Sign-OPT"][20000],

        PurePriorSignOPT_prior1_1000=result["Pure-Prior-Sign-OPT-1prior"][1000],
        PurePriorSignOPT_prior1_2000=result["Pure-Prior-Sign-OPT-1prior"][2000],
        PurePriorSignOPT_prior1_5000=result["Pure-Prior-Sign-OPT-1prior"][5000],
        PurePriorSignOPT_prior1_8000=result["Pure-Prior-Sign-OPT-1prior"][8000],
        PurePriorSignOPT_prior1_10000=result["Pure-Prior-Sign-OPT-1prior"][10000],
        PurePriorSignOPT_prior1_12000=result["Pure-Prior-Sign-OPT-1prior"][12000],
        PurePriorSignOPT_prior1_15000=result["Pure-Prior-Sign-OPT-1prior"][15000],
        PurePriorSignOPT_prior1_18000=result["Pure-Prior-Sign-OPT-1prior"][18000],
        PurePriorSignOPT_prior1_20000=result["Pure-Prior-Sign-OPT-1prior"][20000],

        PurePriorSignOPT_prior2_1000=result["Pure-Prior-Sign-OPT-2prior"][1000],
        PurePriorSignOPT_prior2_2000=result["Pure-Prior-Sign-OPT-2prior"][2000],
        PurePriorSignOPT_prior2_5000=result["Pure-Prior-Sign-OPT-2prior"][5000],
        PurePriorSignOPT_prior2_8000=result["Pure-Prior-Sign-OPT-2prior"][8000],
        PurePriorSignOPT_prior2_10000=result["Pure-Prior-Sign-OPT-2prior"][10000],
        PurePriorSignOPT_prior2_12000=result["Pure-Prior-Sign-OPT-2prior"][12000],
        PurePriorSignOPT_prior2_15000=result["Pure-Prior-Sign-OPT-2prior"][15000],
        PurePriorSignOPT_prior2_18000=result["Pure-Prior-Sign-OPT-2prior"][18000],
        PurePriorSignOPT_prior2_20000=result["Pure-Prior-Sign-OPT-2prior"][20000],

        PurePriorOPT_prior1_1000=result["Pure-Prior-OPT-1prior"][1000],
        PurePriorOPT_prior1_2000=result["Pure-Prior-OPT-1prior"][2000],
        PurePriorOPT_prior1_5000=result["Pure-Prior-OPT-1prior"][5000],
        PurePriorOPT_prior1_8000=result["Pure-Prior-OPT-1prior"][8000],
        PurePriorOPT_prior1_10000=result["Pure-Prior-OPT-1prior"][10000],
        PurePriorOPT_prior1_12000=result["Pure-Prior-OPT-1prior"][12000],
        PurePriorOPT_prior1_15000=result["Pure-Prior-OPT-1prior"][15000],
        PurePriorOPT_prior1_18000=result["Pure-Prior-OPT-1prior"][18000],
        PurePriorOPT_prior1_20000=result["Pure-Prior-OPT-1prior"][20000],

        PurePriorOPT_prior2_1000=result["Pure-Prior-OPT-2prior"][1000],
        PurePriorOPT_prior2_2000=result["Pure-Prior-OPT-2prior"][2000],
        PurePriorOPT_prior2_5000=result["Pure-Prior-OPT-2prior"][5000],
        PurePriorOPT_prior2_8000=result["Pure-Prior-OPT-2prior"][8000],
        PurePriorOPT_prior2_10000=result["Pure-Prior-OPT-2prior"][10000],
        PurePriorOPT_prior2_12000=result["Pure-Prior-OPT-2prior"][12000],
        PurePriorOPT_prior2_15000=result["Pure-Prior-OPT-2prior"][15000],
        PurePriorOPT_prior2_18000=result["Pure-Prior-OPT-2prior"][18000],
        PurePriorOPT_prior2_20000=result["Pure-Prior-OPT-2prior"][20000],

        PriorSignOPT_prior1_1000=result["Prior-Sign-OPT-1prior"][1000],
        PriorSignOPT_prior1_2000=result["Prior-Sign-OPT-1prior"][2000],
        PriorSignOPT_prior1_5000=result["Prior-Sign-OPT-1prior"][5000],
        PriorSignOPT_prior1_8000=result["Prior-Sign-OPT-1prior"][8000],
        PriorSignOPT_prior1_10000=result["Prior-Sign-OPT-1prior"][10000],
        PriorSignOPT_prior1_12000=result["Prior-Sign-OPT-1prior"][12000],
        PriorSignOPT_prior1_15000=result["Prior-Sign-OPT-1prior"][15000],
        PriorSignOPT_prior1_18000=result["Prior-Sign-OPT-1prior"][18000],
        PriorSignOPT_prior1_20000=result["Prior-Sign-OPT-1prior"][20000],

        PriorSignOPT_prior2_1000=result["Prior-Sign-OPT-2prior"][1000],
        PriorSignOPT_prior2_2000=result["Prior-Sign-OPT-2prior"][2000],
        PriorSignOPT_prior2_5000=result["Prior-Sign-OPT-2prior"][5000],
        PriorSignOPT_prior2_8000=result["Prior-Sign-OPT-2prior"][8000],
        PriorSignOPT_prior2_10000=result["Prior-Sign-OPT-2prior"][10000],
        PriorSignOPT_prior2_12000=result["Prior-Sign-OPT-2prior"][12000],
        PriorSignOPT_prior2_15000=result["Prior-Sign-OPT-2prior"][15000],
        PriorSignOPT_prior2_18000=result["Prior-Sign-OPT-2prior"][18000],
        PriorSignOPT_prior2_20000=result["Prior-Sign-OPT-2prior"][20000],

        PriorOPT_prior1_1000=result["Prior-OPT-1prior"][1000],
        PriorOPT_prior1_2000=result["Prior-OPT-1prior"][2000],
        PriorOPT_prior1_5000=result["Prior-OPT-1prior"][5000],
        PriorOPT_prior1_8000=result["Prior-OPT-1prior"][8000],
        PriorOPT_prior1_10000=result["Prior-OPT-1prior"][10000],
        PriorOPT_prior1_12000=result["Prior-OPT-1prior"][12000],
        PriorOPT_prior1_15000=result["Prior-OPT-1prior"][15000],
        PriorOPT_prior1_18000=result["Prior-OPT-1prior"][18000],
        PriorOPT_prior1_20000=result["Prior-OPT-1prior"][20000],

        PriorOPT_prior2_1000=result["Prior-OPT-2prior"][1000],
        PriorOPT_prior2_2000=result["Prior-OPT-2prior"][2000],
        PriorOPT_prior2_5000=result["Prior-OPT-2prior"][5000],
        PriorOPT_prior2_8000=result["Prior-OPT-2prior"][8000],
        PriorOPT_prior2_10000=result["Prior-OPT-2prior"][10000],
        PriorOPT_prior2_12000=result["Prior-OPT-2prior"][12000],
        PriorOPT_prior2_15000=result["Prior-OPT-2prior"][15000],
        PriorOPT_prior2_18000=result["Prior-OPT-2prior"][18000],
        PriorOPT_prior2_20000=result["Prior-OPT-2prior"][20000],

    )
    )


def draw_tables_for_ImageNet(result, surrogate_1, surrogate_2):
    print("""
                & HSJA \\cite{{chen2019hopskipjumpattack}} & {HSJA_1000_untargeted} & {HSJA_2000_untargeted} & {HSJA_5000_untargeted} & {HSJA_8000_untargeted} & {HSJA_10000_untargeted}  & {HSJA_1000_targeted} & {HSJA_2000_targeted} & {HSJA_5000_targeted} & {HSJA_8000_targeted} & {HSJA_10000_targeted} & {HSJA_15000_targeted} & {HSJA_20000_targeted}\\\\
				& TA \\cite{{ma2021finding}} & {TA_1000_untargeted} & {TA_2000_untargeted} & {TA_5000_untargeted} & {TA_8000_untargeted} & {TA_10000_untargeted}  & {TA_1000_targeted} & {TA_2000_targeted} & {TA_5000_targeted} & {TA_8000_targeted} & {TA_10000_targeted}& {TA_15000_targeted} & {TA_20000_targeted}\\\\
				& G-TA \\cite{{ma2021finding}}  & {GTA_1000_untargeted} & {GTA_2000_untargeted} & {GTA_5000_untargeted} & {GTA_8000_untargeted} & {GTA_10000_untargeted}  & {GTA_1000_targeted} & {GTA_2000_targeted} & {GTA_5000_targeted} & {GTA_8000_targeted} & {GTA_10000_targeted} & {GTA_15000_targeted} & {GTA_20000_targeted}\\\\
				& QEBA \\cite{{li2020qeba}} & {QEBA_1000_untargeted} & {QEBA_2000_untargeted} & {QEBA_5000_untargeted} & {QEBA_8000_untargeted} & {QEBA_10000_untargeted}  & {QEBA_1000_targeted} & {QEBA_2000_targeted} & {QEBA_5000_targeted} & {QEBA_8000_targeted} & {QEBA_10000_targeted} & {QEBA_15000_targeted} & {QEBA_20000_targeted}\\\\
                & Sign-OPT \\cite{{cheng2019sign}} & {SignOPT_1000_untargeted} & {SignOPT_2000_untargeted} & {SignOPT_5000_untargeted} & {SignOPT_8000_untargeted} & {SignOPT_10000_untargeted}  & {SignOPT_1000_targeted} & {SignOPT_2000_targeted} & {SignOPT_5000_targeted} & {SignOPT_8000_targeted} & {SignOPT_10000_targeted} & {SignOPT_15000_targeted} & {SignOPT_20000_targeted}\\\\
				& SVM-OPT \\cite{{cheng2019sign}} & {SVMOPT_1000_untargeted} & {SVMOPT_2000_untargeted} & {SVMOPT_5000_untargeted} & {SVMOPT_8000_untargeted} & {SVMOPT_10000_untargeted}  & {SVMOPT_1000_targeted} & {SVMOPT_2000_targeted} & {SVMOPT_5000_targeted} & {SVMOPT_8000_targeted} & {SVMOPT_10000_targeted}& {SVMOPT_15000_targeted} & {SVMOPT_20000_targeted}\\\\
                & GeoDA \\cite{{rahmati2020geoda}} & {GeoDA_1000_untargeted} & {GeoDA_2000_untargeted} & {GeoDA_5000_untargeted} & {GeoDA_8000_untargeted} & {GeoDA_10000_untargeted}  & {GeoDA_1000_targeted} & {GeoDA_2000_targeted} & {GeoDA_5000_targeted} & {GeoDA_8000_targeted} & {GeoDA_10000_targeted} & {GeoDA_15000_targeted} & {GeoDA_20000_targeted}\\\\
                & Evolutionary \\cite{{dong2019efficient}} & {Evolutionary_1000_untargeted} & {Evolutionary_2000_untargeted} & {Evolutionary_5000_untargeted} & {Evolutionary_8000_untargeted} & {Evolutionary_10000_untargeted}  & {Evolutionary_1000_targeted} & {Evolutionary_2000_targeted} & {Evolutionary_5000_targeted} & {Evolutionary_8000_targeted} & {Evolutionary_10000_targeted}  & {Evolutionary_15000_targeted} & {Evolutionary_20000_targeted}\\\\
                & SurFree \\cite{{maho2021surfree}}  & {SurFree_1000_untargeted} & {SurFree_2000_untargeted} & {SurFree_5000_untargeted} & {SurFree_8000_untargeted} & {SurFree_10000_untargeted}  & {SurFree_1000_targeted} & {SurFree_2000_targeted} & {SurFree_5000_targeted} & {SurFree_8000_targeted} & {SurFree_10000_targeted}  & {SurFree_15000_targeted} & {SurFree_20000_targeted}\\\\
                & Triangle Attack \\cite{{wang2022triangle}} & {TriangleAttack_1000_untargeted} & {TriangleAttack_2000_untargeted} & {TriangleAttack_5000_untargeted} & {TriangleAttack_8000_untargeted} & {TriangleAttack_10000_untargeted}  & {TriangleAttack_1000_targeted} & {TriangleAttack_2000_targeted} & {TriangleAttack_5000_targeted} & {TriangleAttack_8000_targeted} & {TriangleAttack_10000_targeted} & {TriangleAttack_15000_targeted} & {TriangleAttack_20000_targeted}\\\\
                & CGBA-H \\cite{{reza2023cgba}} & {CGBA_1000_untargeted} & {CGBA_2000_untargeted} & {CGBA_5000_untargeted} & {CGBA_8000_untargeted} & {CGBA_10000_untargeted}  & {CGBA_1000_targeted} & {CGBA_2000_targeted} & {CGBA_5000_targeted} & {CGBA_8000_targeted} & {CGBA_10000_targeted} & {CGBA_15000_targeted} & {CGBA_20000_targeted}\\\\
                & SQBA\\textsubscript{{\\tiny {surrogate_1}}} \\cite{{Park_2024_sqba}} & {SQBA_prior1_1000_untargeted} & {SQBA_prior1_2000_untargeted} & {SQBA_prior1_5000_untargeted} & {SQBA_prior1_8000_untargeted} & {SQBA_prior1_10000_untargeted}  & {SQBA_prior1_1000_targeted} & {SQBA_prior1_2000_targeted} & {SQBA_prior1_5000_targeted} & {SQBA_prior1_8000_targeted} & {SQBA_prior1_10000_targeted}  & {SQBA_prior1_15000_targeted} & {SQBA_prior1_20000_targeted}\\\\
                & SQBA\\textsubscript{{\\tiny {surrogate_2}}} \\cite{{Park_2024_sqba}} & {SQBA_prior2_1000_untargeted} & {SQBA_prior2_2000_untargeted} & {SQBA_prior2_5000_untargeted} & {SQBA_prior2_8000_untargeted} & {SQBA_prior2_10000_untargeted}  & {SQBA_prior2_1000_targeted} & {SQBA_prior2_2000_targeted} & {SQBA_prior2_5000_targeted} & {SQBA_prior2_8000_targeted} & {SQBA_prior2_10000_targeted}  & {SQBA_prior2_15000_targeted} & {SQBA_prior2_20000_targeted}\\\\
                & BBA\\textsubscript{{\\tiny {surrogate_1}}} \\cite{{brunner2019guessing}}  & {BBA_prior1_1000_untargeted} & {BBA_prior1_2000_untargeted} & {BBA_prior1_5000_untargeted} & {BBA_prior1_8000_untargeted} & {BBA_prior1_10000_untargeted}  & {BBA_prior1_1000_targeted} & {BBA_prior1_2000_targeted} & {BBA_prior1_5000_targeted} & {BBA_prior1_8000_targeted} & {BBA_prior1_10000_targeted}  & {BBA_prior1_15000_targeted} & {BBA_prior1_20000_targeted}\\\\
                & BBA\\textsubscript{{\\tiny {surrogate_2}}} \\cite{{brunner2019guessing}}  & {BBA_prior2_1000_untargeted} & {BBA_prior2_2000_untargeted} & {BBA_prior2_5000_untargeted} & {BBA_prior2_8000_untargeted} & {BBA_prior2_10000_untargeted}  & {BBA_prior2_1000_targeted} & {BBA_prior2_2000_targeted} & {BBA_prior2_5000_targeted} & {BBA_prior2_8000_targeted} & {BBA_prior2_10000_targeted}  & {BBA_prior2_15000_targeted} & {BBA_prior2_20000_targeted}\\\\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}}} & {PriorSignOPT_prior1_1000_untargeted} & {PriorSignOPT_prior1_2000_untargeted} & {PriorSignOPT_prior1_5000_untargeted} & {PriorSignOPT_prior1_8000_untargeted} & {PriorSignOPT_prior1_10000_untargeted}  & {PriorSignOPT_prior1_1000_targeted} & {PriorSignOPT_prior1_2000_targeted} & {PriorSignOPT_prior1_5000_targeted} & {PriorSignOPT_prior1_8000_targeted} & {PriorSignOPT_prior1_10000_targeted} & {PriorSignOPT_prior1_15000_targeted} & {PriorSignOPT_prior1_20000_targeted} \\\\
                & Prior-Sign-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}}   & {PriorSignOPT_2priors_1000_untargeted} & {PriorSignOPT_2priors_2000_untargeted} & {PriorSignOPT_2priors_5000_untargeted} & {PriorSignOPT_2priors_8000_untargeted} & {PriorSignOPT_2priors_10000_untargeted}  & {PriorSignOPT_2priors_1000_targeted} & {PriorSignOPT_2priors_2000_targeted} & {PriorSignOPT_2priors_5000_targeted} & {PriorSignOPT_2priors_8000_targeted} & {PriorSignOPT_2priors_10000_targeted} & {PriorSignOPT_2priors_15000_targeted} & {PriorSignOPT_2priors_20000_targeted} \\\\
                & Prior-Sign-OPT\\textsubscript{{\\tiny $\\theta_0^\\text{{PGD}}$ + {surrogate_1}}}  & {PriorSignOPT_PGD_init_theta_1000_untargeted} & {PriorSignOPT_PGD_init_theta_2000_untargeted} & {PriorSignOPT_PGD_init_theta_5000_untargeted} & {PriorSignOPT_PGD_init_theta_8000_untargeted} & {PriorSignOPT_PGD_init_theta_10000_untargeted}  & {PriorSignOPT_PGD_init_theta_1000_targeted} & {PriorSignOPT_PGD_init_theta_2000_targeted} & {PriorSignOPT_PGD_init_theta_5000_targeted} & {PriorSignOPT_PGD_init_theta_8000_targeted} & {PriorSignOPT_PGD_init_theta_10000_targeted}  & {PriorSignOPT_PGD_init_theta_15000_targeted} & {PriorSignOPT_PGD_init_theta_20000_targeted} \\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}}} & {PriorOPT_prior1_1000_untargeted} & {PriorOPT_prior1_2000_untargeted} & {PriorOPT_prior1_5000_untargeted} & {PriorOPT_prior1_8000_untargeted} & {PriorOPT_prior1_10000_untargeted}  & {PriorOPT_prior1_1000_targeted} & {PriorOPT_prior1_2000_targeted} & {PriorOPT_prior1_5000_targeted} & {PriorOPT_prior1_8000_targeted} & {PriorOPT_prior1_10000_targeted} & {PriorOPT_prior1_15000_targeted} & {PriorOPT_prior1_20000_targeted}\\\\
                & Prior-OPT\\textsubscript{{\\tiny {surrogate_1}\&{surrogate_2}}} & {PriorOPT_2priors_1000_untargeted} & {PriorOPT_2priors_2000_untargeted} & {PriorOPT_2priors_5000_untargeted} & {PriorOPT_2priors_8000_untargeted} & {PriorOPT_2priors_10000_untargeted}  & {PriorOPT_2priors_1000_targeted} & {PriorOPT_2priors_2000_targeted} & {PriorOPT_2priors_5000_targeted} & {PriorOPT_2priors_8000_targeted} & {PriorOPT_2priors_10000_targeted} & {PriorOPT_2priors_15000_targeted} & {PriorOPT_2priors_20000_targeted} \\\\
                & Prior-OPT\\textsubscript{{\\tiny $\\theta_0^\\text{{PGD}}$ + {surrogate_1}}} & {PriorOPT_PGD_init_theta_1000_untargeted} & {PriorOPT_PGD_init_theta_2000_untargeted} & {PriorOPT_PGD_init_theta_5000_untargeted} & {PriorOPT_PGD_init_theta_8000_untargeted} & {PriorOPT_PGD_init_theta_10000_untargeted}  & {PriorOPT_PGD_init_theta_1000_targeted} & {PriorOPT_PGD_init_theta_2000_targeted} & {PriorOPT_PGD_init_theta_5000_targeted} & {PriorOPT_PGD_init_theta_8000_targeted} & {PriorOPT_PGD_init_theta_10000_targeted} & {PriorOPT_PGD_init_theta_15000_targeted} & {PriorOPT_PGD_init_theta_20000_targeted} \\\\ 
                """.format(
        surrogate_1=surrogate_1,
        surrogate_2=surrogate_2,

        BBA_prior1_1000_untargeted=result[False]["BBA-prior1"][1000],
        BBA_prior1_2000_untargeted=result[False]["BBA-prior1"][2000],
        BBA_prior1_5000_untargeted=result[False]["BBA-prior1"][5000],
        BBA_prior1_8000_untargeted=result[False]["BBA-prior1"][8000],
        BBA_prior1_10000_untargeted=result[False]["BBA-prior1"][10000],

        BBA_prior2_1000_untargeted=result[False]["BBA-prior2"][1000],
        BBA_prior2_2000_untargeted=result[False]["BBA-prior2"][2000],
        BBA_prior2_5000_untargeted=result[False]["BBA-prior2"][5000],
        BBA_prior2_8000_untargeted=result[False]["BBA-prior2"][8000],
        BBA_prior2_10000_untargeted=result[False]["BBA-prior2"][10000],

        HSJA_1000_untargeted=result[False]["HSJA"][1000],
        HSJA_2000_untargeted=result[False]["HSJA"][2000],
        HSJA_5000_untargeted=result[False]["HSJA"][5000],
        HSJA_8000_untargeted=result[False]["HSJA"][8000],
        HSJA_10000_untargeted=result[False]["HSJA"][10000],

        TA_1000_untargeted=result[False]["TA"][1000],
        TA_2000_untargeted=result[False]["TA"][2000],
        TA_5000_untargeted=result[False]["TA"][5000],
        TA_8000_untargeted=result[False]["TA"][8000],
        TA_10000_untargeted=result[False]["TA"][10000],

        GTA_1000_untargeted=result[False]["G-TA"][1000],
        GTA_2000_untargeted=result[False]["G-TA"][2000],
        GTA_5000_untargeted=result[False]["G-TA"][5000],
        GTA_8000_untargeted=result[False]["G-TA"][8000],
        GTA_10000_untargeted=result[False]["G-TA"][10000],

        QEBA_1000_untargeted=result[False]["QEBA"][1000],
        QEBA_2000_untargeted=result[False]["QEBA"][2000],
        QEBA_5000_untargeted=result[False]["QEBA"][5000],
        QEBA_8000_untargeted=result[False]["QEBA"][8000],
        QEBA_10000_untargeted=result[False]["QEBA"][10000],

        SignOPT_1000_untargeted=result[False]["Sign-OPT"][1000],
        SignOPT_2000_untargeted=result[False]["Sign-OPT"][2000],
        SignOPT_5000_untargeted=result[False]["Sign-OPT"][5000],
        SignOPT_8000_untargeted=result[False]["Sign-OPT"][8000],
        SignOPT_10000_untargeted=result[False]["Sign-OPT"][10000],

        SVMOPT_1000_untargeted=result[False]["SVM-OPT"][1000],
        SVMOPT_2000_untargeted=result[False]["SVM-OPT"][2000],
        SVMOPT_5000_untargeted=result[False]["SVM-OPT"][5000],
        SVMOPT_8000_untargeted=result[False]["SVM-OPT"][8000],
        SVMOPT_10000_untargeted=result[False]["SVM-OPT"][10000],

        GeoDA_1000_untargeted=result[False]["GeoDA"][1000],
        GeoDA_2000_untargeted=result[False]["GeoDA"][2000],
        GeoDA_5000_untargeted=result[False]["GeoDA"][5000],
        GeoDA_8000_untargeted=result[False]["GeoDA"][8000],
        GeoDA_10000_untargeted=result[False]["GeoDA"][10000],

        Evolutionary_1000_untargeted=result[False]["Evolutionary"][1000],
        Evolutionary_2000_untargeted=result[False]["Evolutionary"][2000],
        Evolutionary_5000_untargeted=result[False]["Evolutionary"][5000],
        Evolutionary_8000_untargeted=result[False]["Evolutionary"][8000],
        Evolutionary_10000_untargeted=result[False]["Evolutionary"][10000],

        SurFree_1000_untargeted=result[False]["SurFree"][1000],
        SurFree_2000_untargeted=result[False]["SurFree"][2000],
        SurFree_5000_untargeted=result[False]["SurFree"][5000],
        SurFree_8000_untargeted=result[False]["SurFree"][8000],
        SurFree_10000_untargeted=result[False]["SurFree"][10000],

        TriangleAttack_1000_untargeted=result[False]["Triangle Attack"][1000],
        TriangleAttack_2000_untargeted=result[False]["Triangle Attack"][2000],
        TriangleAttack_5000_untargeted=result[False]["Triangle Attack"][5000],
        TriangleAttack_8000_untargeted=result[False]["Triangle Attack"][8000],
        TriangleAttack_10000_untargeted=result[False]["Triangle Attack"][10000],

        CGBA_1000_untargeted=result[False]["CGBA_H"][1000],
        CGBA_2000_untargeted=result[False]["CGBA_H"][2000],
        CGBA_5000_untargeted=result[False]["CGBA_H"][5000],
        CGBA_8000_untargeted=result[False]["CGBA_H"][8000],
        CGBA_10000_untargeted=result[False]["CGBA_H"][10000],

        SQBA_prior1_1000_untargeted=result[False]["SQBA-prior1"][1000],
        SQBA_prior1_2000_untargeted=result[False]["SQBA-prior1"][2000],
        SQBA_prior1_5000_untargeted=result[False]["SQBA-prior1"][5000],
        SQBA_prior1_8000_untargeted=result[False]["SQBA-prior1"][8000],
        SQBA_prior1_10000_untargeted=result[False]["SQBA-prior1"][10000],

        SQBA_prior2_1000_untargeted=result[False]["SQBA-prior2"][1000],
        SQBA_prior2_2000_untargeted=result[False]["SQBA-prior2"][2000],
        SQBA_prior2_5000_untargeted=result[False]["SQBA-prior2"][5000],
        SQBA_prior2_8000_untargeted=result[False]["SQBA-prior2"][8000],
        SQBA_prior2_10000_untargeted=result[False]["SQBA-prior2"][10000],

        PriorSignOPT_prior1_1000_untargeted=result[False]["Prior-Sign-OPT-prior1"][1000],
        PriorSignOPT_prior1_2000_untargeted=result[False]["Prior-Sign-OPT-prior1"][2000],
        PriorSignOPT_prior1_5000_untargeted=result[False]["Prior-Sign-OPT-prior1"][5000],
        PriorSignOPT_prior1_8000_untargeted=result[False]["Prior-Sign-OPT-prior1"][8000],
        PriorSignOPT_prior1_10000_untargeted=result[False]["Prior-Sign-OPT-prior1"][10000],

        PriorSignOPT_2priors_1000_untargeted=result[False]["Prior-Sign-OPT-2priors"][1000],
        PriorSignOPT_2priors_2000_untargeted=result[False]["Prior-Sign-OPT-2priors"][2000],
        PriorSignOPT_2priors_5000_untargeted=result[False]["Prior-Sign-OPT-2priors"][5000],
        PriorSignOPT_2priors_8000_untargeted=result[False]["Prior-Sign-OPT-2priors"][8000],
        PriorSignOPT_2priors_10000_untargeted=result[False]["Prior-Sign-OPT-2priors"][10000],

        PriorSignOPT_PGD_init_theta_1000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][1000],
        PriorSignOPT_PGD_init_theta_2000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][2000],
        PriorSignOPT_PGD_init_theta_5000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][5000],
        PriorSignOPT_PGD_init_theta_8000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][8000],
        PriorSignOPT_PGD_init_theta_10000_untargeted=result[False]["Prior-Sign-OPT_PGD_init_theta"][10000],

        PriorOPT_prior1_1000_untargeted=result[False]["Prior-OPT-prior1"][1000],
        PriorOPT_prior1_2000_untargeted=result[False]["Prior-OPT-prior1"][2000],
        PriorOPT_prior1_5000_untargeted=result[False]["Prior-OPT-prior1"][5000],
        PriorOPT_prior1_8000_untargeted=result[False]["Prior-OPT-prior1"][8000],
        PriorOPT_prior1_10000_untargeted=result[False]["Prior-OPT-prior1"][10000],

        PriorOPT_2priors_1000_untargeted=result[False]["Prior-OPT-2priors"][1000],
        PriorOPT_2priors_2000_untargeted=result[False]["Prior-OPT-2priors"][2000],
        PriorOPT_2priors_5000_untargeted=result[False]["Prior-OPT-2priors"][5000],
        PriorOPT_2priors_8000_untargeted=result[False]["Prior-OPT-2priors"][8000],
        PriorOPT_2priors_10000_untargeted=result[False]["Prior-OPT-2priors"][10000],

        PriorOPT_PGD_init_theta_1000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][1000],
        PriorOPT_PGD_init_theta_2000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][2000],
        PriorOPT_PGD_init_theta_5000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][5000],
        PriorOPT_PGD_init_theta_8000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][8000],
        PriorOPT_PGD_init_theta_10000_untargeted=result[False]["Prior-OPT_PGD_init_theta"][10000],

        BBA_prior1_1000_targeted=result[True]["BBA-prior1"][1000],
        BBA_prior1_2000_targeted=result[True]["BBA-prior1"][2000],
        BBA_prior1_5000_targeted=result[True]["BBA-prior1"][5000],
        BBA_prior1_8000_targeted=result[True]["BBA-prior1"][8000],
        BBA_prior1_10000_targeted=result[True]["BBA-prior1"][10000],
        BBA_prior1_15000_targeted=result[True]["BBA-prior1"][15000],
        BBA_prior1_20000_targeted=result[True]["BBA-prior1"][20000],

        BBA_prior2_1000_targeted=result[True]["BBA-prior2"][1000],
        BBA_prior2_2000_targeted=result[True]["BBA-prior2"][2000],
        BBA_prior2_5000_targeted=result[True]["BBA-prior2"][5000],
        BBA_prior2_8000_targeted=result[True]["BBA-prior2"][8000],
        BBA_prior2_10000_targeted=result[True]["BBA-prior2"][10000],
        BBA_prior2_15000_targeted=result[True]["BBA-prior2"][15000],
        BBA_prior2_20000_targeted=result[True]["BBA-prior2"][20000],

        HSJA_1000_targeted=result[True]["HSJA"][1000],
        HSJA_2000_targeted=result[True]["HSJA"][2000],
        HSJA_5000_targeted=result[True]["HSJA"][5000],
        HSJA_8000_targeted=result[True]["HSJA"][8000],
        HSJA_10000_targeted=result[True]["HSJA"][10000],
        HSJA_15000_targeted=result[True]["HSJA"][15000],
        HSJA_20000_targeted=result[True]["HSJA"][20000],

        TA_1000_targeted=result[True]["TA"][1000],
        TA_2000_targeted=result[True]["TA"][2000],
        TA_5000_targeted=result[True]["TA"][5000],
        TA_8000_targeted=result[True]["TA"][8000],
        TA_10000_targeted=result[True]["TA"][10000],
        TA_15000_targeted=result[True]["TA"][15000],
        TA_20000_targeted=result[True]["TA"][20000],

        GTA_1000_targeted=result[True]["G-TA"][1000],
        GTA_2000_targeted=result[True]["G-TA"][2000],
        GTA_5000_targeted=result[True]["G-TA"][5000],
        GTA_8000_targeted=result[True]["G-TA"][8000],
        GTA_10000_targeted=result[True]["G-TA"][10000],
        GTA_15000_targeted=result[True]["G-TA"][15000],
        GTA_20000_targeted=result[True]["G-TA"][20000],

        QEBA_1000_targeted=result[True]["QEBA"][1000],
        QEBA_2000_targeted=result[True]["QEBA"][2000],
        QEBA_5000_targeted=result[True]["QEBA"][5000],
        QEBA_8000_targeted=result[True]["QEBA"][8000],
        QEBA_10000_targeted=result[True]["QEBA"][10000],
        QEBA_15000_targeted=result[True]["QEBA"][15000],
        QEBA_20000_targeted=result[True]["QEBA"][20000],

        SignOPT_1000_targeted=result[True]["Sign-OPT"][1000],
        SignOPT_2000_targeted=result[True]["Sign-OPT"][2000],
        SignOPT_5000_targeted=result[True]["Sign-OPT"][5000],
        SignOPT_8000_targeted=result[True]["Sign-OPT"][8000],
        SignOPT_10000_targeted=result[True]["Sign-OPT"][10000],
        SignOPT_15000_targeted=result[True]["Sign-OPT"][15000],
        SignOPT_20000_targeted=result[True]["Sign-OPT"][20000],

        SVMOPT_1000_targeted=result[True]["SVM-OPT"][1000],
        SVMOPT_2000_targeted=result[True]["SVM-OPT"][2000],
        SVMOPT_5000_targeted=result[True]["SVM-OPT"][5000],
        SVMOPT_8000_targeted=result[True]["SVM-OPT"][8000],
        SVMOPT_10000_targeted=result[True]["SVM-OPT"][10000],
        SVMOPT_15000_targeted=result[True]["SVM-OPT"][15000],
        SVMOPT_20000_targeted=result[True]["SVM-OPT"][20000],

        GeoDA_1000_targeted=result[True]["GeoDA"][1000],
        GeoDA_2000_targeted=result[True]["GeoDA"][2000],
        GeoDA_5000_targeted=result[True]["GeoDA"][5000],
        GeoDA_8000_targeted=result[True]["GeoDA"][8000],
        GeoDA_10000_targeted=result[True]["GeoDA"][10000],
        GeoDA_15000_targeted=result[True]["GeoDA"][15000],
        GeoDA_20000_targeted=result[True]["GeoDA"][20000],

        Evolutionary_1000_targeted=result[True]["Evolutionary"][1000],
        Evolutionary_2000_targeted=result[True]["Evolutionary"][2000],
        Evolutionary_5000_targeted=result[True]["Evolutionary"][5000],
        Evolutionary_8000_targeted=result[True]["Evolutionary"][8000],
        Evolutionary_10000_targeted=result[True]["Evolutionary"][10000],
        Evolutionary_15000_targeted=result[True]["Evolutionary"][15000],
        Evolutionary_20000_targeted=result[True]["Evolutionary"][20000],

        SurFree_1000_targeted=result[True]["SurFree"][1000],
        SurFree_2000_targeted=result[True]["SurFree"][2000],
        SurFree_5000_targeted=result[True]["SurFree"][5000],
        SurFree_8000_targeted=result[True]["SurFree"][8000],
        SurFree_10000_targeted=result[True]["SurFree"][10000],
        SurFree_15000_targeted=result[True]["SurFree"][15000],
        SurFree_20000_targeted=result[True]["SurFree"][20000],

        TriangleAttack_1000_targeted=result[True]["Triangle Attack"][1000],
        TriangleAttack_2000_targeted=result[True]["Triangle Attack"][2000],
        TriangleAttack_5000_targeted=result[True]["Triangle Attack"][5000],
        TriangleAttack_8000_targeted=result[True]["Triangle Attack"][8000],
        TriangleAttack_10000_targeted=result[True]["Triangle Attack"][10000],
        TriangleAttack_15000_targeted=result[True]["Triangle Attack"][15000],
        TriangleAttack_20000_targeted=result[True]["Triangle Attack"][20000],

        CGBA_1000_targeted=result[True]["CGBA_H"][1000],
        CGBA_2000_targeted=result[True]["CGBA_H"][2000],
        CGBA_5000_targeted=result[True]["CGBA_H"][5000],
        CGBA_8000_targeted=result[True]["CGBA_H"][8000],
        CGBA_10000_targeted=result[True]["CGBA_H"][10000],
        CGBA_15000_targeted=result[True]["CGBA_H"][15000],
        CGBA_20000_targeted=result[True]["CGBA_H"][20000],

        SQBA_prior1_1000_targeted=result[True]["SQBA-prior1"][1000],
        SQBA_prior1_2000_targeted=result[True]["SQBA-prior1"][2000],
        SQBA_prior1_5000_targeted=result[True]["SQBA-prior1"][5000],
        SQBA_prior1_8000_targeted=result[True]["SQBA-prior1"][8000],
        SQBA_prior1_10000_targeted=result[True]["SQBA-prior1"][10000],
        SQBA_prior1_15000_targeted=result[True]["SQBA-prior1"][15000],
        SQBA_prior1_20000_targeted=result[True]["SQBA-prior1"][20000],

        SQBA_prior2_1000_targeted=result[True]["SQBA-prior2"][1000],
        SQBA_prior2_2000_targeted=result[True]["SQBA-prior2"][2000],
        SQBA_prior2_5000_targeted=result[True]["SQBA-prior2"][5000],
        SQBA_prior2_8000_targeted=result[True]["SQBA-prior2"][8000],
        SQBA_prior2_10000_targeted=result[True]["SQBA-prior2"][10000],
        SQBA_prior2_15000_targeted=result[True]["SQBA-prior2"][15000],
        SQBA_prior2_20000_targeted=result[True]["SQBA-prior2"][20000],

        PriorSignOPT_prior1_1000_targeted=result[True]["Prior-Sign-OPT-prior1"][1000],
        PriorSignOPT_prior1_2000_targeted=result[True]["Prior-Sign-OPT-prior1"][2000],
        PriorSignOPT_prior1_5000_targeted=result[True]["Prior-Sign-OPT-prior1"][5000],
        PriorSignOPT_prior1_8000_targeted=result[True]["Prior-Sign-OPT-prior1"][8000],
        PriorSignOPT_prior1_10000_targeted=result[True]["Prior-Sign-OPT-prior1"][10000],
        PriorSignOPT_prior1_15000_targeted=result[True]["Prior-Sign-OPT-prior1"][15000],
        PriorSignOPT_prior1_20000_targeted=result[True]["Prior-Sign-OPT-prior1"][20000],

        PriorSignOPT_2priors_1000_targeted=result[True]["Prior-Sign-OPT-2priors"][
            1000],
        PriorSignOPT_2priors_2000_targeted=result[True]["Prior-Sign-OPT-2priors"][
            2000],
        PriorSignOPT_2priors_5000_targeted=result[True]["Prior-Sign-OPT-2priors"][
            5000],
        PriorSignOPT_2priors_8000_targeted=result[True]["Prior-Sign-OPT-2priors"][
            8000],
        PriorSignOPT_2priors_10000_targeted=result[True]["Prior-Sign-OPT-2priors"][
            10000],
        PriorSignOPT_2priors_15000_targeted=result[True]["Prior-Sign-OPT-2priors"][
            15000],
        PriorSignOPT_2priors_20000_targeted=result[True]["Prior-Sign-OPT-2priors"][
            20000],

        PriorSignOPT_PGD_init_theta_1000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            1000],
        PriorSignOPT_PGD_init_theta_2000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            2000],
        PriorSignOPT_PGD_init_theta_5000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            5000],
        PriorSignOPT_PGD_init_theta_8000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            8000],
        PriorSignOPT_PGD_init_theta_10000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            10000],
        PriorSignOPT_PGD_init_theta_15000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            15000],
        PriorSignOPT_PGD_init_theta_20000_targeted=result[True]["Prior-Sign-OPT_PGD_init_theta"][
            20000],

        PriorOPT_prior1_1000_targeted=result[True]["Prior-OPT-prior1"][1000],
        PriorOPT_prior1_2000_targeted=result[True]["Prior-OPT-prior1"][2000],
        PriorOPT_prior1_5000_targeted=result[True]["Prior-OPT-prior1"][5000],
        PriorOPT_prior1_8000_targeted=result[True]["Prior-OPT-prior1"][8000],
        PriorOPT_prior1_10000_targeted=result[True]["Prior-OPT-prior1"][10000],
        PriorOPT_prior1_15000_targeted=result[True]["Prior-OPT-prior1"][15000],
        PriorOPT_prior1_20000_targeted=result[True]["Prior-OPT-prior1"][20000],

        PriorOPT_2priors_1000_targeted=result[True]["Prior-OPT-2priors"][1000],
        PriorOPT_2priors_2000_targeted=result[True]["Prior-OPT-2priors"][2000],
        PriorOPT_2priors_5000_targeted=result[True]["Prior-OPT-2priors"][5000],
        PriorOPT_2priors_8000_targeted=result[True]["Prior-OPT-2priors"][8000],
        PriorOPT_2priors_10000_targeted=result[True]["Prior-OPT-2priors"][10000],
        PriorOPT_2priors_15000_targeted=result[True]["Prior-OPT-2priors"][15000],
        PriorOPT_2priors_20000_targeted=result[True]["Prior-OPT-2priors"][20000],

        PriorOPT_PGD_init_theta_1000_targeted=result[True]["Prior-OPT_PGD_init_theta"][1000],
        PriorOPT_PGD_init_theta_2000_targeted=result[True]["Prior-OPT_PGD_init_theta"][2000],
        PriorOPT_PGD_init_theta_5000_targeted=result[True]["Prior-OPT_PGD_init_theta"][5000],
        PriorOPT_PGD_init_theta_8000_targeted=result[True]["Prior-OPT_PGD_init_theta"][8000],
        PriorOPT_PGD_init_theta_10000_targeted=result[True]["Prior-OPT_PGD_init_theta"][10000],
        PriorOPT_PGD_init_theta_15000_targeted=result[True]["Prior-OPT_PGD_init_theta"][15000],
        PriorOPT_PGD_init_theta_20000_targeted=result[True]["Prior-OPT_PGD_init_theta"][20000],
    )
    )


if __name__ == "__main__":
    dataset = "ImageNet"
    norm = "l2"
    arch = "gcvit_base"
    query_budgets = [1000, 2000, 5000, 8000, 10000,12000,15000,18000,20000]
    if dataset == "CIFAR-10":
        dim = 32 * 32 * 3
    else:
        if "inception" in arch:
            dim = 299 * 299 * 3
        else:
            dim = 224 * 224 * 3
    success_distortion_threshold = math.sqrt(0.001 * dim)
    print(success_distortion_threshold)
    result_distortion, result_ASR = fetch_json_content_for_multiple_priors(dataset, norm, True, arch, query_budgets, success_distortion_threshold,"mean_distortion")
    draw_tables(result_distortion, "ResNet50", "ConViT", "CrossViT", "MaxViT", "ViT")
    draw_tables(result_ASR, "ResNet50", "ConViT", "CrossViT", "MaxViT", "ViT")