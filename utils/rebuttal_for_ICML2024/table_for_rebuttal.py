from collections import defaultdict

import bisect
import numpy as np
import json
import os

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


# method_name_to_paper = {"tangent_attack":"TA",
#                         "ellipsoid_tangent_attack":"G-TA", "GeoDA":"GeoDA",
#                         "HSJA":"HSJA",  "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
#                          "Evolutionary":"Evolutionary", "SurFree":"SurFree",
#                         "TriangleAttack":"Triangle Attack", "PriorSignOPT":"Prior-Sign-OPT",
#                         "PriorOPT":"Prior-OPT","biased_boundary_attack":"BBA",
#                         "SQBA":"SQBA", "PDA":"PDA"
#                         }
method_name_to_paper = {
                        "HSJA":"HSJA",  "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                        "PriorSignOPT":"Prior-Sign-OPT","CGBA_H":"CGBA-H","AHA":"AHA",
                        "PriorOPT":"Prior-OPT",
                       "tangent_attack":"TA",  "ellipsoid_tangent_attack":"G-TA",  "GeoDA":"GeoDA",  "Evolutionary":"Evolutionary", "SurFree":"SurFree","TriangleAttack":"Triangle Attack",
                        "PriorOPT_theta0":"PriorOPT_theta0",
                        "SQBA": "SQBA","biased_boundary_attack":"BBA"}
#                         }


def from_defense_method_to_dir_path(dataset, method, norm, targeted):
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
    elif method == "CGBA_H":
        if targeted:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "AHA":
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
    elif method == "PriorOPT_theta0":
        path = "{method}-{dataset}-{norm}-{target_str}_with_PGD_init_theta".format(method=method[:method.index("_")], dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PriorSignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "QEBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SQBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    return path


def read_json_and_extract(json_path):
    with (open(json_path, "r")) as file_obj:
        print("reading :{}".format(json_path))
        json_content = json.load(file_obj)
        distortion = defaultdict(dict)
        for img_id, query_distortion_dict in json_content["distortion"].items():
            distortion[int(img_id)] = query_distortion_dict
        return distortion, json_content

def read_defense_json_and_extract(json_path):
    with (open(json_path, "r")) as file_obj:
        json_content = json.load(file_obj)
        return json_content
def get_file_name_list(dataset, method_name_to_paper, norm, defense, targeted):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        if not defense:
            file_path = "F:/logs/hard_label_attack_complete/" + from_method_to_dir_path(dataset, method, norm, targeted)
        else:
            file_path = "F:/logs/hard_label_attack_complete/" + from_defense_method_to_dir_path(dataset, method, norm, targeted)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets, want_key):
    mean_and_median_distortions = {}
    for query_budget in query_budgets:
        distortion_list = []
        for image_index, query_distortion in distortion_dict.items():
            query_distortion = {float(query):float(dist) for query,dist in query_distortion.items()}
            queries = list(query_distortion.keys())
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
        if want_key == "mean_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(mean_distortion.item(),3))
        elif want_key =="median_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(median_distortion.item(),3))
    return mean_and_median_distortions


def fetch_all_json_content_given_contraint(dataset, norm, arch, targeted, query_budgets, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm,False, targeted)
    result = {}

    for method, folder in folder_list.items():
        for file_name in os.listdir(folder):
            new_method = None
            if file_name.startswith(arch) and file_name.endswith(".json"):
                file_path = folder + "/"+file_name
                if targeted:
                    if method in ["Triangle Attack", "RayS", "GeoDA"]:
                        result[method] = defaultdict(lambda: "-")
                        continue
                if norm == "l2":
                    if "RayS" == method:
                        result[method] = defaultdict(lambda: "-")
                        continue
                elif norm == "linf":
                    if method in ["Evolutionary","SurFree","Triangle Attack","RayS"]:
                        result[method] = defaultdict(lambda: "-")
                        continue
                if method in ["Prior-Sign-OPT","Prior-OPT"] and "surrogates_" in file_name:  # FIXME
                    continue
                if method == "SQBA" and "surrogate_inceptionresnetv2_result.json" not in file_name:  # FIXME
                    continue

                print(file_path)
                distortion_dict, json_content = read_json_and_extract(file_path)
                if not method.endswith("theta0"):
                    if "surrogate_arch" in json_content["args"] or "surrogate_archs" in json_content["args"]:
                        if json_content["args"]["surrogate_arch"] is not None:
                            new_method = method+"1"
                            if "surrogate_archs" in json_content["args"]:
                                assert json_content["args"]["surrogate_archs"] is None
                        elif "surrogate_archs" in json_content["args"] and json_content["args"]["surrogate_archs"] is not None:
                            new_method = method + str(len(json_content["args"]["surrogate_archs"]))
                mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets,want_key)
                if new_method:
                    result[new_method] = mean_and_median_distortions
                else:
                    result[method] = mean_and_median_distortions
    return result


def fetch_defense_json_content_given_contraint(dataset, norm, targeted, query_budgets, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm,True, targeted)
    result = {}
    for method, folder in folder_list.items():
        for file_name in os.listdir(folder):
            new_method = None
            if file_name.startswith("resnet-50") and file_name.endswith("adv_train_result.json"):
                file_path = folder + "/"+file_name
                if targeted:
                    if method in ["Triangle Attack", "RayS", "GeoDA"]:
                        result[method] = defaultdict(lambda: "-")
                        continue
                if norm == "l2":
                    if "RayS" == method:
                        result[method] = defaultdict(lambda: "-")
                        continue
                elif norm == "linf":
                    if method in ["TA","G-TA","Evolutionary",
                                  "SurFree","Triangle Attack","RayS"]:
                        result[method] = defaultdict(lambda: "-")
                        continue
                json_content = read_defense_json_and_extract(file_path)
                if "theta0" not in method:
                    if "surrogate_defense_models" in json_content["args"]:
                        if "surrogate_archs" in json_content["args"]:  # Prior-Sign-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)
                            new_method = method +"_" + "&".join([defense_model+"("+json_content["args"]["surrogate_archs"][idx]+")"
                                                            for idx, defense_model in enumerate(json_content["args"]["surrogate_defense_models"])])
                    elif "surrogate_arch" in json_content["args"]:
                        if json_content["args"]["surrogate_arch"] is not None:
                            new_method = method + "_"+json_content["args"]["surrogate_arch"]
                            assert json_content["args"]["surrogate_archs"] is None
                        elif json_content["args"]["surrogate_archs"] is not None: # Prior-Sign-OPT_resnet-110&densenet-bc-100-12
                            new_method = method +"_" + "&".join(json_content["args"]["surrogate_archs"])
                print(file_path)
                mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(json_content["distortion"], query_budgets,want_key)
                if new_method:
                    result[new_method] = mean_and_median_distortions
                else:
                    result[method] = mean_and_median_distortions
    return result

def draw_table(result):
    print(r"""
            | Method | @1K | @3K | @5K | @7K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- | :- |
            | HSJA | {HSJA_1K} |{HSJA_3K} | {HSJA_5K} | {HSJA_7K} | {HSJA_8K} | {HSJA_10K} |
            | TA | {TA_1K} |{TA_3K} | {TA_5K} | {TA_7K} | {TA_8K} | {TA_10K} |
            | G-TA | {GTA_1K} |{GTA_3K} | {GTA_5K} | {GTA_7K} | {GTA_8K} | {GTA_10K} |
            | GeoDA | {GeoDA_1K} |{GeoDA_3K} | {GeoDA_5K} | {GeoDA_7K} | {GeoDA_8K} | {GeoDA_10K} |
            | Evolutionary | {Evolutionary_1K} |{Evolutionary_3K} | {Evolutionary_5K} | {Evolutionary_7K} | {Evolutionary_8K} | {Evolutionary_10K} |
            | SurFree | {SurFree_1K} |{SurFree_3K} | {SurFree_5K} | {SurFree_7K} | {SurFree_8K} | {SurFree_10K} |
            | Triangle Attack | {TriA_1K} |{TriA_3K} | {TriA_5K} | {TriA_7K} | {TriA_8K} | {TriA_10K} |
            | SVM-OPT | {SVMOPT_1K} |{SVMOPT_3K} | {SVMOPT_5K} | {SVMOPT_7K} | {SVMOPT_8K} | {SVMOPT_10K} |
            | Sign-OPT  | {SignOPT_1K} |{SignOPT_3K} | {SignOPT_5K} | {SignOPT_7K} | {SignOPT_8K} | {SignOPT_10K} |
            | $\text{{SQBA}}_\text{{IncResV2}}$ | {SQBA_1K} |{SQBA_3K} | {SQBA_5K} | {SQBA_7K} | {SQBA_8K} | {SQBA_10K} |
            | $\text{{BBA}}_\text{{IncResV2}}$ | {BBA_1K} |{BBA_3K} | {BBA_5K} | {BBA_7K} | {BBA_8K} | {BBA_10K} |
            | CGBA-H | {CGBA_1K} | {CGBA_3K} | {CGBA_5K} | {CGBA_7K} | {CGBA_8K} | {CGBA_10K} |
            | AHA | {AHA_1K} | {AHA_3K} | {AHA_5K} | {AHA_7K} | {AHA_8K} | {AHA_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{IncResV2}}$ | {PriorSignOPT1_1K} | {PriorSignOPT1_3K} | {PriorSignOPT1_5K} | {PriorSignOPT1_7K} | {PriorSignOPT1_8K} | {PriorSignOPT1_10K} |
            | $\text{{Prior-OPT}}_\text{{IncResV2}}$ | {PriorOPT1_1K} | {PriorOPT1_3K} | {PriorOPT1_5K} |  {PriorOPT1_7K} | {PriorOPT1_8K} | {PriorOPT1_10K} |
           | $\text{{Prior-OPT}} _ {{\theta _ 0^\text{{PGD}} + \text{{IncResV2}}$ | {PriorOPT_theta0_1K} | {PriorOPT_theta0_3K} | {PriorOPT_theta0_5K} |  {PriorOPT_theta0_7K} | {PriorOPT_theta0_8K} | {PriorOPT_theta0_10K} |
                                """.format(
        HSJA_1K=result["HSJA"][1000],
        HSJA_3K=result["HSJA"][3000],
        HSJA_5K=result["HSJA"][5000],
        HSJA_7K=result["HSJA"][7000],
        HSJA_8K=result["HSJA"][8000],
        HSJA_10K=result["HSJA"][10000],

        TA_1K=result["TA"][1000],
        TA_3K=result["TA"][3000],
        TA_5K=result["TA"][5000],
        TA_7K=result["TA"][7000],
        TA_8K=result["TA"][8000],
        TA_10K=result["TA"][10000],

        GTA_1K=result["G-TA"][1000],
        GTA_3K=result["G-TA"][3000],
        GTA_5K=result["G-TA"][5000],
        GTA_7K=result["G-TA"][7000],
        GTA_8K=result["G-TA"][8000],
        GTA_10K=result["G-TA"][10000],

        GeoDA_1K=result["GeoDA"][1000],
        GeoDA_3K=result["GeoDA"][3000],
        GeoDA_5K=result["GeoDA"][5000],
        GeoDA_7K=result["GeoDA"][7000],
        GeoDA_8K=result["GeoDA"][8000],
        GeoDA_10K=result["GeoDA"][10000],

        Evolutionary_1K=result["Evolutionary"][1000],
        Evolutionary_3K=result["Evolutionary"][3000],
        Evolutionary_5K=result["Evolutionary"][5000],
        Evolutionary_7K=result["Evolutionary"][7000],
        Evolutionary_8K=result["Evolutionary"][8000],
        Evolutionary_10K=result["Evolutionary"][10000],

        SurFree_1K=result["SurFree"][1000],
        SurFree_3K=result["SurFree"][3000],
        SurFree_5K=result["SurFree"][5000],
        SurFree_7K=result["SurFree"][7000],
        SurFree_8K=result["SurFree"][8000],
        SurFree_10K=result["SurFree"][10000],

        TriA_1K=result["Triangle Attack"][1000],
        TriA_3K=result["Triangle Attack"][3000],
        TriA_5K=result["Triangle Attack"][5000],
        TriA_7K=result["Triangle Attack"][7000],
        TriA_8K=result["Triangle Attack"][8000],
        TriA_10K=result["Triangle Attack"][10000],


        SQBA_1K=result["SQBA1"][1000],
        SQBA_3K=result["SQBA1"][3000],
        SQBA_5K=result["SQBA1"][5000],
        SQBA_7K=result["SQBA1"][7000],
        SQBA_8K=result["SQBA1"][8000],
        SQBA_10K=result["SQBA1"][10000],

        BBA_1K=result["BBA1"][1000],
        BBA_3K=result["BBA1"][3000],
        BBA_5K=result["BBA1"][5000],
        BBA_7K=result["BBA1"][7000],
        BBA_8K=result["BBA1"][8000],
        BBA_10K=result["BBA1"][10000],

        SignOPT_1K=result["Sign-OPT"][1000],
        SignOPT_3K=result["Sign-OPT"][3000],
        SignOPT_5K=result["Sign-OPT"][5000],
        SignOPT_7K=result["Sign-OPT"][7000],
        SignOPT_8K=result["Sign-OPT"][8000],
        SignOPT_10K=result["Sign-OPT"][10000],

        SVMOPT_1K=result["SVM-OPT"][1000],
        SVMOPT_3K=result["SVM-OPT"][3000],
        SVMOPT_5K=result["SVM-OPT"][5000],
        SVMOPT_7K=result["SVM-OPT"][7000],
        SVMOPT_8K=result["SVM-OPT"][8000],
        SVMOPT_10K=result["SVM-OPT"][10000],

        CGBA_1K=result["CGBA-H"][1000],
        CGBA_3K=result["CGBA-H"][3000],
        CGBA_5K=result["CGBA-H"][5000],
        CGBA_7K=result["CGBA-H"][7000],
        CGBA_8K=result["CGBA-H"][8000],
        CGBA_10K=result["CGBA-H"][10000],

        AHA_1K=result["AHA"][1000],
        AHA_3K=result["AHA"][3000],
        AHA_5K=result["AHA"][5000],
        AHA_7K=result["AHA"][7000],
        AHA_8K=result["AHA"][8000],
        AHA_10K=result["AHA"][10000],

        PriorSignOPT1_1K=result["Prior-Sign-OPT1"][1000],
        PriorSignOPT1_3K=result["Prior-Sign-OPT1"][3000],
        PriorSignOPT1_5K=result["Prior-Sign-OPT1"][5000],
        PriorSignOPT1_7K=result["Prior-Sign-OPT1"][7000],
        PriorSignOPT1_8K=result["Prior-Sign-OPT1"][8000],
        PriorSignOPT1_10K=result["Prior-Sign-OPT1"][10000],


        PriorOPT1_1K=result["Prior-OPT1"][1000],
        PriorOPT1_3K=result["Prior-OPT1"][3000],
        PriorOPT1_5K=result["Prior-OPT1"][5000],
        PriorOPT1_7K=result["Prior-OPT1"][7000],
        PriorOPT1_8K=result["Prior-OPT1"][8000],
        PriorOPT1_10K=result["Prior-OPT1"][10000],

        PriorOPT_theta0_1K=result["PriorOPT_theta0"][1000],
        PriorOPT_theta0_3K=result["PriorOPT_theta0"][3000],
        PriorOPT_theta0_5K=result["PriorOPT_theta0"][5000],
        PriorOPT_theta0_7K=result["PriorOPT_theta0"][7000],
        PriorOPT_theta0_8K=result["PriorOPT_theta0"][8000],
        PriorOPT_theta0_10K=result["PriorOPT_theta0"][10000],
    )
    )
def draw_table_for_comparing_transfer_methods(result):
    print(r"""
            | Method | @1K | @3K | @5K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- |
            | HSJA | {HSJA_1K} | {HSJA_3K} | {HSJA_5K} | {HSJA_8K} | {HSJA_10K} |
            | SVM-OPT | {SVMOPT_1K} | {SVMOPT_3K} | {SVMOPT_5K} | {SVMOPT_8K} | {SVMOPT_10K} |
            | Sign-OPT | {SignOPT_1K} | {SignOPT_3K} | {SignOPT_5K} | {SignOPT_8K} | {SignOPT_10K} |
            | CGBA-H | {CGBA_1K} | {CGBA_3K} | {CGBA_5K} | {CGBA_8K} | {CGBA_10K} |
            | AHA | {AHA_1K} | {AHA_3K} | {AHA_5K} | {AHA_8K} | {AHA_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50}}$ | {PriorSignOPT1_1K} | {PriorSignOPT1_3K} | {PriorSignOPT1_5K} | {PriorSignOPT1_8K} | {PriorSignOPT1_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50}}$ | {PriorOPT1_1K} | {PriorOPT1_3K} | {PriorOPT1_5K} | {PriorOPT1_8K} | {PriorOPT1_10K} |
                                """.format(
        HSJA_1K=result["HSJA"][1000],
        HSJA_3K=result["HSJA"][3000],
        HSJA_5K=result["HSJA"][5000],
        HSJA_8K=result["HSJA"][8000],
        HSJA_10K=result["HSJA"][10000],

        SignOPT_1K=result["Sign-OPT"][1000],
        SignOPT_3K=result["Sign-OPT"][3000],
        SignOPT_5K=result["Sign-OPT"][5000],
        SignOPT_8K=result["Sign-OPT"][8000],
        SignOPT_10K=result["Sign-OPT"][10000],

        SVMOPT_1K=result["SVM-OPT"][1000],
        SVMOPT_3K=result["SVM-OPT"][3000],
        SVMOPT_5K=result["SVM-OPT"][5000],
        SVMOPT_8K=result["SVM-OPT"][8000],
        SVMOPT_10K=result["SVM-OPT"][10000],

        CGBA_1K=result["CGBA-H"][1000],
        CGBA_3K=result["CGBA-H"][3000],
        CGBA_5K=result["CGBA-H"][5000],
        CGBA_8K=result["CGBA-H"][8000],
        CGBA_10K=result["CGBA-H"][10000],

        AHA_1K=result["AHA"][1000],
        AHA_3K=result["AHA"][3000],
        AHA_5K=result["AHA"][5000],
        AHA_8K=result["AHA"][8000],
        AHA_10K=result["AHA"][10000],

        PriorSignOPT1_1K=result["Prior-Sign-OPT1"][1000],
        PriorSignOPT1_3K=result["Prior-Sign-OPT1"][3000],
        PriorSignOPT1_5K=result["Prior-Sign-OPT1"][5000],
        PriorSignOPT1_8K=result["Prior-Sign-OPT1"][8000],
        PriorSignOPT1_10K=result["Prior-Sign-OPT1"][10000],


        PriorOPT1_1K=result["Prior-OPT1"][1000],
        PriorOPT1_3K=result["Prior-OPT1"][3000],
        PriorOPT1_5K=result["Prior-OPT1"][5000],
        PriorOPT1_8K=result["Prior-OPT1"][8000],
        PriorOPT1_10K=result["Prior-OPT1"][10000],

    )
    )


def draw_table_for_more_priors_result(result):
    print(r"""
            | Method | @1K | @3K | @5K |  @8K |  @10K |
            | :- | :- |:- | :- | :- | :- |
            | HSJA | {HSJA_1K} | {HSJA_3K} | {HSJA_5K} | {HSJA_8K} | {HSJA_10K} |
            | Sign-OPT | {SignOPT_1K} | {SignOPT_3K} | {SignOPT_5K} | {SignOPT_8K} | {SignOPT_10K} |
            | SVM-OPT | {SVMOPT_1K} | {SVMOPT_3K} | {SVMOPT_5K} | {SVMOPT_8K} | {SVMOPT_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50}}$ | {PriorSignOPT1_1K} | {PriorSignOPT1_3K} | {PriorSignOPT1_5K} | {PriorSignOPT1_8K} | {PriorSignOPT1_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154}}$  | {PriorSignOPT2_1K} | {PriorSignOPT2_3K} | {PriorSignOPT2_5K} | {PriorSignOPT2_8K} | {PriorSignOPT2_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101}}$  | {PriorSignOPT3_1K} | {PriorSignOPT3_3K} | {PriorSignOPT3_5K} | {PriorSignOPT3_8K} | {PriorSignOPT3_10K} |
            | $\text{{Prior-Sign-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13}}$   | {PriorSignOPT4_1K} | {PriorSignOPT4_3K} | {PriorSignOPT4_5K} | {PriorSignOPT4_8K} | {PriorSignOPT4_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50}}$ | {PriorOPT1_1K} | {PriorOPT1_3K} | {PriorOPT1_5K} | {PriorOPT1_8K} | {PriorOPT1_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50\\&SENet154}}$  | {PriorOPT2_1K} | {PriorOPT2_3K} | {PriorOPT2_5K} | {PriorOPT2_8K} | {PriorOPT2_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101}}$ | {PriorOPT3_1K} | {PriorOPT3_3K} | {PriorOPT3_5K} | {PriorOPT3_8K} | {PriorOPT3_10K} |
            | $\text{{Prior-OPT}}_\text{{ResNet50\\&SENet154\\&ResNeXt101\\&VGG13}}$  | {PriorOPT4_1K} | {PriorOPT4_3K} | {PriorOPT4_5K} | {PriorOPT4_8K} | {PriorOPT4_10K} |
                                """.format(
        HSJA_1K=result["HSJA"][1000],
        HSJA_3K=result["HSJA"][3000],
        HSJA_5K=result["HSJA"][5000],
        HSJA_8K=result["HSJA"][8000],
        HSJA_10K=result["HSJA"][10000],

        SignOPT_1K=result["Sign-OPT"][1000],
        SignOPT_3K=result["Sign-OPT"][3000],
        SignOPT_5K=result["Sign-OPT"][5000],
        SignOPT_8K=result["Sign-OPT"][8000],
        SignOPT_10K=result["Sign-OPT"][10000],

        SVMOPT_1K=result["SVM-OPT"][1000],
        SVMOPT_3K=result["SVM-OPT"][3000],
        SVMOPT_5K=result["SVM-OPT"][5000],
        SVMOPT_8K=result["SVM-OPT"][8000],
        SVMOPT_10K=result["SVM-OPT"][10000],

        PriorSignOPT1_1K=result["Prior-Sign-OPT1"][1000],
        PriorSignOPT1_3K=result["Prior-Sign-OPT1"][3000],
        PriorSignOPT1_5K=result["Prior-Sign-OPT1"][5000],
        PriorSignOPT1_8K=result["Prior-Sign-OPT1"][8000],
        PriorSignOPT1_10K=result["Prior-Sign-OPT1"][10000],

        PriorSignOPT2_1K=result["Prior-Sign-OPT2"][1000],
        PriorSignOPT2_3K=result["Prior-Sign-OPT2"][3000],
        PriorSignOPT2_5K=result["Prior-Sign-OPT2"][5000],
        PriorSignOPT2_8K=result["Prior-Sign-OPT2"][8000],
        PriorSignOPT2_10K=result["Prior-Sign-OPT2"][10000],

        PriorSignOPT3_1K=result["Prior-Sign-OPT3"][1000],
        PriorSignOPT3_3K=result["Prior-Sign-OPT3"][3000],
        PriorSignOPT3_5K=result["Prior-Sign-OPT3"][5000],
        PriorSignOPT3_8K=result["Prior-Sign-OPT3"][8000],
        PriorSignOPT3_10K=result["Prior-Sign-OPT3"][10000],

        PriorSignOPT4_1K=result["Prior-Sign-OPT4"][1000],
        PriorSignOPT4_3K=result["Prior-Sign-OPT4"][3000],
        PriorSignOPT4_5K=result["Prior-Sign-OPT4"][5000],
        PriorSignOPT4_8K=result["Prior-Sign-OPT4"][8000],
        PriorSignOPT4_10K=result["Prior-Sign-OPT4"][10000],

        PriorOPT1_1K=result["Prior-OPT1"][1000],
        PriorOPT1_3K=result["Prior-OPT1"][3000],
        PriorOPT1_5K=result["Prior-OPT1"][5000],
        PriorOPT1_8K=result["Prior-OPT1"][8000],
        PriorOPT1_10K=result["Prior-OPT1"][10000],

        PriorOPT2_1K=result["Prior-OPT2"][1000],
        PriorOPT2_3K=result["Prior-OPT2"][3000],
        PriorOPT2_5K=result["Prior-OPT2"][5000],
        PriorOPT2_8K=result["Prior-OPT2"][8000],
        PriorOPT2_10K=result["Prior-OPT2"][10000],

        PriorOPT3_1K=result["Prior-OPT3"][1000],
        PriorOPT3_3K=result["Prior-OPT3"][3000],
        PriorOPT3_5K=result["Prior-OPT3"][5000],
        PriorOPT3_8K=result["Prior-OPT3"][8000],
        PriorOPT3_10K=result["Prior-OPT3"][10000],

        PriorOPT4_1K=result["Prior-OPT4"][1000],
        PriorOPT4_3K=result["Prior-OPT4"][3000],
        PriorOPT4_5K=result["Prior-OPT4"][5000],
        PriorOPT4_8K=result["Prior-OPT4"][8000],
        PriorOPT4_10K=result["Prior-OPT4"][10000],
    )
    )


def draw_table_for_defensive_models(result):
    print(r"""
| Method () | @1K | @3K | @5K |  @8K |  @10K |
| :- | :- |:- | :- | :- | :- |
| HSJA | {HSJA_1K} | {HSJA_3K} | {HSJA_5K} | {HSJA_8K} | {HSJA_10K} |
| Sign-OPT | {SignOPT_1K} | {SignOPT_3K} | {SignOPT_5K} | {SignOPT_8K} | {SignOPT_10K} |
| SVM-OPT | {SVMOPT_1K} | {SVMOPT_3K} | {SVMOPT_5K} | {SVMOPT_8K} | {SVMOPT_10K} |
| $\text{{Prior-Sign-OPT}}_\text{{ResNet110}}$ | {PriorSignOPT_ResNet110_1K} | {PriorSignOPT_ResNet110_3K} | {PriorSignOPT_ResNet110_5K} | {PriorSignOPT_ResNet110_8K} | {PriorSignOPT_ResNet110_10K} |
| $\text{{Prior-Sign-OPT}}_\text{{ResNet110\\&DenseNetBC110}}$ | {PriorSignOPT_ResNet110_DenseNetBC110_1K} | {PriorSignOPT_ResNet110_DenseNetBC110_3K} | {PriorSignOPT_ResNet110_DenseNetBC110_5K} | {PriorSignOPT_ResNet110_DenseNetBC110_8K} | {PriorSignOPT_ResNet110_DenseNetBC110_10K} |
| $\text{{Prior-Sign-OPT}}_\text{{AT(ResNet110)}}$ | {PriorSignOPT_ResNet110AT_1K} | {PriorSignOPT_ResNet110AT_3K} | {PriorSignOPT_ResNet110AT_5K} | {PriorSignOPT_ResNet110AT_8K} | {PriorSignOPT_ResNet110AT_10K} |
| $\text{{Prior-Sign-OPT}}_\text{{AT(VGG13BN)}}$ | {PriorSignOPT_VGG13AT_1K} | {PriorSignOPT_VGG13AT_3K} | {PriorSignOPT_VGG13AT_5K} | {PriorSignOPT_VGG13AT_8K} | {PriorSignOPT_VGG13AT_10K} |
| $\text{{Prior-Sign-OPT}}_\text{{AT(ResNet110)\\&AT(VGG13BN)}}$  | {PriorSignOPT_ResNet110AT_VGG13AT_1K} | {PriorSignOPT_ResNet110AT_VGG13AT_3K} | {PriorSignOPT_ResNet110AT_VGG13AT_5K} | {PriorSignOPT_ResNet110AT_VGG13AT_8K} | {PriorSignOPT_ResNet110AT_VGG13AT_10K} |
| $\text{{Prior-Sign-OPT}}_\text{{TRADES(ResNet50)\\&FeatureScatter(ResNet50)}}$  | {PriorSignOPT_ResNet50TRADES_ResNet50FS_1K} | {PriorSignOPT_ResNet50TRADES_ResNet50FS_3K} | {PriorSignOPT_ResNet50TRADES_ResNet50FS_5K} | {PriorSignOPT_ResNet50TRADES_ResNet50FS_8K} | {PriorSignOPT_ResNet50TRADES_ResNet50FS_10K} |
| $\text{{Prior-OPT}}_\text{{ResNet110}}$ | {PriorOPT_ResNet110_1K} | {PriorOPT_ResNet110_3K} | {PriorOPT_ResNet110_5K} | {PriorOPT_ResNet110_8K} | {PriorOPT_ResNet110_10K} |
| $\text{{Prior-OPT}}_\text{{ResNet110\\&DenseNetBC110}}$ | {PriorOPT_ResNet110_DenseNetBC110_1K} | {PriorOPT_ResNet110_DenseNetBC110_3K} | {PriorOPT_ResNet110_DenseNetBC110_5K} | {PriorOPT_ResNet110_DenseNetBC110_8K} | {PriorOPT_ResNet110_DenseNetBC110_10K} |
| $\text{{Prior-OPT}}_\text{{AT(ResNet110)}}$ | {PriorOPT_ResNet110AT_1K} | {PriorOPT_ResNet110AT_3K} | {PriorOPT_ResNet110AT_5K} | {PriorOPT_ResNet110AT_8K} | {PriorOPT_ResNet110AT_10K} |
| $\text{{Prior-OPT}}_\text{{AT(VGG13BN)}}$ | {PriorOPT_VGG13AT_1K} | {PriorOPT_VGG13AT_3K} | {PriorOPT_VGG13AT_5K} | {PriorOPT_VGG13AT_8K} | {PriorOPT_VGG13AT_10K} |
| $\text{{Prior-OPT}}_\text{{AT(ResNet110)\\&AT(VGG13BN)}}$  | {PriorOPT_ResNet110AT_VGG13AT_1K} | {PriorOPT_ResNet110AT_VGG13AT_3K} | {PriorOPT_ResNet110AT_VGG13AT_5K} | {PriorOPT_ResNet110AT_VGG13AT_8K} | {PriorOPT_ResNet110AT_VGG13AT_10K} |
| $\text{{Prior-OPT}}_\text{{TRADES(ResNet50)\\&FeatureScatter(ResNet50)}}$  | {PriorOPT_ResNet50TRADES_ResNet50FS_1K} | {PriorOPT_ResNet50TRADES_ResNet50FS_3K} | {PriorOPT_ResNet50TRADES_ResNet50FS_5K} | {PriorOPT_ResNet50TRADES_ResNet50FS_8K} | {PriorOPT_ResNet50TRADES_ResNet50FS_10K} |
                                """.format(
        HSJA_1K=result["HSJA"][1000],
        HSJA_3K=result["HSJA"][3000],
        HSJA_5K=result["HSJA"][5000],
        HSJA_8K=result["HSJA"][8000],
        HSJA_10K=result["HSJA"][10000],

        SignOPT_1K=result["Sign-OPT"][1000],
        SignOPT_3K=result["Sign-OPT"][3000],
        SignOPT_5K=result["Sign-OPT"][5000],
        SignOPT_8K=result["Sign-OPT"][8000],
        SignOPT_10K=result["Sign-OPT"][10000],

        SVMOPT_1K=result["SVM-OPT"][1000],
        SVMOPT_3K=result["SVM-OPT"][3000],
        SVMOPT_5K=result["SVM-OPT"][5000],
        SVMOPT_8K=result["SVM-OPT"][8000],
        SVMOPT_10K=result["SVM-OPT"][10000],

        PriorSignOPT_ResNet110_1K=result["Prior-Sign-OPT_resnet-110"][1000],
        PriorSignOPT_ResNet110_3K=result["Prior-Sign-OPT_resnet-110"][3000],
        PriorSignOPT_ResNet110_5K=result["Prior-Sign-OPT_resnet-110"][5000],
        PriorSignOPT_ResNet110_8K=result["Prior-Sign-OPT_resnet-110"][8000],
        PriorSignOPT_ResNet110_10K=result["Prior-Sign-OPT_resnet-110"][10000],

        PriorSignOPT_ResNet110_DenseNetBC110_1K=result["Prior-Sign-OPT_resnet-110&densenet-bc-100-12"][1000],
        PriorSignOPT_ResNet110_DenseNetBC110_3K=result["Prior-Sign-OPT_resnet-110&densenet-bc-100-12"][3000],
        PriorSignOPT_ResNet110_DenseNetBC110_5K=result["Prior-Sign-OPT_resnet-110&densenet-bc-100-12"][5000],
        PriorSignOPT_ResNet110_DenseNetBC110_8K=result["Prior-Sign-OPT_resnet-110&densenet-bc-100-12"][8000],
        PriorSignOPT_ResNet110_DenseNetBC110_10K=result["Prior-Sign-OPT_resnet-110&densenet-bc-100-12"][10000],

        PriorSignOPT_ResNet110AT_1K=result["Prior-Sign-OPT_adv_train(resnet-110)"][1000],
        PriorSignOPT_ResNet110AT_3K=result["Prior-Sign-OPT_adv_train(resnet-110)"][3000],
        PriorSignOPT_ResNet110AT_5K=result["Prior-Sign-OPT_adv_train(resnet-110)"][5000],
        PriorSignOPT_ResNet110AT_8K=result["Prior-Sign-OPT_adv_train(resnet-110)"][8000],
        PriorSignOPT_ResNet110AT_10K=result["Prior-Sign-OPT_adv_train(resnet-110)"][10000],

        PriorSignOPT_VGG13AT_1K=result["Prior-Sign-OPT_adv_train(vgg13_bn)"][1000],
        PriorSignOPT_VGG13AT_3K=result["Prior-Sign-OPT_adv_train(vgg13_bn)"][3000],
        PriorSignOPT_VGG13AT_5K=result["Prior-Sign-OPT_adv_train(vgg13_bn)"][5000],
        PriorSignOPT_VGG13AT_8K=result["Prior-Sign-OPT_adv_train(vgg13_bn)"][8000],
        PriorSignOPT_VGG13AT_10K=result["Prior-Sign-OPT_adv_train(vgg13_bn)"][10000],

        PriorSignOPT_ResNet110AT_VGG13AT_1K=result["Prior-Sign-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][1000],
        PriorSignOPT_ResNet110AT_VGG13AT_3K=result["Prior-Sign-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][3000],
        PriorSignOPT_ResNet110AT_VGG13AT_5K=result["Prior-Sign-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][5000],
        PriorSignOPT_ResNet110AT_VGG13AT_8K=result["Prior-Sign-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][8000],
        PriorSignOPT_ResNet110AT_VGG13AT_10K=result["Prior-Sign-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][10000],

        PriorSignOPT_ResNet50TRADES_ResNet50FS_1K=result["Prior-Sign-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][
            1000],
        PriorSignOPT_ResNet50TRADES_ResNet50FS_3K=result["Prior-Sign-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][
            3000],
        PriorSignOPT_ResNet50TRADES_ResNet50FS_5K=result["Prior-Sign-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][
            5000],
        PriorSignOPT_ResNet50TRADES_ResNet50FS_8K=result["Prior-Sign-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][
            8000],
        PriorSignOPT_ResNet50TRADES_ResNet50FS_10K=result["Prior-Sign-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][
            10000],

        PriorOPT_ResNet110_1K=result["Prior-OPT_resnet-110"][1000],
        PriorOPT_ResNet110_3K=result["Prior-OPT_resnet-110"][3000],
        PriorOPT_ResNet110_5K=result["Prior-OPT_resnet-110"][5000],
        PriorOPT_ResNet110_8K=result["Prior-OPT_resnet-110"][8000],
        PriorOPT_ResNet110_10K=result["Prior-OPT_resnet-110"][10000],

        PriorOPT_ResNet110_DenseNetBC110_1K=result["Prior-OPT_resnet-110&densenet-bc-100-12"][1000],
        PriorOPT_ResNet110_DenseNetBC110_3K=result["Prior-OPT_resnet-110&densenet-bc-100-12"][3000],
        PriorOPT_ResNet110_DenseNetBC110_5K=result["Prior-OPT_resnet-110&densenet-bc-100-12"][5000],
        PriorOPT_ResNet110_DenseNetBC110_8K=result["Prior-OPT_resnet-110&densenet-bc-100-12"][8000],
        PriorOPT_ResNet110_DenseNetBC110_10K=result["Prior-OPT_resnet-110&densenet-bc-100-12"][10000],

        PriorOPT_ResNet110AT_1K=result["Prior-OPT_adv_train(resnet-110)"][1000],
        PriorOPT_ResNet110AT_3K=result["Prior-OPT_adv_train(resnet-110)"][3000],
        PriorOPT_ResNet110AT_5K=result["Prior-OPT_adv_train(resnet-110)"][5000],
        PriorOPT_ResNet110AT_8K=result["Prior-OPT_adv_train(resnet-110)"][8000],
        PriorOPT_ResNet110AT_10K=result["Prior-OPT_adv_train(resnet-110)"][10000],

        PriorOPT_VGG13AT_1K=result["Prior-OPT_adv_train(vgg13_bn)"][1000],
        PriorOPT_VGG13AT_3K=result["Prior-OPT_adv_train(vgg13_bn)"][3000],
        PriorOPT_VGG13AT_5K=result["Prior-OPT_adv_train(vgg13_bn)"][5000],
        PriorOPT_VGG13AT_8K=result["Prior-OPT_adv_train(vgg13_bn)"][8000],
        PriorOPT_VGG13AT_10K=result["Prior-OPT_adv_train(vgg13_bn)"][10000],

        PriorOPT_ResNet110AT_VGG13AT_1K=result["Prior-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][1000],
        PriorOPT_ResNet110AT_VGG13AT_3K=result["Prior-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][3000],
        PriorOPT_ResNet110AT_VGG13AT_5K=result["Prior-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][5000],
        PriorOPT_ResNet110AT_VGG13AT_8K=result["Prior-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][8000],
        PriorOPT_ResNet110AT_VGG13AT_10K=result["Prior-OPT_adv_train(resnet-110)&adv_train(vgg13_bn)"][10000],

        PriorOPT_ResNet50TRADES_ResNet50FS_1K=result["Prior-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][1000],
        PriorOPT_ResNet50TRADES_ResNet50FS_3K=result["Prior-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][3000],
        PriorOPT_ResNet50TRADES_ResNet50FS_5K=result["Prior-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][5000],
        PriorOPT_ResNet50TRADES_ResNet50FS_8K=result["Prior-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][8000],
        PriorOPT_ResNet50TRADES_ResNet50FS_10K=result["Prior-OPT_TRADES(resnet-50)&feature_scatter(resnet-50)"][10000],
    )
    )

if __name__ == "__main__":
    dataset = "ImageNet"
    norm = "l2"
    targeted = False
    defense_model = False
    if not defense_model:
        if "CIFAR" in dataset:
            archs = ['pyramidnet272']
        else:
            archs = ["inceptionv4"]
        query_budgets = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        for arch in archs:
            print("=============={}=================".format(arch))
            result = fetch_all_json_content_given_contraint(dataset, norm, arch, False,query_budgets, "mean_distortion")
            draw_table(result)
    else:
        archs = ["resnet-50"]
        query_budgets = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        result = fetch_defense_json_content_given_contraint(dataset, norm, False, query_budgets, "mean_distortion")
        draw_table_for_defensive_models(result)
