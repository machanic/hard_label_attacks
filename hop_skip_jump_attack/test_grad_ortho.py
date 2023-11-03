# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.svm import LinearSVC
import torch
from torch import nn
import numpy as np


def HSJA_sign_gradient(rv, model, decision_boundary=0.0):
    decisions = (model(rv) > decision_boundary).view(-1)
    decision_shape = [decisions.size(0),1]
    fval = 2 * decisions.float().view(decision_shape) - 1.0  # B, 1
    # Baseline subtraction (when fval differs)
    if torch.mean(fval).item() == 1.0:  # label changes.
        gradf = torch.mean(rv, dim=0)
    elif torch.mean(fval).item() == -1.0:  # label not change.
        gradf = -torch.mean(rv, dim=0)
    else:
        # fval -= torch.mean(fval)
        gradf = torch.mean(fval * rv, dim=0)

    # Get the gradient direction.
    gradf = gradf / torch.norm(gradf, p=2)

    return gradf

def normalize(grad):
    return grad / torch.sqrt(torch.sum(torch.square(grad)))




def svm_gradient(rv, model, c, decision_boundary=0.0):
    decisions = (model(rv) > decision_boundary).view(-1)
    svm_model = LinearSVC(C=float(c), dual=True)
    y_train = decisions.int().cpu().numpy()

    #print(y_train)
    x_train = rv.reshape(num_evals, -1).cpu().numpy()
    svm_model.fit(x_train, y_train)
    normal_vector = svm_model.coef_[0]
    gradf = torch.tensor(normal_vector.reshape(-1))
    gradf = gradf / torch.norm(gradf, p=2)
    return gradf


class Classifier(nn.Module):
    def __int__(self, func_type:str, dim:int, weight=None):
        self.func_type = func_type
        if func_type == "linear":
            assert weight is not None
            self.linear_layer = nn.Linear(dim, 1, bias=False, device="cpu")
            self.linear_layer.weight.data = weight

    def forward(self, x):
        if self.func_type == "linear":
            return self.linear_layer(x)
        elif self.func_type == "sigmoid":
            return torch.nn.functional.sigmoid(self.linear_layer(x))  # to [0, 1]
        elif self.func_type == "tanh":
            return torch.nn.functional.tanh(self.linear_layer(x))
        elif self.func_type == "relu":
            return torch.nn.functional.relu(self.linear_layer(x))
        elif self.func_type == "leaky_relu":
            return torch.nn.functional.leaky_relu(self.linear_layer(x))
        elif self.func_type == "elu":
            return torch.nn.functional.elu(self.linear_layer(x))



if __name__ == "__main__":

    dim = 1000
    num_evals = 100
    weight = torch.randn(size=(dim,))
    weight = weight/torch.norm(weight,p=2,keepdim=True)
    weight = weight.view(1, dim)
    linear_model = Classifier("linear", dim, weight)
    surrogate_model_sigmoid = Classifier("sigmoid",dim)
    # surrogate_model_tanh = Classifier("tanh", dim)
    decision_boundary_linear = 0.0
    decision_boundary_sigmoid = 0.5

    HSJA_sim_avg = 0
    SVM_sim_avg = 0
    HSJA_ortho_sim_avg = 0
    SVM_ortho_sim_avg = 0
    for i in range(1000):
        c = 1.0
        noise_shape = [num_evals, dim]
        rv = torch.randn(*noise_shape)
        rv = rv / torch.sqrt(torch.sum(torch.mul(rv, rv), dim=1, keepdim=True))
        ortho_rv = torch.linalg.qr(rv.t())[0].t()

        grad_SVM = svm_gradient(rv, linear_model, c, decision_boundary_linear)
        grad_SVM_ortho = svm_gradient(ortho_rv, linear_model, c, decision_boundary_linear)
        grad_HSJA = HSJA_sign_gradient(rv, linear_model, decision_boundary_linear)
        grad_HSJA_ortho = HSJA_sign_gradient(ortho_rv, linear_model, decision_boundary_linear)

        HSJA_sim = torch.cosine_similarity(weight.view(dim), grad_HSJA, dim=0).item()
        SVM_sim = torch.cosine_similarity(weight.view(dim), grad_SVM, dim=0).item()
        HSJA_ortho_sim = torch.cosine_similarity(weight.view(dim), grad_HSJA_ortho, dim=0).item()
        SVM_ortho_sim = torch.cosine_similarity(weight.view(dim), grad_SVM_ortho, dim=0).item()
        HSJA_sim_avg += HSJA_sim
        SVM_sim_avg += SVM_sim
        HSJA_ortho_sim_avg += HSJA_ortho_sim
        SVM_ortho_sim_avg += SVM_ortho_sim


    HSJA_sim_avg/=1000.0
    SVM_sim_avg/=1000.0
    HSJA_ortho_sim_avg/=1000.0
    SVM_ortho_sim_avg/=1000.0
    print("C={} HSJA : {}".format(c, HSJA_sim_avg))
    print("C={} SVM : {}".format(c, SVM_sim_avg))
    print("C={} HSJA ortho : {}".format(c, HSJA_ortho_sim_avg))
    print("C={} SVM ortho : {}".format(c, SVM_ortho_sim_avg))
    # zhaopingheng = 0
    # if zhaopingheng == 1:
    #     for i in range(100):
    #         weight = torch.randn(size=(3 * size * size,))
    #         weight = weight / torch.norm(weight, p=2)
    #         weight = weight.view(1, 3 * size * size)
    #         num_evals = 100
    #         noise_shape = [num_evals, 3 * size * size]
    #         rv = torch.randn(*noise_shape)
    #         rv = rv / torch.sqrt(torch.sum(torch.mul(rv, rv), dim=1, keepdim=True))
    #         grad_SVM = svm_gradient(rv, weight, 1.0)
    #
    #         HSJA_sim = torch.cosine_similarity(weight.view(3 * size * size), grad_HSJA, dim=0).item()
    #         SVM_sim = torch.cosine_similarity(weight.view(3 * size * size), grad_SVM, dim=0).item()
    #
    #         print("HSJA : {}".format(HSJA_sim))
    #         print("SVM : {}".format(SVM_sim))