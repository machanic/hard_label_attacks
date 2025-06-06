"""
Provides a class that represents an adversarial example.

"""

import numpy as np
import numbers
import glog as log
import torch
from torch.nn import functional as F

from QEBA.utils import Distance, MSE

class StopAttack(Exception):
    """Exception thrown to request early stopping of an attack
    if a given (optional!) threshold is reached."""
    pass


class Adversarial(object):
    """Defines an adversarial that should be found and stores the result.

    The :class:`Adversarial` class represents a single adversarial example
    for a given model, criterion and reference input. It can be passed to
    an adversarial attack to find the actual adversarial perturbation.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` instance
        The criterion that determines which inputs are adversarial.
    unperturbed : a :class:`numpy.ndarray`
        The unperturbed input to which the adversarial input should be as close as possible.
    original_class : int
        The ground-truth label of the unperturbed input.
    distance : a :class:`Distance` class
        The measure used to quantify how close inputs are.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.

    """

    def __init__(self, model, criterion, unperturbed, original_class, distance=MSE, threshold=None, verbose=False,
                 targeted_attack=False):
        # unperturbed, original_class和实际上传入的是 tgt_image, tgt_label

        self.__model = model
        self.__criterion = criterion  # 放入true label的 TargetClass(true_labels[0].item())
        self.__unperturbed = unperturbed
        self.__unperturbed_for_distance = unperturbed
        self.__original_class = original_class
        self.__distance = distance

        if threshold is not None and not isinstance(threshold, Distance):
            threshold = distance(value=threshold)
        self.__threshold = threshold

        self.verbose = verbose

        self._best_adversarial = None
        self.__best_distance = distance(value=np.inf)
        self.__best_adversarial_output = None

        self._total_prediction_calls = 0
        self._total_gradient_calls = 0

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0
        self.targeted_attack = targeted_attack

        # check if the original input is already adversarial
        self._check_unperturbed()
        self.torch_to_numpy_dtype_dict = {
            torch.bool: np.bool,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.complex64: np.complex64,
        }

    def _check_unperturbed(self):
        try:
            self.forward_one(self.__unperturbed)
        except StopAttack:
            # if a threshold is specified and the unperturbed input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.

    def _reset(self):
        self._best_adversarial = None
        self.__best_distance = self.__distance(value=np.inf)
        self.__best_adversarial_output = None

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        self._check_unperturbed()

    @property
    def perturbed(self):
        """The best adversarial example found so far."""
        return self._best_adversarial

    @property
    def output(self):
        """The model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        return self.__best_adversarial_output

    @property
    def adversarial_class(self):
        """The argmax of the model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        if self.output is None:
            return None
        return torch.argmax(self.output)

    @property
    def distance(self):
        """The distance of the adversarial input to the original input."""
        return self.__best_distance

    @property
    def unperturbed(self):
        """The original input."""
        return self.__unperturbed

    @property
    def original_class(self):
        """The class of the original input (ground-truth, not model prediction)."""
        return self.__original_class

    @property
    def _model(self):  # pragma: no cover
        """Should not be used."""
        return self.__model

    @property
    def _criterion(self):  # pragma: no cover
        """Should not be used."""
        return self.__criterion

    @property
    def _distance(self):  # pragma: no cover
        """Should not be used."""
        return self.__distance

    def set_distance_dtype(self, dtype):
        assert self.torch_to_numpy_dtype_dict[dtype] >= self.__unperturbed.cpu().numpy().dtype
        # assert dtype >= self.__unperturbed.dtype
        self.__unperturbed_for_distance = self.__unperturbed.type(dtype)

    def reset_distance_dtype(self):
        self.__unperturbed_for_distance = self.__unperturbed

    def normalized_distance(self, x):
        """Calculates the distance of a given input x to the original input.

        Parameters
        ----------
        x : `torch.tensor`
            The input x that should be compared to the original input.

        Returns
        -------
        :class:`Distance`
            The distance between the given input and the original input.

        """
        return self.__distance(
            self.__unperturbed_for_distance, x,
            bounds=self.bounds())

    def reached_threshold(self):
        """Returns True if a threshold is given and the currently
        best adversarial distance is smaller than the threshold."""
        return self.__threshold is not None \
            and self.__best_distance <= self.__threshold

    def __new_adversarial(self, x, predictions, in_bounds):
        x = x.clone()  # to prevent accidental inplace changes
        distance = self.normalized_distance(x)
        if in_bounds and self.__best_distance > distance:
            # new best adversarial
            if self.verbose:
                log.info('new best adversarial: {}'.format(distance))

            self._best_adversarial = x
            self.__best_distance = distance
            self.__best_adversarial_output = predictions

            self._best_prediction_calls = self._total_prediction_calls
            self._best_gradient_calls = self._total_gradient_calls

            if self.reached_threshold():
                raise StopAttack

            return True, distance
        return False, distance

    def __is_adversarial(self, x, predictions, in_bounds):
        """Interface to criterion.is_adverarial that calls
        __new_adversarial if necessary.

        Parameters
        ----------
        x : :class:`torch.tensor`
            The input that should be checked.
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some input x.
        label : int
            The label of the unperturbed reference input.

        """
        if not self.targeted_attack:
            is_adversarial = self.__criterion.is_adversarial(
                predictions, self.__original_class)  # __original_class 传入了但是没鸟用，这个is_adversarial的返回：与true label一致，返回True，否则False
        else:
            is_adversarial = self.__criterion.is_adversarial(
                predictions)
        assert isinstance(is_adversarial, bool) or \
            isinstance(is_adversarial, np.bool_) or \
            isinstance(is_adversarial, torch.cuda.BoolTensor)
        if is_adversarial:  # 与true label 一致时
            is_best, distance = self.__new_adversarial(
                x, predictions, in_bounds)
        else:
            is_best = False
            distance = None
        return is_adversarial, is_best, distance

    def target_class(self):
        """Interface to criterion.target_class for attacks.

        """
        try:
            target_class = self.__criterion.target_class()  # 返回true label
        except AttributeError:
            target_class = None
        return target_class

    def num_classes(self):
        n = self.__model.num_classes
        assert isinstance(n, numbers.Number)
        return n

    def bounds(self):
        min_, max_ = self.__model.input_range
        assert isinstance(min_, numbers.Number)
        assert isinstance(max_, numbers.Number)
        assert min_ < max_
        return min_, max_

    def in_bounds(self, input_):
        min_, max_ = self.bounds()
        return min_ <= input_.min() and input_.max() <= max_


    def has_gradient(self):
        """Returns true if _backward and _forward_backward can be called
        by an attack, False otherwise.

        """
        return not self.__model.no_grad


    def forward_one(self, x, strict=True, return_details=False):
        """Interface to model.forward_one for attacks.

        Parameters
        ----------
        x : `torch.tensor`
            Single input with shape as expected by the model
            (without the batch dimension).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        if x.dim() == 3:
            x = x.unsqueeze(0)
        predictions = self.__model.forward(x.cuda()).squeeze(0)
        is_adversarial, is_best, distance = self.__is_adversarial(
            x, predictions, in_bounds)

        assert predictions.dim() == 1
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def forward(self, inputs, greedy=False, strict=True, return_details=False):
        """Interface to model.forward for attacks.

        Parameters
        ----------
        inputs : `torch.tensor`
            Batch of inputs with shape as expected by the model.
        greedy : bool
            Whether the first adversarial should be returned.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        if strict:
            in_bounds = self.in_bounds(inputs)
            assert in_bounds
        assert inputs.dim() == 4
        self._total_prediction_calls += len(inputs)
        predictions = self.__model.forward(inputs.cuda())

        assert predictions.dim() == 2
        assert predictions.shape[0] == inputs.shape[0]

        if return_details:
            assert greedy

        adversarials = []
        for i in range(len(predictions)):
            if strict:
                in_bounds_i = True
            else:
                in_bounds_i = self.in_bounds(inputs[i])
            is_adversarial, is_best, distance = self.__is_adversarial(
                inputs[i], predictions[i], in_bounds_i)  # is_adversarial 其实是 predict == true label
            if is_adversarial and greedy:
                if return_details:
                    return predictions, is_adversarial, i, is_best, distance
                else:
                    return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if greedy:  # pragma: no cover
            # no adversarial found
            if return_details:
                return predictions, False, None, False, None
            else:
                return predictions, False, None

        is_adversarial = torch.tensor(adversarials)  #  # is_adversarial 其实是 predict == true label
        assert is_adversarial.dim() == 1
        assert is_adversarial.shape[0] == inputs.shape[0]
        return predictions, is_adversarial




    def backward_one(self, gradient, x=None, strict=True):
        """Interface to model.backward_one for attacks.

        Parameters
        ----------
        gradient : `torch.tensor`
            Gradient of some loss w.r.t. the logits.
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).

        Returns
        -------
        gradient : `torch.tensor`
            The gradient w.r.t the input.

        See Also
        --------
        :meth:`gradient`

        """
        assert self.has_gradient()
        assert gradient.dim() == 1

        if x is None:
            x = self.__unperturbed

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = self.__model.backward_one(gradient, x)

        assert gradient.shape == x.shape
        return gradient
