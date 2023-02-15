"""
 swdatatoolkit, a project at the Data Mining Lab
 (http://dmlab.cs.gsu.edu/) of Georgia State University (http://www.gsu.edu/).

 Copyright (C) 2022 Georgia State University

 This program is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the Free Software
 Foundation version 3.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy
import numpy as np
from . import PatchSize
from ..edgedetection import GradientCalculator, Gradient
from .util import PeakDetector
from typing import Callable
from scipy.stats import skew, kurtosis, moment
from abc import ABCMeta, abstractmethod


class BaseParamCalculator(metaclass=ABCMeta):
    """
    This is a base abstract class for calculating parameters over some patch of a 2D array.

    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        if patch_size is None:
            raise TypeError("PatchSize cannot be None in ParamCalculator constructor.")
        self._patch_size = patch_size

    @property
    @abstractmethod
    def calc_func(self) -> Callable:
        """
        This polymorphic property is designed to return the parameter calculation function to be applied to each
        patch of the input data.

        :return: :py:class:`typing.Callable` that is the parameter calculation function over a patch of a 2D array.
        """
        pass

    def calculate_parameter(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        This polymorphic method is designed to compute some image parameter. The parameters shall be calculated by
        iterating over a given 2D in a patch by patch manner, calculating the parameter for the pixel values within that
        patch.

        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image

        :return: either a :py:class:`numpy.ndarray` of the parameter value for each patch within the original input
            :py:class:`numpy.ndarray`, or a single value representing the parameter value of the entire input
            :py:class:`numpy.ndarray`.

        """
        if data is None or not isinstance(data, np.ndarray):
            raise TypeError("Data cannot be None and must be of type numpy.ndarray")
        if self._patch_size is None:
            raise TypeError("PatchSize cannot be None in calculator.")

        image_h = data.shape[0]
        image_w = data.shape[1]

        if self._patch_size is PatchSize.FULL:
            return self.calc_func(data)

        p_size = self._patch_size.value

        if image_w % p_size != 0:
            raise ValueError("Width of data must be divisible by given patch size!")
        if image_h % p_size != 0:
            raise ValueError("Height of data must be divisible by given patch size!")

        div_h = image_h // p_size
        div_w = image_w // p_size

        vals = np.zeros((int(div_h), int(div_w)))
        for row in range(div_h):
            for col in range(div_w):
                start_r = p_size * row
                end_r = start_r + p_size
                start_c = p_size * col
                end_c = start_c + p_size
                vals[row, col] = self.calc_func(data[start_r:end_r, start_c:end_c])

        return vals


class MeanParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the mean parameter over some patch of a 2D array.
    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """

        super().__init__(patch_size)
        self._calc_func = np.mean

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


class StdDeviationParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the standard deviation parameter over some patch of a 2D array.

    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)
        self._calc_func = np.std

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# SkewnessParamCalculator
####################################
class SkewnessParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the skewness parameter over some patch of a 2D array.

    See :py:class:`scipy.stats.skew` for additional information on the calculation for
    each cell.
    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        val = skew(data, axis=None)
        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# KurtosisParamCalculator
###################################
class KurtosisParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the kurtosis parameter over some patch of a 2D array.

    See :py:class:`scipy.stats.kurtosis` for additional information on the calculation for
    each cell.
    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        val = kurtosis(data, axis=None)
        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# RelativeSmoothnessParamCalculator
###################################
class RelativeSmoothnessParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the relative smoothness parameter over some patch of a 2D array.


    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            the patch size to calculate the parameter over.

        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        val = np.var(data)
        val = 1 - (1.0 / (1 + val))
        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# UniformityParamCalculator
###################################
class UniformityParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the uniformity parameter over some patch of a 2D array.


    """

    def __init__(self, patch_size: PatchSize, n_bins: int, min_val: float, max_val: float):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            The patch size to calculate the parameter over.
        :param n_bins: int or sequence of scalars or str
            The number of bins to use when constructing the frequency histogram for each patch.
            If bins is an int, it defines the number of equal-width bins in the given range.
            If bins is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths. See :py:class:`numpy.histogram`
            as it is used internally
        :param min_val: py:float
            The minimum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored
        :param max_val: float
            The maximum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored. The max_val must be greater than or equal to min_val.

        """
        super().__init__(patch_size)

        if n_bins is None:
            raise TypeError("n_bins cannot be None in UniformityParamCalculator constructor.")
        if min_val is None:
            raise TypeError("min_val cannot be None in UniformityParamCalculator constructor.")
        if max_val is None:
            raise TypeError("max_val cannot be None in UniformityParamCalculator constructor.")

        if min_val > max_val:
            raise ValueError("max_val cannot be less than min_val in UniformityParamCalculator constructor.")

        self._n_bins = n_bins
        self._range = (min_val, max_val)

    @property
    def calc_func(self) -> Callable:
        return self.__calc_uniformity

    def __calc_uniformity(self, data: numpy.ndarray) -> float:
        """
        Helper method that performs the uniformity calculation for one patch.

        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image
        :return: The uniformity parameter value for the patch passed in

        """

        hist, bin_edges = np.histogram(data, self._n_bins, range=self._range)
        image_h = data.shape[0]
        image_w = data.shape[1]

        n_pix = float(image_w * image_h)
        sum = 0.0
        for i in range(len(hist)):
            count = hist[i]
            if count == 0:
                continue
            prob = hist[i] / n_pix
            sum += np.power(prob, 2)

        return sum


###################################
# EntropyParamCalculator
###################################
class EntropyParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the entropy parameter over some patch of a 2D array.
    The calculation is performed in the following manner:

    .. math:: E = - \\sum_{i=1}^{N} p(z_i)* log_2(p(z_i))

    where:

    - :math:`p` is the histogram of a patch

    - :math:`z_i` is the intensity value of the i-th pixel in the patch

    - :math:`p(z_i)` is the frequency of the intensity :math:`z_i` in the histogram of the patch

    """

    def __init__(self, patch_size: PatchSize, n_bins: int, min_val: float, max_val: float):
        """
        Constructor

        :param patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
            The patch size to calculate the parameter over.
        :param n_bins: int or sequence of scalars or str
            The number of bins to use when constructing the frequency histogram for each patch.
            If bins is an int, it defines the number of equal-width bins in the given range.
            If bins is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths. See :py:class:`numpy.histogram`
            as it is used internally
        :param min_val: py:float
            The minimum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored
        :param max_val: float
            The maximum value to use when constructing the frequency histogram for each patch.
            Values outside the range are ignored. The max_val must be greater than or equal to min_val.

        """
        super().__init__(patch_size)

        if n_bins is None:
            raise TypeError("n_bins cannot be None in EntropyParamCalculator constructor.")
        if min_val is None:
            raise TypeError("min_val cannot be None in EntropyParamCalculator constructor.")
        if max_val is None:
            raise TypeError("max_val cannot be None in EntropyParamCalculator constructor.")

        if min_val > max_val:
            raise ValueError("max_val cannot be less than min_val in EntropyParamCalculator constructor.")

        self._n_bins = n_bins
        self._range = (min_val, max_val)

    @property
    def calc_func(self) -> Callable:
        return self.__calc_entropy

    def __calc_entropy(self, data: numpy.ndarray) -> float:
        """
        Helper method that performs the entropy calculation for one patch.

        :param data: :py:class:`numpy.ndarray`
            2D matrix representing some image
        :return: The entropy parameter value for the patch passed in

        """

        hist, bin_edges = np.histogram(data, self._n_bins, range=self._range)
        image_h = data.shape[0]
        image_w = data.shape[1]

        n_pix = float(image_w * image_h)
        sum = 0.0
        for i in range(len(hist)):
            count = hist[i]
            if count == 0:
                continue
            prob = hist[i] / n_pix
            sum += prob * (np.log2(prob))

        return 0 - sum


###################################
# TContrastParamCalculator
###################################
class TContrastParamCalculator(BaseParamCalculator):
    """
    This class is for calculating the Tamura Contrast parameter over some patch of a 2D array.
    The calculation is performed in the following manner:

    .. math:: C = \\frac{\\sigma^{2}}{{\\mu_4}^{0.25}}

    where:

    - :math:`\\sigma^{2}` is the variance of the intensity values in the patch

    - :math:`\\mu_4` is kurtosis (4-th moment about the mean) of the intensity values in the patch

    This formula is an approximation proposed by Tamura et al. in "Textual Features Corresponding Visual
    Perception" and investigated in "On Using SIFT Descriptors for Image Parameter Evaluation"

    """

    def __init__(self, patch_size: PatchSize):
        """
        Constructor

        :param patch_size: Tthe patch size to calculate the parameter over.
        :type patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
        """
        super().__init__(patch_size)

    @staticmethod
    def _calc_func(data: numpy.ndarray) -> numpy.ndarray:
        kurt_val = moment(data, moment=4, axis=None)
        if kurt_val == 0:
            return 0.0
        std_val = np.std(data)

        # TContrast = (sd ^ 2)/(kurtosis ^ 0.25)
        val = np.power(std_val, 2) / np.power(kurt_val, 0.25)
        if np.isnan(val):
            return 0.0

        return val

    @property
    def calc_func(self) -> Callable:
        return self._calc_func


###################################
# TDirectionalityParamCalculator
###################################
class TDirectionalityParamCalculator(BaseParamCalculator):

    def __init__(self, patch_size: PatchSize, gradient_calculator: GradientCalculator, peak_detector: PeakDetector,
                 quantization_level: int):
        """
        Constructor

        :param patch_size: The patch size to calculate the parameter over.
        :param gradient_calculator: Calculator for gradient of pixel values in the image being processed.
        :param peak_detector: Object that finds the local maxima in an ordered series of values
        :param quantization_level: The quantization level for the continuous spectrum of the angles of gradients. This
            is the number of bins in a histogram of gradient angles.
        :type patch_size: :py:class:`swdatatoolkit.imageparam.PatchSize`
        """
        super().__init__(patch_size)
        self._gradient_calculator = gradient_calculator
        self._peak_detector = peak_detector
        self._quantization_level = quantization_level
        self._radius_threshold_percentage = 0.15
        self._insignificant_radius = 1e-4

    @property
    def calc_func(self) -> Callable:
        pass

    def __calc_directionality(self, data: numpy.ndarray) -> float:

        gradient = self._gradient_calculator.calculate_gradient_polar(data)

        gradient_theta = gradient.gx
        gradient_radii = gradient.gy


