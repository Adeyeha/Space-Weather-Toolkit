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

import math
import numpy
import numpy as np


class Gradient:
    """
    This class is for holding the gradient values of an image in the x and y
    direction.
    """

    def __init__(self, gx: numpy.ndarray = None, gy: numpy.ndarray = None, gd: numpy.ndarray = None, nx: int = None,
                 ny: int = None):
        """
        Constructor requires nx and ny to be set if not passing in gx. Otherwise, it assumes that gy and gd are provided
        and the same shape of gx or need to be constructed in the same shape as gx.

        :param gx: A 2D matrix representing some image
        :param gy: A 2D matrix representing some image
        :param gd: A 2D matrix representing some image
        :param nx: The number of columns in the image this gradient object represents
        :param ny: The number of rows in the image ths gradient object represents
        :type gx: :py:class:`numpy.ndarray`
        :type gy: :py:class:`numpy.ndarray`
        :type gd: :py:class:`numpy.ndarray`
        :type nx: int
        :type ny: int
        :raises ValueError: If no `nx` and `ny` are supplied when no `gx` array is supplied.  Also if `gy` or `gd` are
                    supplied and are of a different size than a supplied `gx` or the `nx` and `ny` size.

        """
        if gx is None:
            if nx is None or ny is None:
                raise ValueError("nx and ny cannot be none when gx is not set!")
            else:
                self._gx = numpy.zeros([ny, nx])
        else:
            self._gx = gx
            ny = gx.shape[0]
            nx = gx.shape[1]

        if gy is None:
            self._gy = numpy.zeros([ny, nx])
        else:
            if gy.shape[0] == ny and gy.shape[1] == nx:
                self._gy = gy
            else:
                raise ValueError("gy array shape must match supplied size of gx")

        if gd is None:
            self._gd = numpy.zeros([ny, nx])
        else:
            if gd.shape[0] == ny and gd.shape[1] == nx:
                self._gd = gd
            else:
                raise ValueError("gd array shape must match supplied size of gx")

    @property
    def gx(self) -> numpy.ndarray:
        """
        In case of Gradient in the Cartesian system, this is the matrix of gradient values when comparing pixels
        in the X direction. In case of Gradient in the Polar system, this is the matrix of angles.

        :return:
        """
        return self._gx

    @gx.setter
    def gx(self, x, y, value):
        self._gx[y, x] = value

    @property
    def gy(self) -> numpy.ndarray:
        """
        In case of Gradient in the Cartesian system, this is the matrix of gradient values when comparing pixels
        in the Y direction. In case of Gradient in the Polar system, this is the matrix of Radii.

        :return:
        """
        return self._gy

    @gy.setter
    def gy(self, x, y, value):
        self._gy[y, x] = value

    @property
    def gd(self) -> numpy.ndarray:
        """
        This is the same for both Cartesian and Polar system. This is an auxiliary matrix to help distinguish
        the zero values in the Polar system. Gradient in the Polar system gets zero in the following cases and
        without 'gd' there is no way to distinguish them: 1. gx[i][j] = 0, gy[i][j] = 0 --&gt; because (i,j)
        lies on a solid area. 2. gx[i][j] = 0, gy[i][j] = 0 --&gt; because (i,j) lies on a vertical line of
        width 1 px. 3. gx[i][j] = 0, gy[i][j] = 0 --&gt; because (i,j) lies on a horizontal line of width 2 px.

        :return:
        """
        return self._gd

    @gd.setter
    def gd(self, x, y, value):
        self._gd[y, x] = value


class GradientCalculator:
    """
    A class for calculating the gradient of pixel intensities on source images.
    This class uses different kernels, depending on the provided gradient operator name, and calculated both horizontal
    and vertical derivatives, as well as the magnitude and angle of changes in the color intensity of pixels.

    There exists other operators which have not been yet implemented in this class:
        - `Prewitt operator <https://en.wikipedia.org/wiki/Prewitt_operator>`_
        - `Sobel operator <https://en.wikipedia.org/wiki/Sobel_operator>`_
        - `Roberts operator <https://en.wikipedia.org/wiki/Roberts_cross>`_

    """

    def __init__(self, gradient_name: str = 'sobel'):
        """
        This constructor sets the gradient operator based on the given name.

        :param gradient_name: Name of the gradient operator among the following options:

            - prewitt: `Prewitt operator <https://en.wikipedia.org/wiki/Prewitt_operator>`_
            - sobel: `Sobel operator <https://en.wikipedia.org/wiki/Sobel_operator>`_ (default)
            - roberts: `Roberts operator <https://en.wikipedia.org/wiki/Roberts_cross>`_
        :type gradient_name: str
        :raises NotImplementedError: If supplied string is something other than the above listed options

        """
        if gradient_name == 'prewitt':
            self._x_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            self._y_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        elif gradient_name == 'sobel':
            self._x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            self._y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif gradient_name == 'roberts':
            self._x_kernel = np.array([[1, 0], [0, -1]])
            self._y_kernel = np.array([[0, 1], [-1, 0]])
        else:
            raise NotImplementedError("Filter Type {} not implemented.".format(gradient_name))

    def calculate_gradient_polar(self, image: numpy.ndarray) -> Gradient:
        """
        Calculates the gradient of pixel intensities on the input image and returns the results in a Polar coordinate
        system.

        :param image: A :py:class:`numpy.ndarray` 2D matrix representing some image
        :return: An object that represents the gradient as gx=theta and gy=r with range of theta is (-3.14, +3.14)
        :type image: :py:class:`numpy.ndarray`
        :rtype: :py:class:`swdatatoolkit.edgedetection.Gradient`
        :raises TypeError: If image is not an :py:class:`numpy.ndarray`
        :raises ValueError: If image is zero sized in either dimension

        """
        grad_cart = self.calculate_gradient_cart(image)
        gx = grad_cart.gx
        gy = grad_cart.gy
        gd = grad_cart.gd

        im_shape = image.shape

        # Angles of the gradient
        t_grad = np.zeros(im_shape)
        # Radius (magnitude) of the gradient
        r_grad = np.zeros(im_shape)

        for row in range(im_shape[0]):
            for col in range(im_shape[1]):
                r_grad[row, col] = math.hypot(gx[row, col], gy[row, col])
                """
                As the definition of atan2 indicates (Wikipedia/atan2), this function returns zero in 2 cases (this was 
                also tested and it confirms that there are ONLY these two cases:
                
                1. If x > 0, y = 0: This is a meaningful zero, since in a triangle, the angle against an edge of length 
                zero (y = 0) is zero. 
                
                2. If x = y = 0: This is mathematically undefined, but in Math library, atan2(0,0) returns zero. This
                zero represent a solid region with no particular texture. This zero should be treated differently.
                
                If we do not distinguish these two cases, then in the histogram of angles, we will not be able to ignore 
                the bin at hist[0] which is disproportionately larger than other bins.
                
                NOTE: In TDirectionalityParamCalculator, we set will hist[0] to zero to avoid detecting this bin as a 
                dominant peak.
                """

                # In case x=y=0, the followings statements are needed to distinguish horizontal (or vertical) lines on
                # a solid bg.

                if gx[row, col] == 0 and gy[row, col] == 0:
                    # If this pixel lies on a solid region
                    if gd[row, col] == 0:
                        t_grad[row, col] = - math.pi  # Reserving this constant for meaningless zeros on a solid region
                    elif gd[row, col] > 0:
                        # If this pixel lies on a vertical line
                        t_grad[row, col] = math.pi
                    else:
                        # If this pixel lies on a horizontal line
                        t_grad[row, col] = math.pi / 2.0
                elif gx[row, col] > 0 and gy[row, col] == 0:
                    t_grad[row, col] = math.pi  # This constant is reserved for meaningful zeros
                else:
                    t_grad[row, col] = math.atan2(gy[row, col], gx[row, col])

        polar_grad = Gradient(t_grad, r_grad, gd)
        return polar_grad

    def calculate_gradient_cart(self, image: numpy.ndarray) -> Gradient:
        """
        Calculates the gradient of pixel intensities on the input image and returns the results in the original
        Cartesian coordinate system.

        :param image: A 2D matrix representing some image
        :return: An object that represents the gradient in x and y direction
        :type image: :py:class:`numpy.ndarray`
        :rtype: :py:class:`swdatatoolkit.edgedetection.Gradient`
        :raises: :TypeError: If image is not an :py:class:`numpy.ndarray`
        :raises: :ValueError: If image is zero sized in either dimension
        """
        if not isinstance(image, numpy.ndarray):
            raise TypeError('The image must be of type numpy.ndarray!')

        im_shape = image.shape
        if im_shape[0] == 0 or im_shape[1] == 0:
            raise ValueError('The image must be non-zero size!')

        x_grad = np.zeros(im_shape)
        y_grad = np.zeros(im_shape)
        d_grad = np.zeros(im_shape)

        grad = Gradient(x_grad, y_grad, d_grad)

        kern_shape = self._x_kernel.shape
        for row in range(im_shape[0]):
            for col in range(im_shape[1]):

                if not (row == 0 or row == im_shape[0] - 1 or col == 0 or col == im_shape[1] - 1):
                    # xsum = 0.0
                    # ysum = 0.0
                    im_slice = image[row - 1:row + kern_shape[0] - 1, col - 1:col + kern_shape[1] - 1]

                    xsum = numpy.tensordot(im_slice, self._x_kernel, axes=2)
                    ysum = numpy.tensordot(im_slice, self._y_kernel, axes=2)
                    # for kr in range(kern_shape[0]):
                    #    for kc in range(kern_shape[1]):
                    #        px = im_slice[kr, kc]
                    #        xsum += (px * self._x_kernel[kr, kc])
                    #        ysum += px * self._y_kernel[kr, kc]

                    x_grad[row, col] = float(xsum)
                    y_grad[row, col] = float(ysum)
                    d_grad[row, col] = abs(im_slice[0, 1] - im_slice[1, 0])

        return grad
