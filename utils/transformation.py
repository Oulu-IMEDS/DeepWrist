from solt.base_transforms import DataDependentSamplingTransform, PaddingPropertyHolder, ImageTransform
from solt.constants import allowed_paddings
from solt.data import DataContainer, KeyPoints
import numpy as np
import cv2
from solt.utils import img_shape_checker
import matplotlib.pyplot as plt
from localizer.kneel_before_wrist.data.utils import solt2torchhm


def get_random_or_fixed_color(colors):
    if isinstance(colors, tuple) or isinstance(colors, list):
        if len(colors) != 2:
            raise ValueError('color range should be between two valuse')
        if colors[0] > colors[1]:
            tmp = colors[0]
            colors[0] = colors[1]
            colors[1] = tmp
        color = np.random.randint(low=colors[0], high=colors[1])
    else:
        color = colors
    return color

class SIDES:
    LEFT = 0
    RIGHT = 1
    TOP = 3
    BOTTOM = 4
    LEFT_RIGHT = 5
    LEFT_TOP = 6
    LEFT_BOTTOM = 7
    RIGHT_TOP = 8
    RIGHT_BOTTOM = 9
    TOP_BOTTOM = 10
    LEFT_RIGHT_TOP = 11
    LEFT_RIGHT_BOTTOM = 12
    LEFT_TOP_BOTTOM = 13
    RIGHT_TOP_BOTTOM = 14
    ALL = 15
    RANDOM = 16
    LEFTS = [LEFT, LEFT_TOP, LEFT_BOTTOM, LEFT_RIGHT, LEFT_RIGHT_TOP, LEFT_RIGHT_BOTTOM, LEFT_TOP_BOTTOM, ALL]
    RIGHTS = [RIGHT, RIGHT_TOP, RIGHT_BOTTOM, LEFT_RIGHT, LEFT_RIGHT_BOTTOM, LEFT_RIGHT_TOP, RIGHT_TOP_BOTTOM,  ALL]
    TOPS = [TOP, LEFT_TOP, RIGHT_TOP, TOP_BOTTOM, LEFT_RIGHT_TOP, RIGHT_TOP_BOTTOM,LEFT_TOP_BOTTOM, ALL]
    BOTTOMS = [BOTTOM, TOP_BOTTOM, LEFT_BOTTOM, RIGHT_BOTTOM, RIGHT_TOP_BOTTOM, LEFT_RIGHT_BOTTOM, LEFT_TOP_BOTTOM, ALL]


class ColorPaddingWithSide(DataDependentSamplingTransform):

    def __init__(self, p, pad_size, side=SIDES.ALL, color=0):
        DataDependentSamplingTransform.__init__(self, p=p)
        if not isinstance(pad_size, tuple) and not isinstance(pad_size, int):
            raise TypeError
        if isinstance(pad_size, tuple) and not len(pad_size) == 4:
            raise ValueError('Pad size must be a single integer or tuple of (left, top, right, bottom)')
        if isinstance(pad_size, int):
            pad_size = (pad_size, pad_size, pad_size, pad_size)

        pad_list = list(pad_size)
        for i in range(len(pad_list)):
            size = pad_list[i]
            if size < 0:
                pad_list[i] = 0
        pad_size = tuple(pad_list)
        self._color = color
        self._pad_size = pad_size
        if side == SIDES.RANDOM:
            side = np.random.randint(low=0, high=100)%16
        self._side = side
        self._padding = cv2.BORDER_CONSTANT
        self._original_size = None

    def sample_transform(self):
        DataDependentSamplingTransform.sample_transform(self)

    def sample_transform_from_data(self, data: DataContainer):
        h, w = DataDependentSamplingTransform.sample_transform_from_data(self, data)
        self._original_size = (w, h)

        if self._side in SIDES.TOPS:
            pad_h_top = self._pad_size[1]
        else:
            pad_h_top = 0
        if self._side in SIDES.BOTTOMS:
            pad_h_bottom = self._pad_size[3]
        else:
            pad_h_bottom = 0
        if self._side in SIDES.LEFTS:
            pad_w_left = self._pad_size[0]
        else:
            pad_w_left = 0
        if self._side in SIDES.RIGHTS:
            pad_w_right = self._pad_size[2]
        else:
            pad_w_right = 0

        self.state_dict = {'pad_h': (pad_h_top, pad_h_bottom), 'pad_w': (pad_w_left, pad_w_right)}

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        color = get_random_or_fixed_color(self._color)
        pad_h_top, pad_h_bottom = self.state_dict['pad_h']
        pad_w_left, pad_w_right = self.state_dict['pad_w']

        if settings['padding'][1] == 'strict':
            padding = allowed_paddings[settings['padding'][0]]

        out = cv2.copyMakeBorder(img, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, self._padding, value=color)
        out = cv2.resize(out, self._original_size)
        return out

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        pts_data = pts.data.copy()

        pad_h_top, pad_h_bottom = self.state_dict['pad_h']
        pad_w_left, pad_w_right = self.state_dict['pad_w']

        pts_data[:, 0] += pad_w_left
        pts_data[:, 1] += pad_h_top
        pts_data = pts_data.astype(np.float32)
        pts_data[:, 1] = (pts_data[:, 1] / (pad_h_top + pts.H + pad_h_bottom)) * self._original_size[1]
        pts_data[:, 0] = (pts_data[:, 0] / (pad_w_left + pts.W + pad_w_right))* self._original_size[0]

        # return KeyPoints(pts_data, pad_h_top + pts.H + pad_h_bottom,
        #                  pad_w_left + pts.W + pad_w_right)
        return KeyPoints(pts_data,  pts.H,
                         pts.W)


class SubSampleUpScale(DataDependentSamplingTransform):

    def __init__(self, p):
        DataDependentSamplingTransform.__init__(self, p=p)
        self.p = p
        self.original_size = None

    def sample_transform(self):
        DataDependentSamplingTransform.sample_transform(self)

    def sample_transform_from_data(self, data: DataContainer):
        DataDependentSamplingTransform.sample_transform_from_data(self, data)

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        self.original_size = img.shape
        timg = cv2.GaussianBlur(img, (3,3), 0.1)
        timg = cv2.resize(timg, (img.shape[0] //2, img.shape[1]//2))

        timg = cv2.GaussianBlur(timg, (3,3), 0.1)
        timg = cv2.resize(timg, (img.shape[0] // 4, img.shape[1] // 4))

        out = cv2.resize(timg, (img.shape[0], img.shape[1]))


        return out

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        return pts


class TriangularMask(ImageTransform):
    ALLOWED_SIDES = [SIDES.LEFT_TOP, SIDES.LEFT_BOTTOM, SIDES.RIGHT_TOP, SIDES.RIGHT_BOTTOM]

    def __init__(self, p, arm_lengths, side, color):
        super(TriangularMask, self).__init__(p=p)
        if side == SIDES.RANDOM:
            index = np.random.randint(low=0, high=100)%4
            side = self.ALLOWED_SIDES[index]
        if side not in self.ALLOWED_SIDES:
            raise ValueError(f'Triangle arms should be consecutive sides {side}' )
        if not isinstance(arm_lengths, tuple) and not isinstance(arm_lengths, int):
            raise TypeError
        if isinstance(arm_lengths, tuple) and not len(arm_lengths) == 2:
            raise ValueError('Pad size must be a single integer or tuple of (left, top, right, bottom)')
        if isinstance(arm_lengths, int):
            arm_lengths = (arm_lengths, arm_lengths, arm_lengths, arm_lengths)

        arm_list = list(arm_lengths)
        for i in range(len(arm_list)):
            size = arm_list[i]
            if size <= 0:
                raise ValueError('An arm of a triangle cannot be 0 or less than 0')
        arm_lengths = tuple(arm_list)
        self._color = color
        self._arm_lengths = arm_lengths

        self._side = side
        self._padding = cv2.BORDER_CONSTANT

    def sample_transform(self):
        pass

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        color = get_random_or_fixed_color(self._color)
        h, w = img.shape[0], img.shape[1]
        if self._side == SIDES.LEFT_TOP:
            origin = (0, 0)
            p1 = (0, self._arm_lengths[1])
            p2 = (self._arm_lengths[0], 0)

        elif self._side == SIDES.LEFT_BOTTOM:
            origin = (0, w)
            p1 = (0, w - self._arm_lengths[1])
            p2 = (self._arm_lengths[0], w)

        elif self._side == SIDES.RIGHT_TOP:
            origin = (h, 0)
            p1 = (h, self._arm_lengths[1])
            p2 = (h - self._arm_lengths[0], 0)
        else:
            origin = (h, w)
            p1 = (h, w - self._arm_lengths[1])
            p2 = (h - self._arm_lengths[0], w)

        self._points = np.array([p1, p2, origin])
        cv2.drawContours(img, [self._points], 0, color, -1)
        return img


class LowVisibilityTransform(ImageTransform):
    def __init__(self, p, alpha, bgcolor=0):
        super(LowVisibilityTransform, self).__init__(p=p)
        if alpha <=0:
            raise ValueError('Image alpha should be greater than 0')
        self._alpha = alpha
        self._bgcolor = bgcolor

    def sample_transform(self):
        pass

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        color = get_random_or_fixed_color(self._bgcolor)
        background = np.ones_like(img)
        background *= color
        background = background.astype(np.uint8)

        beta = 1.0 - self._alpha
        out = img * self._alpha + background * beta
        out = out.astype(np.uint8)
        return out





