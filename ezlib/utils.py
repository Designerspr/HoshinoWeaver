import multiprocessing as mp
import time
from loguru import logger
from math import floor, sqrt
from typing import Callable, Union

import numpy as np
from PIL.ExifTags import TAGS

DTYPE_UPSCALE_MAP = {
    np.dtype('uint8'): np.dtype('uint16'),
    np.dtype('uint16'): np.dtype('uint32'),
    np.dtype('uint32'): np.dtype('uint64')
}

SUPPORT_COLOR_SPACE = ["Adobe RGB", "ProPhoto RGB", "sRGB"]
COMMON_SUFFIX = ["tiff", "tif", "jpg", "png", "jpeg"]
NOT_RECOM_SUFFIX = ["bmp", "gif", "fits"]
RAW_SUFFIX = ["cr2", "cr3", "arw", "nef", "dng"]
SUPPORT_BITS = [8, 16]


class MetaInfo(object):
    """ 

    # MetaInfo

    MetaInfo is used to store metadata. In MetaInfo, tags can be indexed and 
    corresponding values can be got as using string values of tags.

    ## Args:
        init_dict (dict): Metadata dictionary.
        encoding (str, optional): indicates how to decode byte array in metadata. 
                    Defaults to "latin-1".
        str_maxlen (int, optional): the max length for formatting and print, 
                    string longer than which will be shown as "<LongString>". Defaults to 20.
        
    ## Usage
    
    > Initialization. To initialize, simply run `MetaObj=MetaInfo(meta_dict)`, 
    where meta_dict is metadata dictionary.

    > Get available attributes. Since `MetaInfo` is iterable, you can get all 
    available attributes by running `[for tags in MetaObj]`.

    > Get attributes. There are several ways to get a specific attribute. If 
    you want to get "icc_profile", you can use `MetaObj.icc_profile` or 
    `MetaObj.get("icc_profile")`. If you like, 
    `MetaObj.main_dict[MetaObj.tags["icc_profile"]]` can also help you...perhaps.

    > MetaInfo also provides a graceful print list (not exactly). 
    If you want to have a try, run `print(MetaObj)`.

    """

    def __init__(self, init_dict, encoding="latin-1", str_maxlen=20):
        self.main_dict = init_dict
        self.tags = {
            TAGS.get(tag_id, tag_id): tag_id
            for tag_id in self.main_dict
        }
        for tag, tag_id in self.tags.items():
            self.__setattr__(tag, self.main_dict[tag_id])

        self.encoding = encoding
        self.str_maxlen = str_maxlen

    def get(self, string, default=None):
        if string in self.tags:
            return self.main_dict[self.tags[string]]
        if string in self.main_dict:
            return self.main_dict[string]
        return default

    def fmt(self, string):
        string = string.decode(self.encoding) if isinstance(
            string, bytes) else str(string)
        if len(string) > self.str_maxlen:
            return "<LongString>"
        return string

    def __repr__(self) -> str:
        ret_str = "\nMetaInfoList:\n"
        for (key, item) in self.tags.items():
            ret_str += f"{key:<20s}: {self.fmt(self.main_dict[item])} \n"
        return ret_str

    def __iter__(self):
        for tag in self.tags:
            yield tag


def is_support_format(fname) -> bool:
    # suffix check and warning raising
    suffix = fname.split(".")[-1].lower()
    return ((suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX)
            or (suffix in RAW_SUFFIX))


def get_resize(tgt_wh: int, raw_wh: Union[list, tuple]):
    w, h = raw_wh
    tgt_wh_list = [tgt_wh, -1] if w > h else [-1, tgt_wh]
    idn = 0 if tgt_wh_list[0] <= 0 else 1
    idx = 1 - idn
    tgt_wh_list[idn] = int(raw_wh[idn] * tgt_wh_list[idx] / raw_wh[idx])
    return tgt_wh_list


def time_cost_warpper(func: Callable) -> Callable:
    """A decorator that supports to record time cost of the given function.

    Args:
        func (Callable): _description_

    Returns:
        Callable: _description_
    """

    def do_func(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        logger.info(f"{func.__name__} Time Cost: {(time.time()-t0):.2f}s.")
        return res

    return do_func


def get_mp_num(tot_num: int) -> tuple[int, float]:
    """
    设置处理器使用数目，在不超出处理器数目限制的情况下，尽可能使每个处理器叠加sqrt(N)张图像
    推导：n 图像分 m 组叠加，时间开销近似为 [n/m]+m ；min([n/m]+m)-> m取得sqrt(N)
    """
    mp_num = min(floor(sqrt(tot_num)), mp.cpu_count() - 1)
    sub_length = tot_num / mp_num
    return mp_num, sub_length


class GaussianParam(object):
    """

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 mu,
                 var: Union[np.ndarray, int, float] = 0,
                 n: int = 1,
                 ddof: int = 1):
        self.mu = mu
        self.var = var
        self.n = n
        self.ddof = ddof

    def __add__(self, g2):
        g1 = self
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n + g2.mu * g2.n) / (g1.n + g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) +
                   (g2.n - ddof) * g2.var + g2.n * np.square(g2.mu) -
                   (g1.n + g2.n) * np.square(new_mu)) / (g1.n + g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n + g2.n, ddof=ddof)

    def __sub__(self, g2):
        g1 = self
        assert isinstance(g2, GaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert g1.n > g2.n, "generate n<0 fistribution!"
        ddof = g1.ddof
        new_mu = (g1.mu * g1.n - g2.mu * g2.n) / (g1.n - g2.n)
        new_var = ((g1.n - ddof) * g1.var + g1.n * np.square(g1.mu) -
                   (g2.n - ddof) * g2.var - g2.n * np.square(g2.mu) -
                   (g1.n - g2.n) * np.square(new_mu)) / (g1.n - g2.n - ddof)
        return GaussianParam(mu=new_mu, var=new_var, n=g1.n - g2.n, ddof=ddof)


class FastGaussianParam(object):
    """
    GaussianParam, but faster. 通过INT量化+优化数据储存提速。
    Args:
        object (_type_): _description_
    TODO: 验证量化带来的精度损失是否可接受
    （理论可以通过后置除法+提高数据范围提高精度）
    """

    def __init__(self,
                 sum_mu,
                 square_num: Union[np.ndarray, int, float] = None,
                 n: int = 1,
                 ddof: int = 1):
        self.sum_mu = sum_mu
        self.square_sum = square_num if square_num is not None else np.square(
            sum_mu)
        self.n = n
        self.ddof = ddof

    @property
    def mu(self):
        return self.sum_mu // self.n

    @property
    def var(self):
        # TODO: This is not validated.
        return self.square_sum - np.square(self.sum) / (self.n - self.ddof)

    def __add__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        return FastGaussianParam(sum_mu=g1.sum_mu + g2.sum_mu,
                                 square_num=g1.square_sum + g2.square_sum,
                                 n=g1.n + g2.n,
                                 ddof=g1.ddof)

    def __sub__(self, g2):
        g1 = self
        assert isinstance(g2, FastGaussianParam), "unacceptable object"
        assert g1.ddof == g2.ddof, "unmatched var calculation!"
        assert g1.n > g2.n, "generate n<0 fistribution!"
        return FastGaussianParam(sum_mu=g1.sum_mu - g2.sum_mu,
                                 square_num=g1.square_sum - g2.square_sum,
                                 n=g1.n - g2.n,
                                 ddof=g1.ddof)


def test_GaussianParam():
    tot_num = 40
    series = np.random.randn(tot_num)
    base = GaussianParam(mu=series[0])
    for i in range(1, tot_num):
        print("n =", i)
        add = GaussianParam(mu=series[i])
        base = base + add
        print(base.mu, np.mean(series[:i + 1]))
        assert np.abs(base.mu -
                      np.mean(series[:i + 1])) < 1e-8, (base.mu,
                                                        np.mean(series[:i +
                                                                       1]))
        assert np.abs(base.var -
                      np.var(series[:i + 1], ddof=1)) < 1e-8, (base.var,
                                                               np.var(
                                                                   series[:i +
                                                                          1],
                                                                   ddof=1),
                                                               series)
        assert base.n == i + 1
        assert base.ddof == 1