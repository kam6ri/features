import difflib
import re
import unicodedata
from typing import *
from typing import Callable

import kanjize
import numpy as np
import pandas as pd
import simple_geocoding as sg
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, _fit_transform_one, _transform_one
from sklearn.preprocessing import FunctionTransformer
from sklearn_pandas import DataFrameMapper
import faiss

class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(transformer=trans, X=X, y=y, weight=weight, **fit_params)
            for _, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(transformer=trans, X=X, y=None, weight=weight)
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
 
    
class KNNFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, cols, target_col, k=10, d=2, metric=faiss.METRIC_L2, train_with_fit=True):
        self.cols = cols
        self.target_col = target_col
        self.k = k
        self.d = d
        self.quantizer = faiss.IndexFlatL2(d) 
        self.indexer = None
        self.metric = metric
        self.targets = None
        self.train_with_fit = train_with_fit
    
    def train(self, X: pd.DataFrame):
        self.targets = np.append(X[self.target_col].to_numpy(), [None])
        X_ = np.ascontiguousarray(X[self.cols].to_numpy()).astype(np.float32)
        assert X_.shape[1] == self.d
        nlist = len(X_) // 39 
        self.indexer = faiss.IndexIVFFlat(self.quantizer, self.d, nlist, self.metric)
        X_ = np.nan_to_num(X_)
        self.indexer.train(X_)
        self.indexer.add(X_)
            
    def fit(self, X: pd.DataFrame, y=None):
        if self.train_with_fit:
            self.train(X)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_ = np.ascontiguousarray(X[self.cols].to_numpy()).astype(np.float32)
        _, I = self.indexer.search(X_, self.k)
        fn = np.vectorize(lambda x: self.targets[x])
        return fn(I)

# 向きの特徴量を算出
def convert_madori(s: str):
    s = s.replace("ワンルーム", "1R")
    ret = [
        1 if re.match(r"\d*S.*", s) else 0,
        1 if re.match(r"\d*.*LDK.*", s) else 0,
        1 if re.match(r"\d*.*[^L]DK.*", s) else 0,
        1 if re.match(r"\d*.*[^LD]K.*", s) else 0,
        1 if re.match(r"\d*.*LK.*", s) else 0,
        1 if re.match(r"1R", s) else 0,
        int(match.group(0)) if (match := re.match(r"(\d+)(?=[^R]+)", s)) else 0,
    ]
    ret.append(sum(ret))
    return ret

def to_pipe(transformers: list, **kwargs):
    return Pipeline(transformers, **kwargs)

def to_union(transformers: list, **kwargs):
    return PandasFeatureUnion(transformers, **kwargs)

def to_transformer(fn):
    return FunctionTransformer(lambda x: x.apply(fn).apply(pd.Series))

def df_mapper(name: str, cols: Union[str, list[str]], transformer: Optional[BaseEstimator] = None):
    return DataFrameMapper([(cols, transformer, {"alias": name})], input_df=True, df_out=True, default=None)

# 1to1 or 1tox transformation, original columns dropped
def mapping(name: str, cols: Union[str, list[str]], transformer: Optional[BaseEstimator] = None, drop=True, default=False):
    remains = []
    if not drop:
        remains = [(col, None, {"alias": col}) for col in cols] if isinstance(cols, list) else [(cols, None, {"alias": cols})]
    transformer = DataFrameMapper([
        *remains,
        (cols, transformer, {"alias": name})
    ], input_df=True, df_out=True, default=default)
    return name, transformer

# 1to1 or 1tox transformation, original columns dropped
def operation(name: str, cols: Union[str, list[str]], op: Callable, drop=True, default=False):
    transformer = FunctionTransformer(lambda x: x.apply(lambda x: op(x), axis=1))
    remains = []
    if not drop:
        remains = [(col, None, {"alias": col}) for col in cols] if isinstance(cols, list) else [(cols, None, {"alias": cols})]
    transformer = DataFrameMapper([
        *remains,
        (cols, transformer, {"alias": name}),
    ], input_df=True, df_out=True, default=default)

    return name, transformer


# 専有面積の特徴量を算出
def convert_area(s: str):
    return float(match.group(1)) if (match := re.match(r"([\d\.]+)m2", s)) else None


# 築年数の特徴量を算出
def convert_age(s: str):
    return int(match.group(1)) if (match := re.search(r"(\d+)", s)) else 0


# 向きの特徴量を算出
def convert_direction(s: str):
    return [int(p in s) for p in "北東南西"]


# 駐車場の特徴量を算出
def convert_parking(s: str):
    ret = [0, 0]
    if re.search(r"無料", s):
        ret[0] = 1
    elif match := re.match(r"[敷地内近隣]+(?:\d+m)?(\d+)円", s):
        ret[0] = 1
        ret[1] = int(match.group(1))
    return ret

def address_preprocessing(s: str):
    s_ = unicodedata.normalize("NFKC", s)
    groups = re.match(r"(\D*)([\d\-]*)", s_).groups()

    if groups[1] != "":
        if len(nums := groups[1].split("-")) >= 1:
            nums = list(map(lambda x: kanjize.int2kanji(int(x)), nums))
            s_ = groups[0] + f"{nums[0]}丁目"
    
    return s_

def drop_number(s: str):
    s_ = unicodedata.normalize("NFKC", s)
    s_ = re.match(r"(\D*)", s_).groups()[0]
    return s_

# 所在地の特徴量を算出
def convert_address(s: str, geocoder: sg.Geocoding, search_unknown=False):

    s_ = address_preprocessing(s)

    if s_ not in geocoder.addr2pt and search_unknown:
        if len(matches := difflib.get_close_matches(s_, geocoder.addr2pt, n=1)) > 0:
            s_ = matches[0]

    if (coordinates := geocoder.point(s_)) is None:
        coordinates = [None, None]

    return coordinates





OPTIONS = [
    "最上階",
    "追焚機能浴室",
    "宅配ボックス",
    "2口コンロ",
    "IT重説 対応物件",
    "CATV",
    "浴室乾燥機",
    "オートロック",
    "洗面化粧台",
    "駅徒歩10分以内",
    "2沿線利用可",
    "礼金不要",
    "光ファイバー",
    "初期費用カード決済可",
    "角住戸",
    "3駅以上利用可",
    "即入居可",
    "敷地内ごみ置き場",
    "システムキッチン",
    "保証人不要",
    "洗面所独立",
    "都市ガス",
    "温水洗浄便座",
    "ガスコンロ対応",
    "クロゼット",
    "TVインターホン",
    "駐輪場",
    "シューズボックス",
    "フローリング",
    "バルコニー",
    "バストイレ別",
    "室内洗濯置",
    "エアコン",
]

# 詳細条件の特徴量を算出
def convert_option(s: str, options: list[str] = OPTIONS):
    ret = [0 for _ in options]
    for op in s.split("、"):
        if op in options:
            ret[options.index(op)] = 1
    return ret
