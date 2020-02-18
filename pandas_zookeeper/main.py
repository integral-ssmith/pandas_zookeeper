import warnings
import numpy as np
import pandas as pd
import re
import string


@pd.api.extensions.register_dataframe_accessor('zookeeper')
class ZooKeeper:
    def __init__(self, pandas_obj):
        # validate and assign object
        self._validate(pandas_obj)
        self._obj = pandas_obj

        # define incorporated modules - columns consisting of others will not have the dtype changed
        self._INCORPORATED_MODULES = ['builtins', 'numpy', 'pandas']

        # define a possible list of null values
        self._NULL_VALS = [None, np.nan, 'np.nan', 'nan', np.inf, 'np.inf', 'inf', -np.inf, '-np.inf', '', 'n/a', 'na',
                           'N/A', 'NA', 'unknown', 'unk', 'UNKNOWN', 'UNK']

        # assign dtypes and limits
        # boolean
        BOOL_STRINGS_TRUE = ['t', 'true', 'yes', 'on']
        BOOL_STRINGS_FALSE = ['f', 'false', 'no', 'off']
        self._BOOL_MAP_DICT = {i: True for i in BOOL_STRINGS_TRUE}.update({i: False for i in BOOL_STRINGS_FALSE})
        self._DTYPE_BOOL_BASE = np.bool
        self._DTYPE_BOOL_NULLABLE = pd.BooleanDtype()
        # unsigned integers - base and nullable
        self._DTYPES_UINT_BASE = [np.uint8, np.uint16, np.uint32, np.uint64]
        self._DTYPES_UINT_NULLABLE = [pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype()]
        self._LIMIT_LOW_UINT = [np.iinfo(i).min for i in self._DTYPES_UINT_BASE]
        self._LIMIT_HIGH_UINT = [np.iinfo(i).max for i in self._DTYPES_UINT_BASE]
        # signed integers - base and nullable
        self._DTYPES_INT_BASE = [np.int8, np.int16, np.int32, np.int64]
        self._DTYPES_INT_NULLABLE = [pd.Int8Dtype(), pd.Int16Dtype(), pd.Int32Dtype(), pd.Int64Dtype()]
        self._LIMIT_LOW_INT = [np.iinfo(i).min for i in self._DTYPES_INT_BASE]
        self._LIMIT_HIGH_INT = [np.iinfo(i).max for i in self._DTYPES_INT_BASE]
        # floats - nullable by default
        self._DTYPES_FLOAT = [np.float16, np.float32, np.float64]
        # datetime - nullable by default
        self._DTYPE_DATETIME = np.datetime64
        # string
        self._DTYPE_STRING = pd.StringDtype()
        # categorical - nullable by default
        self._DTYPE_CATEGORICAL = pd.CategoricalDtype()

    @staticmethod
    def _validate(obj):
        # any necessary validations here (raise AttributeErrors, etc)
        # todo check isinstance(df, pd.DataFrame) and/or df.empty?
        pass

    # todo add other methods
    """
    automate data profiling
    - pandas_profiling
    - missingo
    - any others?
    unit handling
    - column unit attributes
    - unit conversion
    - column descriptions  
    automate machine learning pre-processing
    - imputation
    - scaling
    - encoding
    """

    def simplify_columns(self):
        # todo add any other needed simplifications
        # get columns
        cols = self._obj.columns.astype('str')
        # replace punctuation and whitespace with underscore
        chars = re.escape(string.punctuation)
        cols = [re.sub(r'[' + chars + ']', '_', col) for col in cols]
        cols = ['_'.join(col.split('\n')) for col in cols]
        cols = [re.sub('\s+', '_', col) for col in cols]
        # drop multiple underscores to a single one
        cols = [re.sub('_+', '_', col) for col in cols]
        # remove trailing or leading underscores
        cols = [col[1:] if col[0] == '_' else col for col in cols]
        cols = [col[:-1] if col[-1] == '_' else col for col in cols]
        # convert to lower case
        cols = [col.lower() for col in cols]
        # reassign column names
        self._obj.columns = cols

    def _minimize_memory_col_int(self, col):
        # get range of values
        val_min = self._obj[col].min()
        val_max = self._obj[col].max()
        # check whether signed or unsigned
        bool_signed = val_min < 0
        # check for null values
        bool_null = np.any(pd.isna(self._obj[col]))
        # get conversion lists
        if bool_signed:
            val_bins_lower = self._LIMIT_LOW_INT
            val_bins_upper = self._LIMIT_HIGH_INT
            if bool_null:
                val_dtypes = self._DTYPES_INT_NULLABLE
            else:
                val_dtypes = self._DTYPES_INT_BASE
        else:
            val_bins_lower = self._LIMIT_LOW_UINT
            val_bins_upper = self._LIMIT_HIGH_UINT
            if bool_null:
                val_dtypes = self._DTYPES_UINT_NULLABLE
            else:
                val_dtypes = self._DTYPES_UINT_BASE
        # apply conversions
        idx = max(np.where(np.array(val_bins_lower) <= val_min)[0][0],
                  np.where(np.array(val_bins_upper) >= val_max)[0][0])
        self._obj[col] = self._obj[col].astype(val_dtypes[idx])

    def _minimize_memory_col_float(self, col, tol):
        if np.sum(self._obj[col] - self._obj[col].apply(lambda x: round(x, 0))) == 0:
            # check if they are actually integers (no decimal values)
            self._minimize_memory_col_int(col)
        else:
            # find the smallest float dtype that has an error less than the tolerance
            for i_dtype in self._DTYPES_FLOAT:
                if np.abs(self._obj[col] - self._obj[col].astype(i_dtype)).max() <= tol:
                    self._obj[col] = self._obj[col].astype(i_dtype)
                    break

    def reduce_memory_usage(self, tol_float=1E-6, category_fraction=0.5, drop_null_cols=True, drop_null_rows=True,
                            reset_index=False, print_reduction=False, print_warnings=True):
        # get the starting memory usage - optional because it can add significant overhead to run time
        if print_reduction:
            mem_start = self._obj.memory_usage(deep=True).values.sum()

        # null value handling
        # apply conversions for null values
        self._obj.replace(self._NULL_VALS, pd.NA, inplace=True)
        # drop null columns and rows
        if drop_null_cols:
            self._obj.dropna(axis=1, how='all', inplace=True)
        if drop_null_rows:
            self._obj.dropna(axis=0, how='all', inplace=True)

        # replace boolean-like strings with booleans
        self._obj.replace(self._BOOL_MAP_DICT, inplace=True)

        # loop by column to predict value
        for i_col, i_dtype in self._obj.dtypes.to_dict().items():
            # skip if column is ful of nulls and wasn't dropped
            if not drop_null_cols:
                if np.all(pd.isna(self._obj[i_col])):
                    continue

            # get non-null values and the unique modules
            vals_not_null = self._obj.loc[pd.notna(self._obj[i_col]), i_col].values
            modules = np.unique([type(val).__module__.split('.')[0] for val in vals_not_null])

            # skip if col contains non-supported modules
            if np.any([val not in self._INCORPORATED_MODULES for val in modules]):
                continue

            # check if any null values are present
            null_vals_present = np.any(pd.isna(self._obj[i_col]))

            # check and  assign dtypes
            # todo add option to coerce small number of values and still proceed with dtype application
            if pd.isna(pd.to_numeric(vals_not_null, errors='coerce')).sum() == 0:
                # numeric dtype
                self._obj[i_col] = pd.to_numeric(self._obj[i_col], errors='coerce')
                vals_not_null = self._obj.loc[pd.notna(self._obj[i_col]), i_col].values
                # check if bool, int, or float
                if np.all(np.logical_or(vals_not_null == 0, vals_not_null == 1)):
                    # boolean
                    if null_vals_present:
                        self._obj[i_col] = self._obj[i_col].astype(self._DTYPE_BOOL_NULLABLE)
                    else:
                        self._obj[i_col] = self._obj[i_col].astype(self._DTYPE_BOOL_BASE)
                else:
                    # apply float, will use int if possible
                    self._minimize_memory_col_float(i_col, tol_float)
            elif pd.isna(pd.to_datetime(vals_not_null, errors='coerce')).sum() == 0:
                # datetime
                # todo add option to split datetime into year col, month col, and day col
                self._obj[i_col] = pd.to_datetime(self._obj[i_col], errors='coerce')
            else:
                # get types
                val_types = np.unique([str(val.__class__).split("'")[1] for val in vals_not_null])
                # check if there are any non-string iterables
                bool_iters = np.any([False if (str(val.__class__).split("'")[1] == 'str') else
                                     (True if hasattr(val, '__iter__') else False) for val in vals_not_null])
                # check if any are strings
                if 'str' in val_types and ((len(val_types) == 1) or not bool_iters):
                    # convert to strings
                    if len(val_types) != 1:
                        self._obj.loc[pd.notna(self._obj[i_col]), i_col] = self._obj.loc[
                            pd.notna(self._obj[i_col]), i_col].apply(lambda x: str(x))
                    # check for value repetition
                    vals_not_null = self._obj.loc[pd.notna(self._obj[i_col]), i_col].values
                    if len(np.unique(vals_not_null)) / len(vals_not_null) <= category_fraction:
                        # todo report pandas bug - categorical requires nan, doesn't work with pd.NA
                        if null_vals_present:
                            self._obj.loc[pd.isna(self._obj[i_col]), i_col] = np.nan
                        self._obj[i_col] = self._obj[i_col].astype(self._DTYPE_CATEGORICAL)
                    else:
                        self._obj[i_col] = self._obj[i_col].astype(self._DTYPE_STRING)
                else:
                    if print_warnings:
                        warnings.warn('Handling for columns with variable types "%s" not implemented' %
                                      (str(val_types)), RuntimeWarning)

        # reset index
        if reset_index:
            self._obj.reset_index(drop=True, inplace=True)

        # get the ending memory usage and output the reduction
        if print_reduction:
            # get end memory
            mem_end = self._obj.memory_usage(deep=True).values.sum()
            # check nearest size increment
            for i_mem_bin, i_mem_str in zip([1E9, 1E6, 1E3, 1], ['GB', 'MB', 'KB', 'B']):
                if (mem_start / i_mem_bin > 1) and (mem_end / i_mem_bin > 1):
                    mem_bin = i_mem_bin
                    mem_str = i_mem_str
                    break
            print('Dataframe memory reduction results:\n'
                  '    Starting memory usage (%s): %.1f\n'
                  '    Ending memory usage (%s): %.1f\n'
                  '    Memory reduction (%%): %.1f'
                  % (mem_str, mem_start / mem_bin, mem_str, mem_end / mem_bin, 100 * (mem_start - mem_end) / mem_start))
        return self._obj
