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
        # assign dtypes and limits
        # todo check if there is any benefit to non-nullable dtype (less space than nullable dtype?)
        self._INT_UNSIGNED_BINS_LOWER = [0, 0, 0, 0]
        self._INT_UNSIGNED_BINS_UPPER = [255, 65535, 4294967295, 18446744073709551615]
        self._INT_SIGNED_BINS_LOWER = [-128, -32768, -2147483648, -9223372036854775808]
        self._INT_SIGNED_BINS_UPPER = [127, 32767, 2147483647, 9223372036854775807]
        self._INT_UNSIGNED_NOT_NULLABLE_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
        self._INT_UNSIGNED_NULLABLE_DTYPES = [pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype()]
        self._INT_SIGNED_NOT_NULLABLE_DTYPES = [np.int8, np.int16, np.int32, np.int64]
        self._INT_SIGNED_NULLABLE_DTYPES = [pd.Int8Dtype(), pd.Int16Dtype(), pd.Int32Dtype(), pd.Int64Dtype()]
        self._FLOAT_DTYPES = [np.float16, np.float32, np.float64]
        # assign incorporated modules
        self._INCORPORATED_MODULES = ['builtins', 'numpy', 'pandas']
        # assign potential boolean strings (lower case)
        self._BOOL_STRINGS_TRUE = ['t', 'true', 'yes', 'on']
        self._BOOL_STRINGS_FALSE = ['f', 'false', 'no', 'off']
        self._BOOL_MAP_DICT = {val: True for val in self._BOOL_STRINGS_TRUE}.update(
            {val: False for val in self._BOOL_STRINGS_FALSE})
        # potential null values
        self._NULL_VALS = [None, np.nan, 'np.nan', 'nan', np.inf, 'np.inf', 'inf', -np.inf, '-np.inf', '', 'n/a', 'na',
                           'N/A', 'NA', 'unknown', 'unk', 'UNKNOWN', 'UNK']

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
            val_bins_lower = self._INT_SIGNED_BINS_LOWER
            val_bins_upper = self._INT_SIGNED_BINS_UPPER
            if bool_null:
                val_dtypes = self._INT_SIGNED_NULLABLE_DTYPES
            else:
                val_dtypes = self._INT_SIGNED_NOT_NULLABLE_DTYPES
        else:
            val_bins_lower = self._INT_UNSIGNED_BINS_LOWER
            val_bins_upper = self._INT_UNSIGNED_BINS_UPPER
            if bool_null:
                val_dtypes = self._INT_UNSIGNED_NULLABLE_DTYPES
            else:
                val_dtypes = self._INT_UNSIGNED_NOT_NULLABLE_DTYPES
        # apply conversions
        idx = max(np.where(np.array(val_bins_lower) <= val_min)[0][0],
                  np.where(np.array(val_bins_upper) >= val_max)[0][0])
        self._obj[col] = self._obj[col].astype(val_dtypes[idx])

    def _minimize_memory_col_float(self, col, tol=1E-9):
        if np.sum(self._obj[col] - self._obj[col].apply(lambda x: round(x, 0))) == 0:
            # check if they are actually integers (no decimal values)
            self._minimize_memory_col_int(col)
        else:
            # find the smallest float dtype that has an error less than the tolerance
            for i_dtype in self._FLOAT_DTYPES:
                if np.abs(self._obj[col] - self._obj[col].astype(i_dtype)).max() <= tol:
                    self._obj[col] = self._obj[col].astype(i_dtype)
                    break

    def reduce_memory_usage(self, tol_float=1E-9, drop_null_cols=True, drop_null_rows=True, print_reduction=False):
        # get the starting memory usage
        if print_reduction:
            mem_start = self._obj.memory_usage(deep=True).values.sum()
        # reset index
        # todo make reset optional, need to adjust indexing below
        self._obj.reset_index(drop=True, inplace=True)
        # convert nulls
        self._obj.replace(self._NULL_VALS, pd.NA, inplace=True)
        # drop null columns and rows
        if drop_null_cols:
            self._obj = self._obj.loc[:, np.array(self._obj.columns)[np.any(pd.notna(self._obj), axis=0)]]
        if drop_null_rows:
            self._obj = self._obj.iloc[np.where(np.any(pd.notna(self._obj), axis=1))[0], :]
            # reset index
            # todo make reset optional, need to adjust indexing below
            self._obj.reset_index(drop=True, inplace=True)
        # loop by column to convert dtypes
        for col, dtype in self._obj.dtypes.to_dict().items():
            # get non-null values
            vals_not_null = self._obj.loc[pd.notna(self._obj[col]), col].values
            # get the modules within the column
            col_modules = np.unique([type(val).__module__.split('.')[0] for val in vals_not_null])
            # apply type rules
            if np.any([val not in self._INCORPORATED_MODULES for val in col_modules]):
                # custom objects or objects from other libraries
                pass
            elif len(vals_not_null) == 0:  # all null column
                pass
            elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating) or \
                    pd.isna(pd.to_numeric(vals_not_null, errors='coerce')).sum() == 0:
                # todo allow sum to be non-zero: optional argument for allowable cell loss?, could print out lost val
                vals_not_null = pd.to_numeric(vals_not_null, errors='coerce')
                self._obj[col] = pd.to_numeric(self._obj[col], errors='coerce')
                # numeric dtype
                if np.all(np.logical_or(vals_not_null == 0, vals_not_null == 1)):  # boolean
                    if len(pd.isna(self._obj[col])) > 0:
                        # nullable boolean
                        self._obj[col] = self._obj[col].astype(pd.BooleanDtype())
                    else:
                        # non-nullable boolean
                        self._obj[col] = self._obj[col].astype(np.bool)
                elif np.issubdtype(dtype, np.integer):  # integer
                    self._minimize_memory_col_int(col)
                else:  # float, will try to convert to integer as well
                    self._minimize_memory_col_float(col, tol=tol_float)
            elif pd.isna(pd.to_datetime(vals_not_null, errors='coerce')).sum() == 0:
                # todo allow sum to be non-zero: optional argument for allowable cell loss?, could print out lost val
                # datetime object
                self._obj[col] = pd.to_datetime(self._obj[col], errors='coerce')
                # todo downcast datetime size if possible without loss
            else:  # non-datetime, non-numeric, builtin dtype (object, etc)
                # todo implement other dtypes - can look to pandas_profiling dtype handling
                # todo separate dtype checking functions (eg all bool in one spot)
                # todo custom dtypes with pandas? e.g. shapely dtype for pandas?
                # get types
                ex_types = np.unique([str(val.__class__).split("'")[1] for val in vals_not_null])
                # get values and counts
                if 'str' in ex_types and len(ex_types) > 1:
                    self._obj.loc[pd.notna(self._obj[col]), col] = self._obj.loc[pd.notna(self._obj[col]), col].apply(
                        lambda x: str(x))
                    vals_not_null = self._obj.loc[pd.notna(self._obj[col]), col].values
                ex_vals, ex_val_counts = np.unique(vals_not_null, return_counts=True)
                # check dtypes
                if np.all([val in self._BOOL_STRINGS_TRUE or val in self._BOOL_STRINGS_FALSE for val in ex_vals]):
                    # boolean
                    vals_col = self._obj[col].map(self._BOOL_MAP_DICT)
                    if len(pd.isna(vals_col)) > 0:
                        # nullable boolean
                        self._obj[col] = vals_col.astype(pd.BooleanDtype())
                    else:
                        # non-nullable boolean
                        self._obj[col] = vals_col.astype(np.bool)
                elif np.all([isinstance(val, str) for val in ex_vals]):  # all strings
                    if np.any(ex_val_counts > 1):  # categorical
                        # temporary workaround - bug with categorical support for NA - using nan instead
                        # todo update once pandas bug is fixed
                        self._obj.loc[pd.isna(self._obj[col]), col] = np.nan
                        self._obj[col] = self._obj[col].astype(pd.CategoricalDtype())
                    else:  # string
                        self._obj[col] = self._obj[col].astype(pd.StringDtype())
                else:
                    # unknown
                    pass
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
