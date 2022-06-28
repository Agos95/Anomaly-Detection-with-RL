# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %%


class TSDataset(object):

    def __init__(self, fname, resample="5min", fillna=True, columns=None, window="1H", stride="15min", label_type="last"):
        """
        Creates a dataset object from the given time series data.

        Parameters
        ----------
        fname : str or Path-like
            Path to data.
        resample : str or pandas offset object
            frequency to use to resample data.
            Must be one of pandas date offset strings or corresponding objects.
        fillna : bool, default True
            If True, fill NaNs with 0 after resampling.
        columns : list of str, optional
            List of columns to use. Default is `None`, which means all columns.
        window : str or pandas offset object, default "1H"
            Time window to use to create timeseries from the dataset.
            Must be one of pandas date offset strings or corresponding objects.
        stride : str or pandas offset object, default="15min"
            Offset between two consecutive timeseries.
            Must be one of pandas date offset strings or corresponding objects.
        label_type : str, default "last"
            Must be one of `["last", "all"]`.
            If `last`, the time series is labelled looking only at the last timestep.
            If `all`, all timepoints in the timeseries are labelled.
        """

        # read data
        cols_names = [
            "overalltime", "overallusers", "overallrbdw", "overallrbdwmean", "overallrbdwstd",
            "overallratedw", "overallratedwmean", "overallratedwstd", "overallmsgdw",
            "overallretxdw", "overallrbup", "overallrbupmean", "overallrbupstd", "overallrateup",
            "overallrateupmean", "overallrateupstd", "overallmsgup", "overallretxup"
        ]
        self.df = pd.read_csv(fname, header=0, names=cols_names)
        self.df = self.df.set_index(pd.to_datetime(self.df["overalltime"], origin=pd.Timestamp(
            "29/06/2016 00:00:00"), unit="s")).drop(columns=["overalltime"])
        self.original_df = self.df.copy()

        # resample dataset
        self.resample = resample
        if self.resample is not None:
            self.df = self.df.resample(self.resample).mean()
            if fillna:
                self.df.fillna(0, inplace=True)

        # select requested columns
        if columns is not None:
            self.df = self.df[columns]

        # known anomalies in the dataset
        self.anomaly = pd.DatetimeIndex(
            pd.date_range(start="2016/07/03 09:00:00",
                          end="2016/07/03 16:00:00", freq=self.resample).tolist() +
            pd.date_range(start="2016/07/10 09:00:00",
                          end="2016/07/10 16:00:00", freq=self.resample).tolist() +
            pd.date_range(start="2016/07/17 09:00:00",
                          end="2016/07/17 16:00:00", freq=self.resample).tolist() +
            pd.date_range(start="2016/07/24 09:00:00",
                              end="2016/07/24 16:00:00", freq=self.resample).tolist())

        # make X, y dataset
        if label_type not in ["last", "all"]:
            print("label must be one of ['last', 'all']. Setting to 'last'.")
            label_type = "last"
        self.label_type = label_type
        self.window = pd.Timedelta(window)
        self.stride = pd.Timedelta(stride)
        self.X, self.y = make_sliding_windows(
            self.df, self.window, self.stride, self.anomaly, self.label_type)

    def train_test_split(self, test_size=.25, force_sequential=False, scaler=None, seed=None):
        """
        Split the dataset into train/test.
        If `self.label_type=="last"` and `force_sequential` is `False`,
        the split is done in a stratify way looking at the labels.
        Otherwise, the split is sequential, with the last `test_size` entries
        given to the test set.

        Parameters
        ----------
        test_size : float, default .25
            Fraction of the dataset to be assigned to the test set.
        force_sequential : bool, default False
            If True force a sequential slit even if `self.label_type=="last"`
        scaler : sklearn scaler, optional
            Sklearn sclaer to use to preprocess data.
        seed : int, optional
            Seed for splitting if using the stratified one.

        Returns
        -------
        X_train, X_test, y_train, y_test : array
            Train/Test split.
        """
        if self.label_type == "last" and not force_sequential:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=seed)
        else:
            tot = len(self.X)
            n = int(tot * (1 - test_size))
            X_train, X_test = self.X[:n], self.X[n + 1:]
            y_train, y_test = self.y[:n], self.y[n + 1:]

        if scaler is not None:
            X_train_shape, X_test_shape = X_train.shape, X_test.shape
            X_train = X_train.reshape(-1, X_train_shape[-1])
            X_test = X_test.reshape(-1, X_test_shape[-1])

            scaler.fit(X_train)
            X_train = scaler.transform(X_train).reshape(X_train_shape)
            X_test = scaler.transform(X_test).reshape(X_test_shape)

        return X_train, X_test, y_train, y_test


# %%

def make_sliding_windows(df, window, stride, anomaly=None, label_type="last"):
    """
    Create sliding windows from a DataFrame and optionally assigns a label based on `anomaly`.
    The label is given looking at the last timestamp of each time series.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset with time index.
    window : pandas Timedelta
        Size of the temporal window.
    stride : pandas Timedelta
        Stride between consecutive temporal windows.
    anomaly : list-like, optional
        List of Timestamps corresponding to anomalies.
    label_type : str, default "last"
        Must be one of `["last", "all"]`.
        If `last`, the time series is labelled looking only at the last timestep.
        If `all`, all timepoints in the timeseries are labelled.

    Returns
    -------
    X : array of shape `(N, n_points, n_features)`
        Numpy array, where:
        - `N` is the number of timeseries
        - `n_points` is the number of timesteps in each timeseries
        - `n_features` is the number of feature in the timeseries

    y : array of shape `(N,)` or `(N, n_points)`
        Numpy array of shape `(N,)` is `label_type=="last"`, or `(N, n_points)` if `label=="all"`.

    """

    start = df.index.min()
    end = df.index.max()

    t0 = start
    t1 = start + window

    X, y = [], []
    while t1 < end:
        X.append(df.loc[t0:t1, :])
        t0 = t0 + stride
        t1 = t0 + window

    if anomaly is not None:
        if label_type == "last":
            for df in X:
                last = df.index.max()
                an = 1 if last in anomaly else 0
                y.append(an)
        elif label_type == "all":
            for df in X:
                an = pd.Series(0, index=df.index)
                an.loc[an.index.intersection(anomaly)] = 1
                y.append(an.values)
        else:
            pass

    X = np.stack([df.values for df in X])
    y = np.stack(y)

    return X, y


# %%
if __name__ == "__main__":
    data = TSDataset("data/rastro.tar.gz", resample="15min",
                     window="3H", stride="15min", label_type="all")
