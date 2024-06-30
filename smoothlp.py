import numpy as np

def bspline(
        x: np.ndarray, 
        xl: int, 
        xr: int, 
        ndx: int, 
        bdeg: int
    ) -> np.ndarray:
    """Generate B-splines.
    
    Args:
        x: Horizon range.
        xl: Minimum horizon.
        xr: Maximum horizon.
        ndx: Number of splines.
        bdeg: Degree of spline.
    
    Returns:
        Matrix of B-splines.
    """
    dx = (xr - xl) / ndx
    t = xl + dx * np.arange(-bdeg, ndx)
    T = np.ones((len(x), 1)) * t
    X = x * np.ones((1, len(t)))
    P = (X - T) / dx
    
    B = (T <= X) & (X <= (T + dx))
    r = np.append(np.arange(1, len(t)), 0)

    for k in range(1, bdeg + 1):
        B = (P * B + (k + 1 - P) * B[:, r]) / k

    return B
  
def lagmat(x: np.ndarray, lag: int) -> np.ndarray:
    """Lag a matrix.
    
    This function creates a lagged version of the input matrix.
    
    Args:
        x: Matrix of data. Can be 1D or 2D numpy array.
        lag: Number of lags to create.
    
    Returns:
        Lagged matrix. The output will have the same number of rows as the input,
        but with lag * k columns, where k is the number of columns in the input.
        The first 'lag' rows will contain NaN values.
    
    Note:
        If the input is a 1D array, it will be reshaped to a 2D array with one column.
    """
    assert len(x.shape) <= 2
    #assert lag > 0

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    t, k = x.shape
    out = np.full((t, lag * k), np.nan)

    for i in range(lag):
        lag_array = x[:-(i+1), :]
        out[(i+1):, (i * k):((i + 1) * k)] = lag_array

    return out
  
def lproj(
        y: np.ndarray,
        x: np.ndarray,
        w: np.ndarray=None,
        style: str="reg",
        H: int=12,
        h1: int=0,
        r: int=0,
        zero: bool=False,
        lam: list[float]=[0]
) -> dict:
    """Run local projections.
    
    This function performs local projections, which is a method for estimating
    impulse response functions.
    
    Args:
        y: Dependent variable (1D numpy array).
        x: Independent variable (1D numpy array).
        w: Additional control variables (2D numpy array, optional).
        style: Projection style, either "reg" for regular or "smooth" for smoothed (default: "reg").
        H: Maximum horizon for projections (default: 12).
        h1: Minimum horizon for projections (default: 0).
        r: Order of difference for smoothing penalty (default: 0).
        zero: Whether to impose zero constraint at the end of the horizon for smoothing (default: False).
        lam: List of smoothing parameters (default: [0]).
    
    Returns:
        A dictionary containing:
        - _y: Stacked dependent variable.
        - _x: Stacked independent variables.
        - theta: Estimated coefficients.
        - ir: Impulse responses.
        - lam: Smoothing parameters.
        - ts: Number of observations.
        - p: Penalty matrix.
        - H: Maximum horizon.
        - h1: Minimum horizon.
        - style: Projection style.
        - t: Number of time periods.
        - mul: Multipliers.
        - basis: B-spline basis (if style is "smooth").
        - xs: Number of coefficients.
        - idx: Index of observations.
    """
    t = len(y)

    assert t > H

    basis = None

    if style == "smooth":
        h_range = np.arange(h1, H + 1).reshape(-1, 1)
        bdeg = 3
        basis = bspline(h_range, h1, H + 1, H + 1, bdeg)[:, :-1]
    #print(basis.shape)
    if w is not None:
        w = np.hstack([np.ones((t, 1)), w])
    else:
        w = np.ones((t, 1))
    
    #delta = np.std(x)

    hr = int(H + 1 - h1)
    ts = t * hr

    if style == "reg":
        xs = hr
    else:
        xs = basis.shape[1]

    #ws = hr
    nw = w.shape[1]

    idx = np.zeros((ts, 2))
    _y = np.full((ts, 1), np.nan)
    _xb = np.full((ts, xs), np.nan)
    _xc = np.full((ts, hr, nw), np.nan)

    for i in range(t - h1):
        idx_beg = i * hr
        idx_end = (i + 1) * hr
        idx[idx_beg:idx_end, 0] = i
        idx[idx_beg:idx_end, 1] = np.arange(h1, H + 1)

        y_range = np.arange(i + h1, min(i + H + 1, t))

        # if i == 0:
        #     y_lag = np.nan
        # else:
        #     y_lag = y[i - 1]
        _y[idx_beg:idx_end, 0] = np.append(y[y_range], np.full((hr - len(y_range)), np.nan))
        #print(y_lag)

        if style == "reg":
            _xb[idx_beg:idx_end, :] = np.eye(hr) * x[i]
        else:
            _xb[idx_beg:idx_end, :] = basis * x[i]

        for j in range(nw):
            _xc[idx_beg:idx_end, :, j] = np.eye(hr) * w[i, j]
    
    _x = _xb
    for j in range(nw):
        _x = np.hstack([_x, _xc[:, :, j]])

    sel = ~np.isnan(_y).reshape(-1)
    idx = idx[sel, :]
    _y = _y[sel]
    _x = _x[sel, :]
    ts = len(_y)
    xx = np.dot(_x.T, _x)
    xy = np.dot(_x.T, _y)
    p = np.zeros((_x.shape[1], _x.shape[1]))
    #print(xx)

    if style == "smooth":
        D = np.eye(xs)

        for _ in range(r):
            D = np.diff(D, axis=0)

        if zero:
            DP = np.zeros(xs)
            DP[xs - 1] = 1
            D = np.vstack([D, DP])
        
        p[:xs, :xs] = np.dot(D.T, D)

    ir = np.zeros((H + 1, len(lam)))
    theta = np.zeros((_x.shape[1], len(lam)))
    mul = np.zeros((hr, len(lam)))

    for j in range(len(lam)):
        a = xx + lam[j] * ts * p
        b = xy.reshape(-1)
        theta[:, j] = np.linalg.solve(a, b)

        if style == "reg":
            mul[:, j] = theta[:xs, j]
        else:
            beta = theta[:xs, j]
            mul[:, j] = np.dot(basis, beta)

        ir[h1:(H +1), j] = mul[:, j]

    return {
        "_y": _y,
        "_x": _x,
        "theta": theta,
        "ir": ir,
        "lam": lam,
        "ts": ts,
        "p": p,
        "H": H,
        "h1": h1,
        "style": style,
        "t": t,
        "mul": mul,
        "basis": basis,
        "xs": xs,
        "idx": idx
    }
    
def lproj_cv(obj: dict, K: int) -> dict:
    """Perform cross-validation for local projections.
    
    This function performs K-fold cross-validation on the local projections
    to select the optimal smoothing parameter.
    
    Args:
        obj: Dictionary containing the results from the lproj function.
        K: Number of folds for cross-validation.
    
    Returns:
        A dictionary containing:
        - rss: Residual sum of squares for each smoothing parameter.
        - idx_opt: Index of the optimal smoothing parameter.
        - ir_opt: Optimal impulse response.
    """
    t = obj["t"]
    ll = len(obj["lam"])
    ind = np.ceil(obj["idx"][:, 0] / t * K).astype(int)
    rss = np.zeros(ll)
    #print(ind)

    for i in range(ll):
        rss_l = np.zeros(K)

        for j in range(K):
            y_in = obj["_y"][ind != j]
            x_in = obj["_x"][ind != j, :]
            y_out = obj["_y"][ind == j]
            x_out = obj["_x"][ind == j, :]
            a = np.dot(x_in.T, x_in) + obj["lam"][i] * obj["ts"] * ((K - 1) / K) * obj["p"]
            b = np.dot(x_in.T, y_in)
            beta = np.linalg.solve(a, b)
            #print(obj["_y"][ind == j])
            rss_l[j] = np.mean((y_out - np.dot(x_out, beta)) ** 2)
            #print(rss_l[j])

        #print(rss_l[0])
        rss[i] = np.nanmean(rss_l)
    
    return {
        "rss": rss,
        "idx_opt": np.argmin(rss),
        "ir_opt": obj["ir"][:, np.argmin(rss)]
    }

def lproj_conf(obj: dict, lam: int = 0) -> dict:
    """Calculate confidence intervals for local projections.
    
    This function computes confidence intervals for the impulse responses
    estimated by the local projections method.
    
    Args:
        obj: Dictionary containing the results from the lproj function.
        lam: Index of the smoothing parameter to use (default: 0).
    
    Returns:
        A dictionary containing:
        - se: Standard errors of the impulse responses.
        - conf: 80% confidence intervals for the impulse responses.
        - irc: Confidence intervals including NaN values for periods before h1.
    """
    import scipy.stats as stats

    u = obj["_y"] - np.dot(obj["_x"], obj["theta"][:, lam].reshape(-1, 1))
    s = obj["_x"] * np.dot(u, np.ones((1, obj["_x"].shape[1])))
    
    bread = np.linalg.inv(obj["_x"].T @ obj["_x"] + obj["lam"][lam] * obj["ts"] * obj["p"])

    nlag = obj["H"]
    weights = np.arange(0, nlag + 1)[::-1] / (nlag)
    v = np.dot(s.T, s)

    for i in range(nlag):
        gamma = np.dot(s[i + 1:obj["t"], :].T, s[:obj["t"] - i - 1, :])
        gpp = gamma + gamma.T
        v += weights[i+1] * gpp

    meat = v

    v = np.dot(bread, np.dot(meat, bread))

    if obj["style"] == "reg":
        se = np.sqrt(np.diag(v[:obj["xs"], :obj["xs"]]))
        conf = np.zeros((len(se), 2))
        conf[:, 0] = obj["mul"][:, lam] + se * stats.norm.ppf(0.10)
        conf[:, 1] = obj["mul"][:, lam] + se * stats.norm.ppf(0.90)
    else:
        v = np.dot(obj["basis"], v[:obj["xs"], :obj["xs"]])
        se = np.sqrt(np.diag(v))
        conf = np.zeros((len(se), 2))
        conf[:, 0] = obj["mul"][:, lam] + se * stats.norm.ppf(0.10)
        conf[:, 1] = obj["mul"][:, lam] + se * stats.norm.ppf(0.90)

    irc = np.full((obj["H"] + 1, 2), np.nan)
    irc[obj["h1"]:obj["H"] + 1, :] = conf

    return {
        "se": se,
        "conf": conf,
        "irc": irc
    }   
