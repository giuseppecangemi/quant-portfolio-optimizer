"""ECB overnight fixings and historical EONIA accrual for EUR factor samples."""
from functools import lru_cache
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, EasterMonday, GoodFriday, Holiday
from pandas.tseries.offsets import CustomBusinessDay

ESTR_ONLY = "ESTR_ONLY"
EONIA_ESTR = "EONIA_ESTR"
FIRST_ESTR_REFERENCE = pd.Timestamp("2019-10-01")
FIRST_ESTR_PUBLICATION = pd.Timestamp("2019-10-02")
FIRST_SUPPORTED_EONIA = pd.Timestamp("2007-01-02")
EONIA_KEY = "EON/D.EONIA_TO.RATE"
ESTR_KEY = "EST/B.EU000A2X2A25.WT"


class TargetHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1), GoodFriday, EasterMonday,
        Holiday("Labour Day", month=5, day=1),
        Holiday("Christmas", month=12, day=25),
        Holiday("Boxing Day", month=12, day=26),
    ]


TARGET_DAY = CustomBusinessDay(calendar=TargetHolidays())


def validate_mode(mode):
    if mode not in (ESTR_ONLY, EONIA_ESTR):
        raise ValueError(f"Unknown EUR risk-free methodology: {mode}")


@lru_cache(maxsize=1)
def load_eonia_rates():
    """Cache validated ECB EONIA observations for the life of the process."""
    url = f"https://data-api.ecb.europa.eu/service/data/{EONIA_KEY}?format=csvdata"
    request = Request(url, headers={"Accept": "text/csv", "User-Agent": "Mozilla/5.0"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=25) as response:
                data = pd.read_csv(BytesIO(response.read()))
            break
        except (URLError, TimeoutError) as exc:
            temporary = not isinstance(exc, HTTPError) or exc.code in (429, 500, 502, 503, 504)
            if attempt or not temporary:
                raise ValueError(f"ECB EONIA {EONIA_KEY} download failed: {exc}") from exc
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid ECB EONIA CSV for {EONIA_KEY}: {exc}") from exc
    required = {"TIME_PERIOD", "OBS_VALUE", "KEY", "UNIT", "UNIT_MULT"}
    if (not required.issubset(data.columns) or not data["KEY"].eq("EON.D.EONIA_TO.RATE").all()
            or not data["UNIT"].eq("PC").all() or not data["UNIT_MULT"].eq(0).all()):
        raise ValueError(f"Unexpected ECB response for {EONIA_KEY}")
    dates = pd.to_datetime(data["TIME_PERIOD"], format="%Y-%m-%d", errors="raise")
    rates = pd.to_numeric(data["OBS_VALUE"], errors="raise")
    observations = pd.Series(rates.to_numpy(), index=dates, name="annual_percent").sort_index()
    observations = observations.loc[FIRST_SUPPORTED_EONIA:FIRST_ESTR_REFERENCE - pd.Timedelta(days=1)]
    if observations.empty or not observations.index.is_unique or not np.isfinite(observations.dropna()).all():
        raise ValueError(f"Invalid observations in ECB EONIA {EONIA_KEY}")
    return observations.dropna()


def fixing_metadata(rates, source):
    """Scheduled publication in Frankfurt local time; ECB CSV has no release vintage."""
    dates = pd.DatetimeIndex(rates.index)
    if source == EONIA_KEY:
        publication = dates + pd.Timedelta(hours=19)
    elif source == ESTR_KEY:
        publication = pd.DatetimeIndex([date + TARGET_DAY for date in dates]) + pd.Timedelta(hours=8)
    else:
        raise ValueError(f"Unsupported ECB overnight series: {source}")
    return pd.DataFrame({
        "annual_percent": rates.to_numpy(), "published_at": publication.tz_localize("Europe/Berlin"),
        "source": source,
    }, index=dates.rename("reference_date"))


def eonia_interval_returns(starts, *, rates=None):
    """Accrue observable EONIA fixings over each historical factor interval.

    Each TARGET fixing covers calendar days until the next TARGET session.
    Cross-fixing intervals compound the ACT/360 legs. A missing fixing invalidates
    its whole leg. Evaluation dates are local midnight, before that day's fixing.
    """
    if starts.dropna().empty:
        return pd.Series(np.nan, index=starts.index)
    rates = load_eonia_rates() if rates is None else rates
    first = max(FIRST_SUPPORTED_EONIA, starts.min() - pd.Timedelta(days=7))
    last = min(FIRST_ESTR_REFERENCE - pd.Timedelta(days=1), starts.index.max())
    target = pd.date_range(first, last, freq=TARGET_DAY)
    if target.empty:
        return pd.Series(np.nan, index=starts.index)
    observed = fixing_metadata(rates.reindex(target), EONIA_KEY)
    valid = starts.dropna()
    valid = valid.loc[(valid >= target[0]) & (valid.index <= FIRST_ESTR_REFERENCE)
                      & (valid < valid.index)]
    active = target.searchsorted(valid, side="right") - 1
    terminal = target.searchsorted(valid.index) - 1
    quotes = observed["annual_percent"].to_numpy()
    published = observed["published_at"].array
    endpoints = valid.index.tz_localize("Europe/Berlin")
    days = (valid.index - pd.DatetimeIndex(valid)).days
    result = pd.Series(np.nan, index=starts.index)
    single = (active == terminal) & (published[active] < endpoints)
    result.loc[valid.index[single]] = quotes[active[single]] / 100 * days[single] / 360
    for i in np.flatnonzero(active < terminal):
        legs = slice(active[i], terminal[i] + 1)
        if (published[legs] >= endpoints[i]).any():
            continue
        boundaries = pd.DatetimeIndex([valid.iloc[i], *target[active[i] + 1:terminal[i] + 1], valid.index[i]])
        accrual_days = np.diff(boundaries.to_numpy()).astype("timedelta64[D]").astype(int)
        result.loc[valid.index[i]] = np.prod(1 + quotes[legs] / 100 * accrual_days / 360) - 1
    return result
