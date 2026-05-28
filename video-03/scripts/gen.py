# Copyright (c) Hebes Intelligence Private Company

import numpy as np
import pandas as pd

FRIDAY = 4


def ar1_noise(n: int, phi: float = 0.85, sigma: float = 0.15, seed=None):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)

    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]

    return x


def rc_temperature(T, tau_hours=2):
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    out[0] = T[0]

    alpha = 1 / tau_hours
    for i in range(1, len(T)):
        out[i] = out[i - 1] + alpha * (T[i] - out[i - 1])

    return out


def create_heating_on(
    temperature: pd.Series,
    heating_schedule: dict = None,
    threshold_val: float = 16,
    threshold_dur: int = 1,
    exclude_weekends: bool = True,
) -> pd.Series:
    schedule = pd.Series(index=temperature.index)

    if heating_schedule is None:
        heating_schedule = {"time": ["08:00", "20:00", "23:00"], "values": [0, 1, 0]}

    segments = heating_schedule["time"]
    values = heating_schedule["values"]
    n_segments = len(segments)

    for i in range(n_segments - 1):
        schedule.loc[
            temperature.between_time(
                segments[i], segments[i + 1], inclusive="left"
            ).index
        ] = values[i + 1]

    schedule.loc[
        temperature.between_time(segments[-1], segments[0], inclusive="left").index
    ] = values[0]

    past_weather = temperature[schedule == 1].groupby(lambda x: x.date()).mean()

    with pd.option_context("future.no_silent_downcasting", True):
        hea_days = (
            (past_weather < threshold_val)
            .shift(1)
            .bfill()
            .infer_objects(copy=False)
            .rolling(threshold_dur, min_periods=1)
            .sum()
        )
        hea_days = hea_days.reindex(temperature.index).ffill()

    schedule = schedule.mask(hea_days < threshold_dur, 0)

    if exclude_weekends:
        schedule = schedule.mask(schedule.index.dayofweek > FRIDAY, 0)

    return schedule


def create_cooling_on(
    temperature: pd.Series,
    cooling_schedule: dict = None,
    threshold_val=23,
    threshold_dur=2,
    exclude_weekends: bool = True,
) -> pd.Series:
    schedule = pd.Series(index=temperature.index)

    if cooling_schedule is None:
        cooling_schedule = {"time": ["08:00", "20:00", "23:00"], "values": [0, 1, 0]}

    segments = cooling_schedule["time"]
    values = cooling_schedule["values"]
    n_segments = len(segments)

    for i in range(n_segments - 1):
        schedule.loc[
            temperature.between_time(
                segments[i], segments[i + 1], inclusive="left"
            ).index
        ] = values[i + 1]

    schedule.loc[
        temperature.between_time(segments[-1], segments[0], inclusive="left").index
    ] = values[0]

    past_weather = temperature[schedule == 1].groupby(lambda x: x.date()).mean()

    with pd.option_context("future.no_silent_downcasting", True):
        coo_days = (
            (past_weather > threshold_val)
            .shift(1)
            .bfill()
            .infer_objects(copy=False)
            .rolling(threshold_dur, min_periods=1)
            .sum()
        )
        coo_days = coo_days.reindex(temperature.index).ffill()

    schedule = schedule.mask(coo_days < threshold_dur, 0)

    if exclude_weekends:
        schedule = schedule.mask(schedule.index.dayofweek > FRIDAY, 0)

    return schedule


def yearly_fourier_multiplier(
    index: pd.DatetimeIndex,
    monthly_factors: dict[int, float],
    *,
    fourier_order: int = 3,
    normalize_mean: bool = True,
) -> pd.Series:
    """
    Smooth annual multiplier from user-defined monthly factors.

    monthly_factors:
        {1: 1.0, 2: 0.95, ..., 12: 1.2}

    fourier_order:
        Similar idea to Prophet yearly seasonality.
        2-4 is enough for smooth annual occupancy variation.
    """

    months = np.arange(1, 13)

    y = np.array(
        [monthly_factors.get(m, 1.0) for m in months],
        dtype=float,
    )

    # Use middle of each month as fitting point
    month_dates = pd.to_datetime([f"2024-{m:02d}-15" for m in months])

    t = month_dates.dayofyear.to_numpy() / 365.25

    X = [np.ones(len(t))]

    for k in range(1, fourier_order + 1):
        X.append(np.sin(2 * np.pi * k * t))
        X.append(np.cos(2 * np.pi * k * t))

    X = np.column_stack(X)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    t_index = index.dayofyear.to_numpy() / 365.25

    X_index = [np.ones(len(index))]

    for k in range(1, fourier_order + 1):
        X_index.append(np.sin(2 * np.pi * k * t_index))
        X_index.append(np.cos(2 * np.pi * k * t_index))

    X_index = np.column_stack(X_index)

    multiplier = X_index @ coef
    multiplier = np.clip(multiplier, 0.0, None)

    result = pd.Series(multiplier, index=index)

    if normalize_mean:
        result = result / result.mean()

    return result


def create_occ_profile(
    index: pd.DatetimeIndex,
    density: dict = None,
    exclude_weekends: bool = True,
):
    if density is None:
        density = {
            "time": ["08:00", "14:00", "20:00", "23:00"],
            "values": [0, 0.5, 1, 0],
        }

    profile = pd.Series(index=index)

    segments = density["time"]
    values = density["values"]
    n_segments = len(segments)

    for i in range(n_segments - 1):
        profile.iloc[
            index.indexer_between_time(
                segments[i], segments[i + 1], include_start=True, include_end=False
            )
        ] = values[i + 1]

    profile.iloc[
        index.indexer_between_time(
            segments[-1], segments[0], include_start=True, include_end=False
        )
    ] = values[0]

    if exclude_weekends:
        profile = profile.mask(profile.index.dayofweek > FRIDAY, 0)

    return profile


def generate_heating_energy(
    T_outdoor: pd.Series,
    heating_on: pd.Series,
    occupancy: pd.Series,  # expected 0..1 profile
    *,
    T_sp: float = 21.0,
    UA: float = 3.0,
    max_occupants: int = 40,
    base_load_heating: float = 20.0,
    occ_factor: float = 0.08,  # per occupant
    occ_ar_phi: float = 0.80,
    occ_ar_sigma: float = 0.12,
    noise_std: float = 0.5,
    zero_day_probability: float = 0.0,
    monthly_factors: dict[int, float] | None = None,
    fourier_order: int = 3,
    seed: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Synthetic building energy consumption with:
    - temperature-driven heating load
    - noisy autoregressive realized occupancy
    - base load during heating and non-heating hours
    """

    rng = np.random.default_rng(seed)

    T_outdoor = T_outdoor.astype(float)
    heating_on = heating_on.reindex(T_outdoor.index).fillna(0).astype(float)
    occupancy = occupancy.reindex(T_outdoor.index).fillna(0).astype(float)

    # ------------------------------------------------------------------
    # Realized occupants
    # ------------------------------------------------------------------
    occ_noise = ar1_noise(
        len(occupancy),
        phi=occ_ar_phi,
        sigma=occ_ar_sigma,
        seed=seed,
    )

    occ_fraction = (occupancy + occ_noise).clip(0, 1)

    # Integer occupant count
    if monthly_factors is not None:
        seasonal = yearly_fourier_multiplier(
            T_outdoor.index,
            monthly_factors,
            fourier_order=fourier_order,
            normalize_mean=True,
        )

        max_occupants = max_occupants * seasonal

    occupants = rng.binomial(max_occupants, occ_fraction)

    # Random full-day absences
    if zero_day_probability > 0:
        rng = np.random.default_rng(seed)

        days = pd.Index(T_outdoor.index.normalize().unique())

        zero_days = days[rng.random(len(days)) < zero_day_probability]
        occupants = pd.Series(occupants, index=T_outdoor.index).mask(
            T_outdoor.index.normalize().isin(zero_days), 0
        )

    # ------------------------------------------------------------------
    # Heating load
    # ------------------------------------------------------------------
    T_eff = rc_temperature(T_outdoor)
    T_effective_set = np.where(occupancy > 0.2, T_sp, T_eff)
    delta_T = np.clip(T_effective_set - T_eff, 0, None)
    temp_load = UA * delta_T

    # Occupancy-related additional load
    occ_load = occ_factor * occupants

    # Baseload exists both with and without heating
    base_load = np.where(
        heating_on > 0,
        base_load_heating,
        0,
    )

    Q = base_load + heating_on * temp_load + occ_load

    noise = rng.normal(0, noise_std, len(Q))
    Q = Q + noise

    return pd.Series(Q, index=T_outdoor.index).clip(lower=0), pd.Series(
        occupants, index=T_outdoor.index
    )


def generate_cooling_energy(
    T_outdoor: pd.Series,
    cooling_on: pd.Series,
    occupancy: pd.Series,
    *,
    T_sp: float = 24.0,
    UA: float = 3.5,
    max_occupants: int = 40,
    base_load_cooling: float = 20.0,
    occ_factor: float = 0.2,
    occ_ar_phi: float = 0.80,
    occ_ar_sigma: float = 0.12,
    noise_std: float = 0.2,
    zero_day_probability: float = 0.0,
    monthly_factors: dict[int, float] | None = None,
    fourier_order: int = 3,
    seed: int | None = None,
) -> pd.Series:
    """
    Generate synthetic cooling energy consumption.

    Parameters
    ----------
    T_outdoor:
        Outdoor temperature.
    cooling_on:
        Binary cooling availability/schedule, 0/1.
    occupancy:
        Occupancy profile, usually 0..1.
    """

    rng = np.random.default_rng(seed)

    T_outdoor = T_outdoor.astype(float)
    cooling_on = cooling_on.reindex(T_outdoor.index).fillna(0).astype(float)
    occupancy = occupancy.reindex(T_outdoor.index).fillna(0).astype(float)

    # ------------------------------------------------------------------
    # Realized occupants
    # ------------------------------------------------------------------
    occ_noise = ar1_noise(
        len(occupancy),
        phi=occ_ar_phi,
        sigma=occ_ar_sigma,
        seed=seed,
    )

    occ_fraction = (occupancy + occ_noise).clip(0, 1)

    # Integer occupant count
    if monthly_factors is not None:
        seasonal = yearly_fourier_multiplier(
            T_outdoor.index,
            monthly_factors,
            fourier_order=fourier_order,
            normalize_mean=True,
        )

        max_occupants = max_occupants * seasonal

    occupants = rng.binomial(max_occupants, occ_fraction)

    # Random full-day absences
    if zero_day_probability > 0:
        rng = np.random.default_rng(seed)

        days = pd.Index(T_outdoor.index.normalize().unique())

        zero_days = days[rng.random(len(days)) < zero_day_probability]
        occupants = pd.Series(occupants, index=T_outdoor.index).mask(
            T_outdoor.index.normalize().isin(zero_days), 0
        )

    # Temperature-driven cooling load
    T_eff = rc_temperature(T_outdoor)
    T_effective_set = np.where(occupancy > 0.2, T_sp, T_eff)
    delta_T = np.clip(T_eff - T_effective_set, 0, None)
    temp_load = UA * delta_T

    # Occupancy/internal gains increase cooling demand
    occ_load = occ_factor * occupants

    # Baseload exists even when cooling is off
    base_load = np.where(
        cooling_on > 0,
        base_load_cooling,
        0,
    )

    Q = base_load + cooling_on * (temp_load + occ_load)

    noise = rng.normal(0, noise_std, size=len(Q))
    Q = Q + noise

    return pd.Series(Q, index=T_outdoor.index).clip(lower=0), pd.Series(
        occupants, index=T_outdoor.index
    )


def generate_non_hvac_load(
    index: pd.DatetimeIndex,
    occupancy: pd.Series | None = None,
    *,
    base_load: float = 6.0,
    occ_factor: float = 0.01,
    daily_amplitude: float = 0.8,
    weekly_amplitude: float = 0.5,
    ar_phi: float = 0.90,
    ar_sigma: float = 0.25,
    noise_std: float = 0.15,
    seed: int | None = None,
) -> pd.Series:
    """
    Generate stochastic non-HVAC electricity/load.

    Includes:
    - constant base load
    - occupancy-driven internal load
    - daily pattern
    - weekday/weekend pattern
    - AR(1) correlated stochastic variation
    - white noise
    """

    rng = np.random.default_rng(seed)

    if occupancy is None:
        occupancy = pd.Series(0.0, index=index)
    else:
        occupancy = occupancy.reindex(index).fillna(0).astype(float)

    hour = index.hour + index.minute / 60

    # Smooth daily variation, peaking around afternoon
    daily = daily_amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * (hour - 8) / 24))

    # Lower non-HVAC load on weekends
    is_weekend = index.dayofweek > FRIDAY
    weekly = np.where(is_weekend, -weekly_amplitude, 0.0)

    # AR(1) stochastic component
    eps = rng.normal(0, ar_sigma, len(index))
    ar = np.zeros(len(index))

    for t in range(1, len(index)):
        ar[t] = ar_phi * ar[t - 1] + eps[t]

    # Independent measurement/random noise
    white_noise = rng.normal(0, noise_std, len(index))

    load = base_load + occ_factor * occupancy + daily + weekly + ar + white_noise

    return pd.Series(load, index=index).clip(lower=0)


def total_interlock(
    heating_energy: pd.Series,
    cooling_energy: pd.Series,
    heating_on: pd.Series,
    cooling_on: pd.Series,
    occupancy: pd.Series,
    T_outdoor: pd.Series,
    *,
    heating_priority_temp: float = 18.0,
    cooling_priority_temp: float = 22.0,
    mode: str = "temperature",
) -> pd.DataFrame:
    """
    Combine heating and cooling energy while preventing simultaneous operation.

    Parameters
    ----------
    mode:
        "temperature" :
            If both heating and cooling are on, choose heating below
            heating_priority_temp, cooling above cooling_priority_temp,
            and neither inside the deadband.

        "heating_priority" :
            Heating wins whenever both are on.

        "cooling_priority" :
            Cooling wins whenever both are on.

        "larger_load" :
            The larger of heating_energy and cooling_energy wins.
    """

    index = heating_energy.index

    cooling_energy = cooling_energy.reindex(index).fillna(0)
    heating_on = heating_on.reindex(index).fillna(0).astype(bool)
    cooling_on = cooling_on.reindex(index).fillna(0).astype(bool)

    heat_allowed = heating_on.copy()
    cool_allowed = cooling_on.copy()

    conflict = heating_on & cooling_on

    if mode == "temperature":
        if T_outdoor is None:
            raise ValueError("T_outdoor is required when mode='temperature'.")

        T_outdoor = T_outdoor.reindex(index)

        heat_wins = conflict & (T_outdoor < heating_priority_temp)
        cool_wins = conflict & (T_outdoor > cooling_priority_temp)
        neither_wins = conflict & ~(heat_wins | cool_wins)

        heat_allowed.loc[conflict] = False
        cool_allowed.loc[conflict] = False

        heat_allowed.loc[heat_wins] = True
        cool_allowed.loc[cool_wins] = True

        heat_allowed.loc[neither_wins] = False
        cool_allowed.loc[neither_wins] = False

    elif mode == "heating_priority":
        cool_allowed.loc[conflict] = False

    elif mode == "cooling_priority":
        heat_allowed.loc[conflict] = False

    elif mode == "larger_load":
        heat_wins = conflict & (heating_energy >= cooling_energy)
        cool_wins = conflict & (cooling_energy > heating_energy)

        heat_allowed.loc[conflict] = False
        cool_allowed.loc[conflict] = False

        heat_allowed.loc[heat_wins] = True
        cool_allowed.loc[cool_wins] = True

    else:
        raise ValueError(
            "mode must be one of: 'temperature', 'heating_priority', "
            "'cooling_priority', 'larger_load'."
        )

    heating_final = heating_energy.where(heat_allowed, 0)
    cooling_final = cooling_energy.where(cool_allowed, 0)

    non_hvac = generate_non_hvac_load(
        index=T_outdoor.index,
        occupancy=occupancy,
        base_load=6.0,
        seed=42,
    )

    total = non_hvac + heating_final + cooling_final

    return pd.DataFrame(
        {
            "heating": heating_final,
            "cooling": cooling_final,
            "total": total,
            "heating_allowed": heat_allowed.astype(int),
            "cooling_allowed": cool_allowed.astype(int),
            "conflict": conflict.astype(int),
        },
        index=index,
    )
