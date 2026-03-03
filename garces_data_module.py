# my_data_module_5.py

import pandas as pd
import numpy as np

# ─────────────────────────────
# 1) Default parameters & file‐paths
# ─────────────────────────────
_default_parameters = {
    "c_max":      30_000_000, # line budget
    "ref_bus":    1, # the name of the reference bus
    "d_min_prop": 0.9, # amount of demand that has to be served
    "M":          10, # big-M linearization constant
    "Gamma_max":  100_000, # big-M linearization constant
    "S_base":     100, # power base for conversion of p.u. to MW power flow
    "IC_scale":   1000, # factor by which to multiply Investment Cost from csv
    "LSP_scale":  100, # factor by which the load shed penalty is greater than demand bid
    "sigma":      8760, # factor for comparability of lines investments and social welfare
    "PDI_max":    1_000_000, # a number for the Alnowibet market power index - not sure what is sensible here...
    "NUI_max":    2, # a number for the Alnowibet NUI; NUI <= 1, so NUI_max = 2 is safe
    "M_Alnowibet": 1000000 # a big-M for the linearized Alnowibet NUI constraint; can be related to max line flow
}

_default_files = {
    "line_data_filename":  "garces_line_data.csv",
    "offer_data_filename": "garces_offer_data_c.csv",
    "bid_data_filename":   "garces_bid_data.csv",
    "scenarios_filename":  "garces_scenario_probabilities.csv"
}


# ─────────────────────────────
# 2) Data container
# ─────────────────────────────
class DataStore:
    # parameters
    c_max: float
    ref_bus: int
    d_min_prop: float
    M: float
    Gamma_max: float
    S_base: float
    IC_scale: float
    LSP_scale: float
    sigma: float
    PDI_max: float

    # scenario probs
    delta = None

    # line data
    line_data = None
    lines = None
    existing_lines = None
    prospective_lines = None
    o = None
    r = None
    b = None
    b_MW = None
    react = None
    f_max = None
    c = None

    # offer & bid data
    offer_data = None
    bid_data = None
    scenarios = None
    customers = None
    generating_units = None

    d_max_jh = None
    g_max    = None
    lambda_D = None
    c_U      = None
    lambda_G = None

    # demand
    d_calc = None
    d_max  = None
    d_min  = None

    # buses & mappings
    buses             = None
    customers_at_bus  = None
    s_j               = None
    generators_at_bus = None
    s_i               = None


# ─────────────────────────────
# 3) Loader function
# ─────────────────────────────
def load_data(
    parameters:     dict[str, float] = None,
    files:          dict[str, str]   = None,
    scenario_list:  list[str]        = None
):
    """
    Merge defaults with any overrides, then populate DataStore.
    Call this before accessing any `data.<attr>`.
    """
    # 3.1 parameters
    params = {**_default_parameters, **(parameters or {})}
    for k, v in params.items():
        setattr(DataStore, k, v)

    # 3.2 file‐paths
    paths = {**_default_files, **(files or {})}

    # 3.3 scenario probabilities
    full_delta = pd.read_csv(
        paths["scenarios_filename"], index_col="Scenario"
    )["Probability"].to_dict()

    if scenario_list is None:
        sel_delta = full_delta
    else:
        sel_delta = {sc: full_delta[sc] for sc in scenario_list if sc in full_delta}

    # Renormalize so sum(sel_delta.values()) == 1
    total_p = sum(sel_delta.values())
    if total_p <= 0:
        raise ValueError("No positive probabilities in sel_delta")
    DataStore.delta = {sc: p/total_p for sc, p in sel_delta.items()}


    # 3.4 line data
    ld = pd.read_csv(paths["line_data_filename"], index_col="Line")
    DataStore.line_data       = ld
    DataStore.lines           = ld.index
    DataStore.existing_lines  = ld[ld["Existing"] == 1].index
    DataStore.prospective_lines = ld[ld["Existing"] == 0].index

    DataStore.o     = ld["From"].to_dict()
    DataStore.r     = ld["To"].to_dict()
    ld["Susceptance"] = 1 / ld["Reactance"]
    DataStore.b     = ld["Susceptance"].to_dict()
    DataStore.b_MW  = (DataStore.S_base * ld["Susceptance"]).to_dict()
    DataStore.react = ld["Reactance"].to_dict()
    DataStore.f_max = ld["Capacity"].to_dict()
    DataStore.c     = (DataStore.IC_scale * ld["Investment_Cost"]).to_dict()

    # 3.5 offers & bids
    od = pd.read_csv(paths["offer_data_filename"])
    bd = pd.read_csv(paths["bid_data_filename"])

    if scenario_list is not None:
        bd = bd[bd["Scenario"].isin(scenario_list)]
        od = od[od["Scenario"].isin(scenario_list)]

    DataStore.offer_data   = od
    DataStore.bid_data     = bd

    DataStore.scenarios        = bd.Scenario.unique().tolist()
    DataStore.customers        = bd.Customer.unique().tolist()
    DataStore.generating_units = od.Generating_Unit.unique().tolist()

    DataStore.d_max_jh = bd.set_index(
        ["Scenario","Customer","Block"]
    )["Bid_Size"].to_dict()

    DataStore.g_max    = od.set_index(
        ["Scenario","Generating_Unit","Block"]
    )["Offer_Size"].to_dict()

    DataStore.lambda_D = bd.set_index(
        ["Customer","Block"]
    )["Bid_Price"].to_dict()

    DataStore.c_U      = (
        DataStore.LSP_scale * bd.groupby("Customer")["Bid_Price"].max()
    ).to_dict()

    DataStore.lambda_G = od.set_index(
        ["Generating_Unit","Block"]
    )["Offer_Price"].to_dict()

    # 3.6 demand
    dc = bd.groupby(["Scenario","Customer"])["Bid_Size"].sum()
    DataStore.d_calc = dc
    DataStore.d_max  = dc.to_dict()
    DataStore.d_min  = (DataStore.d_min_prop * dc).to_dict()

    # 3.7 buses & mappings
    buses = list(set(od.Bus.unique().tolist() + bd.Bus.unique().tolist()))
    DataStore.buses = buses

    DataStore.customers_at_bus = {
        b: bd[bd["Bus"] == b].Customer.unique().tolist()
        for b in buses
    }
    DataStore.s_j = {
        cust: bus
        for bus, clist in DataStore.customers_at_bus.items()
        for cust in clist
    }

    DataStore.generators_at_bus = {
        b: od[od["Bus"] == b].Generating_Unit.unique().tolist()
        for b in buses
    }
    DataStore.s_i = {
        gen: bus
        for bus, glist in DataStore.generators_at_bus.items()
        for gen in glist
    }


# ─────────────────────────────
# 4) Module‐level proxy for `data.<attr>`
# ─────────────────────────────
# This is super-neat as it basically means you can use data.DataStore.<attr>
# or data.<attr> when accessing the data you have loaded!
def __getattr__(name: str):
    """
    Forward undefined module attributes to DataStore.
    Enables `data.delta` instead of `data.DataStore.delta`.
    """
    if hasattr(DataStore, name):
        return getattr(DataStore, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")