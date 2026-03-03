# wind_data_module_01.py

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import scipy as sp

from shapely.geometry import Point

# Random number generator:
rng = np.random.default_rng()

# Further parameters
# Geographic data
# ---------------
shapefile_path = "geodata/gb.shp"

# Load the UK data
uk = gpd.read_file(shapefile_path)
uk = uk.to_crs(epsg=27700)

# Create bounding box for the UK shapefile:
minx, miny, maxx, maxy = uk.total_bounds

generation_boost = 3


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
    "M_Alnowibet": 1000000, # a big-M for the linearized Alnowibet NUI constraint; can be related to max line flow
    
    "Gamma_wind": 1000, # a big-M for the linearized strong duality constraint in the wind-adapted model
    # Wind parameters
    # ---------------
    # Average wind speed linear regression model parameters:
    "intercept" : 4.106034866084783, # (m/s)
    "coefficients": [-9.53850467e-07, 1.02939352e-06],

    "L_decay" : 723, # (km). Length scale for exponential decay of correlation w/distance.
    "cutin_speed" : 3, # (m/s) Speed at which wind power is available
    "rated_speed" : 13, # (m/s) Speed at which wind power operates at rated capacity
    "cutout_speed": 25, # (m/s) Speed at which wind power unavailable

    "capacity_options" : [100, 200, 400], # (MW) Rated capacities of wind projects
    "kappa" : 45, # (EUR/MWh) Strike price for wind generation

    # Bounding coordinates (in metres) for wind power projects (taken UK coords)
    "x_min": 2500, 
    "x_max": 652500, 
    "y_min": 12500, 
    "y_max": 1182500, 

    # Network parameters
    # ------------------
    # nb_T1_buses = 10 # number of buses for demand and/or traditional generation
    # nb_T2_buses = 3 # number of wind power projects

    "p1" : 10, # PERCENTAGE of extra links for EXISTING network (over and above MST).
    "p2" : 20, # PERCENTAGE of links to consider opening (over and above second MST).
    "p3" : 50, # PERCENTAGE of T1 buses which have existing generation.

    "cost_per_km" : 120_000, # (EUR) Cost per km of transmission line.

    "reactance_mean" : 0.4186666666666667, # (p.u.)
    "reactance_std" : 0.1635386464689276, # (p.u.)
    "reactance_min" : 0.2, # (p.u.)
    "reactance_max" : 0.68, # (p.u.)

    "cap_mean" : 92.33333333333333, # (MW) mean line capacity
    "cap_std" : 11.10955544665142, # (MW)
    "cap_min" : 70, # (MW)
    "cap_max" : 100, # (MW)

    # Offer parameters
    # ----------------
    "nb_scenarios" : 5,

    "gen_cap_low" : 150, # (MW) - minimum capacity of a generating unit
    "gen_cap_high" : 600, # (Mw) - maximum capacity of a generating unit

    "p4" : 2,  # PERCENTAGE of thermal unit offers which are 0.
    "b_offer" : 59.6, # Price of the first MWh
    "m_offer" : 0.0394, # Increase in price for each additional MWh
    "block_premium" : 0.1, # PROPORTION by which to increase the price of the next block

    "mu" : 370, # (kg CO2/MWh) carbon emitted in thermal generation
    "c_CO2" : 0.225, # (EUR / kg) cost of CO2

    # Bid parameters
    # --------------
    "demand_low" : 40, # (MW) - lowest possible total demand by a customer (base case)
    "demand_high" : 240, # (MW) - highest possible total demand by a customer (base case)

    "bid_price_low" : 110, # (EUR/MWh)
    "bid_price_high" : 120, # (EUR/MWh)


    # Miscellaneous parameters
    # ------------------------
    "IC_scale" : 1000 # factor by which to multiply Investment Cost from csv
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

    # additional parameters to make the wind version work
    NUI_max:    float # a number for the Alnowibet NUI; NUI <= 1, so NUI_max = 2 is safe
    M_Alnowibet: float # a big-M for the linearized Alnowibet NUI constraint; can be related to max line flow

    Gamma_wind: float # a big-M for the linearized strong duality constraint in the wind-adapted model
    
    # wind parameters
    # ---------------
    # Average wind speed linear regression model parameters:
    intercept: float # (m/s)
    coefficients: list[float]

    L_decay : float # (km). Length scale for exponential decay of correlation w/distance.
    cutin_speed: float # (m/s) Speed at which wind power is available
    rated_speed : float # (m/s) Speed at which wind power operates at rated capacity
    cutout_speed: float # (m/s) Speed at which wind power unavailable

    capacity_options : list[float] # (MW) Rated capacities of wind projects
    kappa : float # (EUR/MWh) Strike price for wind generation

    # Bounding coordinates (in metres) for wind power projects (taken UK coords)
    x_min : float 
    x_max: float 
    y_min: float 
    y_max: float 

    # Network parameters
    # ------------------
    # nb_T1_buses = 10 # number of buses for demand and/or traditional generation
    # nb_T2_buses = 3 # number of wind power projects

    p1 : float # PERCENTAGE of extra links for EXISTING network (over and above MST).
    p2 : float # PERCENTAGE of links to consider opening (over and above second MST).
    p3 : float # PERCENTAGE of T1 buses which have existing generation.

    cost_per_km : float # (EUR) Cost per km of transmission line.

    reactance_mean : float # (p.u.)
    reactance_std : float # (p.u.)
    reactance_min : float # (p.u.)
    reactance_max : float # (p.u.)

    cap_mean : float # (MW) mean line capacity
    cap_std : float # (MW)
    cap_min : float # (MW)
    cap_max : float # (MW)

    # Offer parameters
    # ----------------
    nb_scenarios : float

    gen_cap_low : float # (MW) - minimum capacity of a generating unit
    gen_cap_high : float # (Mw) - maximum capacity of a generating unit

    p4 : float  # PERCENTAGE of thermal unit offers which are 0.
    b_offer: float # Price of the first MWh
    m_offer: float # Increase in price for each additional MWh
    block_premium : float # PROPORTION by which to increase the price of the next block

    mu : float # (kg CO2/MWh) carbon emitted in thermal generation
    c_CO2 : float # (EUR / kg) cost of CO2

    # Bid parameters
    # --------------
    demand_low : float # (MW) - lowest possible total demand by a customer (base case)
    demand_high : float # (MW) - highest possible total demand by a customer (base case)

    bid_price_low : float # (EUR/MWh)
    bid_price_high: float # (EUR/MWh)


    # Miscellaneous parameters
    # ------------------------
    IC_scale : float

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
    G_plus = None
    D_plus = None

    scenarios = None
    customers = None
    generating_units = None

    d_max_jh = None
    g_max    = None
    lambda_D = None
    c_U      = None
    lambda_G = None

    # wind
    WPPs = None
    p_max = None

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
    WPPs_at_bus       = None
    s_m               = None

    


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


def generate_random_point():
    '''Create a point inside the UK.'''
    for _ in range(10_000): # Attempt at most 10_000 times...
        
        # Generate a point in the bounding box:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        point = Point(x, y)
        
        # Check if it's inside the UK boundary:
        if uk.contains(point).any():
            return point
    raise ValueError("Failed to generate a valid point within UK.")

def generate_network(nb_T1_buses, nb_T2_buses, p1, p2, save_to_csv=False):
    ''' Generate a network.'''
    # Create the locations for the buses with demand and/or traditional generation:
    points = [generate_random_point() for _ in range(nb_T1_buses)]

    # Store their (x,y) positions:
    T1_x = [P.x for P in points]
    T1_y = [P.y for P in points]

    # Create a DataFrame to store the information about the buses.
    # N.B. the buses are marked either T1 (demand/traditional generation) or T2 (
    # wind generation locations).

    # Create T1 DataFrame:
    T1_df = pd.DataFrame(
        {
            "bus_type": "T1",
            "x": T1_x,
            "y": T1_y
        }
    )
    # T1_df['bus_id'] = "T1_" + T1_df.index.astype(str)
    T1_df['Bus'] = "T1_" + (T1_df.index+1).astype(str)
    T1_df = T1_df.set_index('Bus')

    # Extract coordinates
    coords = T1_df[['x', 'y']].values
    T1_bus_ids = T1_df.index.to_numpy()

    T2_df = None
    if nb_T2_buses > 0:
        # Wind power projects can be located anywhere within the bounding box:
        T2_x = rng.uniform(low = DataStore.x_min, high = DataStore.x_max, size = nb_T2_buses)
        T2_y = rng.uniform(low = DataStore.y_min, high = DataStore.y_max, size = nb_T2_buses)

        # Create T2 DataFrame:
        T2_df = pd.DataFrame(
            {
                "bus_type": "T2",
                "x": T2_x,
                "y": T2_y
            }
        )
        # T2_df['bus_id'] = "T2_" + T2_df.index.astype(str)
        T2_df['Bus'] = "T2_" + (T2_df.index+1).astype(str)
        T2_df = T2_df.set_index('Bus')

        # Bring the T1 and T2 DataFrames together as one:
        buses_df = pd.concat([T1_df, T2_df])

        # Closest T1 bus to each T2 bus:
        # ------------------------------
        T2_coords = T2_df[['x', 'y']].values

        # Compute distance matrix between T2 and T1 buses
        dist_matrix_T2_T1 = sp.spatial.distance.cdist(T2_coords, coords)
        # N.B. Rows are T2 buses, columns are T1 buses

        # In a dictionary, store the closest T1 bus for each T2 bus:
        closest_connection = {}
        closest_distance = {}
        for i, T2 in enumerate(T2_coords):
            T2_bus_id = T2_df.index[i]
            j = np.argmin(dist_matrix_T2_T1[i,:])
            T1_bus_id = T1_df.index[j]
            # print(f"Bus {T2_bus_id} is closest to {T1_bus_id}")
            # print("")
            closest_connection[T2_bus_id] = T1_bus_id
            closest_distance[T2_bus_id] = dist_matrix_T2_T1[i,j]

        # Add the closest_connection dictionary as a column to the T2 DataFrame:
        T2_df['closest_T1'] = T2_df.index.map(closest_connection)

    else:
        buses_df = T1_df


    # GENERATE GRAPH FOR THE POTENTIAL TRANSMISSION NETWORK (all edges)
    # -----------------------------------------------------------------
    # The graph consists of T1 buses and the edges between them are the distances.

    # Compute full pairwise distance matrix
    dist_matrix = sp.spatial.distance.cdist(coords, coords)

    G = nx.Graph()
    n = len(T1_bus_ids)

    ebunch = [
        (T1_bus_ids[i], T1_bus_ids[j], dist_matrix[i, j])
        for i in range(n) for j in range(n) if i < j
    ]

    G.add_weighted_edges_from(ebunch)

    # Create a plausible transmission network based on the minimum spanning tree:
    T = nx.minimum_spanning_tree(G)

    # Add some extra edges (the shortest p1% of links NOT already in T):
    candidates = [e for e in G.edges() if not T.has_edge(*e)]
    nb_to_add = int(np.floor(len(candidates)*(p1/100)))
    for u,v in sorted(candidates, key=lambda e: G[e[0]][e[1]]['weight'])[:nb_to_add]:
        T.add_edge(u, v, weight=G[u][v]['weight'])


    # PLAUSIBLE TRANSMISSION EXTENSIONS
    # ---------------------------------
    # Second spanning tree:
    # ---------------------
    # Create complement of the actual minimum spanning tree (TC = G\T):
    G_edges = set(G.edges())
    T_edges = set(T.edges())
    TC_edges = G_edges - T_edges
    TC = nx.Graph()
    ebunch = [(u, v, G[u][v]['weight']) for u,v in TC_edges]
    TC.add_weighted_edges_from(ebunch)

    # Create the minimum spanning of TC = G\T:
    TCT = nx.minimum_spanning_tree(TC)

    # Add some extra edges (the shortest p2% of links NOT already in TCT):
    candidates = [e for e in G.edges() if (not TCT.has_edge(*e) and not T.has_edge(*e))]
    nb_to_add = int(np.floor(len(candidates)*(p2/100)))
    for u,v in sorted(candidates, key=lambda e: G[e[0]][e[1]]['weight'])[:nb_to_add]:
        TCT.add_edge(u, v, weight=G[u][v]['weight'])


    # ADD REACTANCE, CAPACITY, AND INVESTMENT COST DATA TO LINES
    # ----------------------------------------------------------
    a, b = (DataStore.reactance_min - DataStore.reactance_mean) / DataStore.reactance_std, (DataStore.reactance_max - DataStore.reactance_mean) / DataStore.reactance_std
    # print(f"a: {a},\n b: {b},\n reactance_mean: {reactance_mean},\n reactance_std: {reactance_std}")
    reactance_distn = sp.stats.truncnorm(a, b, loc=DataStore.reactance_mean, scale=DataStore.reactance_std)

    a, b = (DataStore.cap_min - DataStore.cap_mean) / DataStore.cap_std, (DataStore.cap_max - DataStore.cap_mean) / DataStore.cap_std
    cap_distn = sp.stats.truncnorm(a, b, loc=DataStore.cap_mean, scale=DataStore.cap_std)

    # Create line data for connections between T1 buses
    # -------------------------------------------------
    dist = [G[e[0]][e[1]]['weight']/1000 for e in T.edges()] # distance in km
    from_u = [u for u, _ in T.edges()]
    to_v = [v for _, v in T.edges()]

    # Construct a DataFrame to contain the existing lines' data:
    existing_lines_df = pd.DataFrame(
        {
            'From': from_u,
            'To': to_v,
            'Distance': dist
        }
    )

    # print(f"len(existing_lines_df): {len(existing_lines_df)}")

    # Add the relevant data to the existing_lines_df:
    existing_lines_df['Investment_Cost'] = (existing_lines_df['Distance'] * DataStore.cost_per_km) / DataStore.IC_scale
    line_capacities = list(cap_distn.rvs(size = len(existing_lines_df)))
    existing_lines_df['Capacity'] = line_capacities
    line_reactances = list(reactance_distn.rvs(size = len(existing_lines_df)))
    existing_lines_df['Reactance'] = line_reactances
    existing_lines_df['Existing'] = 1
    existing_lines_df['Line_Copy'] = 1
    # N.B. Line capacities are the same for all lines in the same corridor.

    # Create copies of the existing lines as potential expansions:
    dist += dist
    from_u += from_u
    to_v += to_v

    el_copy_df = pd.DataFrame(
        {
            'From': from_u,
            'To': to_v,
            'Distance': dist,
            'Line_Copy': len(existing_lines_df)*[2] + len(existing_lines_df)*[3] 
        }
    )

    # Add the relevant data to the existing_lines_df:
    el_copy_df['Investment_Cost'] = (el_copy_df['Distance'] * DataStore.cost_per_km) / DataStore.IC_scale
    el_copy_df['Capacity'] = 2*line_capacities # line_capacities is a list
    el_copy_df['Reactance'] = 2*line_reactances # likewise, this is a list
    el_copy_df['Existing'] = 0

    # Create a single DataFrame for the existing corridors:
    existing_corridors_df = pd.concat([existing_lines_df, el_copy_df])


    # Now create transmission line expansions for the NEW corridors:
    # N.B. This does not yet include the connections for wind projects.
    dist = 3*[G[e[0]][e[1]]['weight']/1000 for e in TCT.edges()] # distance in km
    from_u = 3*[u for u, _ in TCT.edges()]
    to_v = 3*[v for _, v in TCT.edges()]

    # Construct a DataFrame to contain the existing lines' data:
    prospective_lines_df = pd.DataFrame(
        {
            'From': from_u,
            'To': to_v,
            'Distance': dist
        }
    )

    # Add the relevant data to the prospective_lines_df:
    prospective_lines_df['Investment_Cost'] = (prospective_lines_df['Distance'] * DataStore.cost_per_km) / DataStore.IC_scale
    line_capacities = list(cap_distn.rvs(size = len(TCT.edges())))
    prospective_lines_df['Capacity'] = 3*line_capacities
    line_reactances = list(reactance_distn.rvs(size = len(TCT.edges())))
    prospective_lines_df['Reactance'] = 3*line_reactances
    prospective_lines_df['Existing'] = 0
    prospective_lines_df['Line_Copy'] = len(TCT.edges())*[1] + len(TCT.edges())*[2] + len(TCT.edges())*[3]

    # Finally create lines for the connections to the grid:
    if nb_T2_buses > 0:
        from_u = 3*[u for u,_ in closest_connection.items()]
        to_v = 3*[v for _,v in closest_connection.items()]
        dist = 3*[closest_distance[u]/1000 for u,_ in closest_connection.items()] # km

        # Create a DataFrame to store the T2 to T1 connection lines:
        connection_lines_df = pd.DataFrame(
            {
                'From': from_u,
                'To': to_v,
                'Distance': dist
            }
        )

        # Add the relevant data to the connection_lines_df:
        # -------------------------------------------------
        # Calculate investment cost (in EUR 000s) from connection distances:
        connection_lines_df['Investment_Cost'] = (connection_lines_df['Distance'] * DataStore.cost_per_km) / DataStore.IC_scale

        # Connection capacity commensurate with rated capacity:
        line_capacities = nb_T2_buses * [DataStore.capacity_options[1]]  

        # Create capacities for ALL copies of ALL lines:
        connection_lines_df['Capacity'] = 3*line_capacities

        # Create reactances for ALL lines and ALL copies:
        line_reactances = list(reactance_distn.rvs(size = nb_T2_buses))
        connection_lines_df['Reactance'] = 3*line_reactances

        # Ensure lines are not already present:
        connection_lines_df['Existing'] = 0

        # Label the line copies appropriately:
        connection_lines_df['Line_Copy'] = nb_T2_buses*[1] + nb_T2_buses*[2] + nb_T2_buses*[3]

        # Put everything together:
        line_data = pd.concat([existing_corridors_df, prospective_lines_df, connection_lines_df])
    
    else: # If there are no T2 buses...
        line_data = pd.concat([existing_corridors_df, prospective_lines_df])

    line_data['Capacity'] = np.round(line_data['Capacity'])
    line_data['Reactance'] = np.round(line_data['Reactance'], 2)
    line_data['Investment_Cost'] = np.round(line_data['Investment_Cost']/10)
    
    line_data = line_data.sort_values(['Line_Copy', 'From', 'To'])
    line_data.reset_index(drop=True, inplace = True)
    line_data.index = line_data.index + 1
    line_data.index.name = 'Line'
    line_data = line_data[['Line_Copy', 'From', 'To', 'Distance', 'Reactance', 'Capacity', 'Investment_Cost', 'Existing']]

    if save_to_csv:
        line_data.to_csv(save_to_csv['line_data'])

    return T1_df, T2_df, line_data

def generate_offer_data(T1_bus_ids, save_to_csv=False, G_plus = None):
    '''Generate offer data from a list of buses.'''

    def offer_size_fn(unit, block):
        '''Create Offer_Size for each block.'''
        if block == 1:
            # Binomial is so Offer_Size is 0 sometimes (representing an outage):
            return np.floor(0.4 * G_plus[s_i[unit]] * rng.binomial(1, 1-DataStore.p4/100))
        else:
            return np.floor(0.3 * G_plus[s_i[unit]] * rng.binomial(1, 1-DataStore.p4/100))

    def offer_price_fn(offer_size, block):
        base_offer = DataStore.b_offer + DataStore.m_offer * offer_size
        if block == 1:
            return np.round(base_offer)
        else:
            return np.round(offer_price_fn(offer_size,1) * (1 + DataStore.block_premium*(block-1)))


    # Generate the capacity of generators as a uniform r.v. between min and max:
    # N.B. p3% of buses of NO generation.
    if G_plus is None:
        G_plus = {k: np.floor(rng.binomial(1,DataStore.p3/100)*rng.uniform(DataStore.gen_cap_low,DataStore.gen_cap_high)) for k in T1_bus_ids}

    # Get the capacities which are non-zero:
    G_plus_non_zero = {k: v for k,v in G_plus.items() if v > 0}

    # s_i has keys which are generating units and values which are their buses:
    s_i = {gen_id+1: bus_id for gen_id, bus_id in enumerate(G_plus_non_zero.keys())}


    # Create GENERIC offer data first (not by scenario):
    offer_data = pd.DataFrame(
        {
            "Generating_Unit": list(s_i.keys())
        }
    )

    offer_data['Bus'] = s_i.values()

    offer_data = offer_data.merge(pd.DataFrame({'Block': [1, 2, 3]}), how='cross')
    offer_data = offer_data.merge(pd.DataFrame({'Scenario': list(range(1, DataStore.nb_scenarios+1))}), how='cross')

    offer_data = offer_data.sort_values(by = ["Scenario","Generating_Unit","Block"])

    # Add the Offer_Size and Offer_Price to the DataFrame:
    offer_data['Offer_Size'] = offer_data.apply(
    lambda row: offer_size_fn(row['Generating_Unit'], row['Block']), axis=1)

    offer_data['Offer_Price'] = offer_data.apply(
    lambda row: offer_price_fn(row['Offer_Size'], row['Block']), axis=1)

    offer_data['Offer_Size'] = generation_boost * offer_data['Offer_Size']

    offer_data = offer_data[['Scenario', 'Generating_Unit', 'Bus', 'Block', 'Offer_Size', 'Offer_Price']]

    offer_data.reset_index(drop=True, inplace=True)
    offer_data.index = offer_data.index + 1

    if save_to_csv:
        offer_data.to_csv(save_to_csv['offer'], index=False)

    return offer_data, G_plus # By returning G_plus, we can make more scenarios

def generate_bid_data(T1_bus_ids, save_to_csv=False, D_plus=None):
    
    def bid_size_fn(customer, block):
        if block == 1:
            return 0.9 * D_plus[s_j[customer]]
        else:
            return 0.1 * D_plus[s_j[customer]]

    def bid_price_fn(customer, block, bid_price):
            return (0.9)**(block-1) * bid_price[s_j[customer]]

    # Generate total demand at each bus:
    if D_plus is None:
        D_plus = {k: rng.uniform(DataStore.demand_low, DataStore.demand_high) for k in T1_bus_ids}

    # Create dictionary from demand to bus:
    s_j = {cust_id+1: bus_id for cust_id, bus_id in enumerate(D_plus.keys())}

    # Create DataFrame to store bid data:
    bid_data = pd.DataFrame(
        {
            "Customer": list(s_j.keys()),
            "Bus": T1_bus_ids
        }
    )

    # Create 3 Blocks per Customer:
    bid_data = bid_data.merge(pd.DataFrame({'Block': [1, 2, 3]}), how='cross')

    # Create rows for each Scenario (for each Customer and Block)
    bid_data = bid_data.merge(pd.DataFrame({'Scenario': list(range(1, DataStore.nb_scenarios+1))}), how='cross')

    bid_data = bid_data.sort_values(by = ["Scenario","Customer","Block"])

    # Get the price each customer is willing to pay for their first block of demand:
    bid_price = {k: rng.uniform(DataStore.bid_price_low, DataStore.bid_price_high) for k in T1_bus_ids}

    bid_data['Bid_Size']= bid_data.apply(lambda row: bid_size_fn(row['Customer'], row['Block']), axis=1)
    bid_data['Bid_Price'] = bid_data.apply(lambda row: bid_price_fn(row['Customer'], row['Block'], bid_price), axis=1)

    # Adjust the Bid_Size by scenario:
    # Scenario growth factor:
    sgf = {k: rng.uniform(1, 1.183) for k in bid_data['Scenario'].unique()}

    # Bus growth factor:
    bgf = {k: rng.uniform(1,1.83) for k in bid_data['Bus'].unique()}

    def changed_bid_size(bid_size, sgf, bgf):
        return bid_size*sgf*bgf

    bid_data['Bid_Size'] = bid_data.apply(lambda row: changed_bid_size(row['Bid_Size'], sgf[row['Scenario']], bgf[row['Bus']]), axis=1)

    bid_data['Bid_Size'] = np.round(bid_data['Bid_Size'])
    bid_data['Bid_Price'] = np.round(bid_data['Bid_Price'])
    # Tidy up the DataFrame:
    bid_data = bid_data[['Scenario', 'Customer', 'Bus', 'Block', 'Bid_Size', 'Bid_Price']]

    bid_data.reset_index(drop=True, inplace=True)
    bid_data.index = bid_data.index + 1

    # Save to csv if desired:
    if save_to_csv:
        bid_data.to_csv(save_to_csv['bid'], index=False)

    return bid_data, D_plus


def generate_wind_data(T2_df, save_to_csv=False):

    # Formula for the average windspeed at point (x,y); based on linear model:
    def avg_windspeed(x,y):
        return DataStore.intercept + DataStore.coefficients[0]*x + DataStore.coefficients[1]*y

    def P_max_px_fn(V, cutin_speed=DataStore.cutin_speed, rated_speed=DataStore.rated_speed, cutout_speed=DataStore.cutout_speed):
        '''Calculates proportion of Rated_Capacity available, given wind speed.
        Args:
            V: np.array of wind speeds
            cutin_speed: wind turbine cut-in speed (typically 3 m/s)
            rated_speed: wind turbine rated speed (typically 13 m/s)
            cutout_speed: wind turbine cut-out speed (typically 25 m/s)
        Returns:
            result: np.array of proportions of Rated_Capacity available
        '''

        result = np.zeros_like(V, dtype=float)
        
        # Ramp region: between cut-in and rated
        ramp_mask = (V >= cutin_speed) & (V <= rated_speed)
        result[ramp_mask] = (V[ramp_mask] - cutin_speed) / (rated_speed - cutin_speed)
        
        # Flat region: between rated and cut-out
        flat_mask = (V > rated_speed) & (V <= cutout_speed)
        result[flat_mask] = 1
        
        return result

    nb_T2_buses = len(T2_df)
    
    # CREATE SCENARIOS FOR WIND SPEED AT EACH LOCATION
    # ------------------------------------------------
    # Get the average windspeed at the location
    T2_df['Average_Windspeed'] = avg_windspeed(T2_df['x'], T2_df['y'])

    # Extract locations of T2 buses and measure in km NOT m:
    locations = T2_df[['x','y']].to_numpy()/1000

    # Calculate distances between T2 buses:
    D = sp.spatial.distance_matrix(locations, locations)

    # Calculate correlations coefficients for locational wind speeds:
    R = np.exp(-D/DataStore.L_decay)

    # Generate one standard normal variable for each bus and scenario:
    X = rng.normal(size=(nb_T2_buses,DataStore.nb_scenarios))

    # Induce the correlation between buses (correlated within each scenario):
    L = np.linalg.cholesky(R)
    Y = np.matmul(L,X)
    # N.B. Y still contains one r.v. for each bus and scenario.

    # Get the percentiles of each transformed r.v.:
    U = sp.stats.norm.cdf(Y)

    # Create RAYLEIGH r.v.s for each location
    # ---------------------------------------
    # This uses the percentile from U and the average wind speed for the location.

    # Create scale parameters for the Rayleigh distribution:
    scale_params = T2_df['Average_Windspeed'].to_numpy() * (2/np.sqrt(np.pi))
    # N.B. This is because the scale parameter is mean * (2/sqrt(pi)).

    # Create a grid, so it can be used for multiple scenarios:
    scale_params = np.tile(scale_params,(DataStore.nb_scenarios,1)).transpose()

    # Create the wind speeds:
    k = 2 # To create the Rayleigh distribution (Weibull special case)
    V = sp.stats.weibull_min.ppf(U, c=k, scale=scale_params)

    # Calculate the percentage of Rated_Capacity available by bus in each scenario:
    P_max_px = P_max_px_fn(V)

    # Each row is a bus, each column is a scenario:
    P_max_px_df = pd.DataFrame(P_max_px, index=T2_df.index)
    P_max_px_df.columns = pd.Index(range(1,P_max_px.shape[1]+1), name='Scenario')

    # Use P_max_px_df to create a DataFrame with one row for each Bus | Scenario:
    # The value is the proportion of Rated_Capacity available for that Bus-Scenario.
    proportions_long = (
        P_max_px_df
        .rename_axis(index='Bus')
        .reset_index()
        .melt(id_vars='Bus', var_name='Scenario', value_name='Proportion')
    )

    # Store each wind power project at each bus with its capacity:
    WPPs = pd.DataFrame(
        [(bus, f'{bus}_W{i}', cap) for bus in T2_df.index for i, cap in enumerate(DataStore.capacity_options, start=1)],
        columns=['Bus', 'WPP', 'Rated_Capacity']
    )

    # Create the final DataFrame for the wind data
    # --------------------------------------------
    # Merge the WPPs data with the available proportions of Rated_Capacity:
    out = proportions_long.merge(WPPs, on='Bus', how='inner')

    # Use the scenario-based proportion factor to calculate offer size:
    out['Offer_Size'] = np.round(out['Rated_Capacity'] * out['Proportion'])

    # Tidy up the DataFrame:
    out = out[['Scenario', 'WPP', 'Bus', 'Rated_Capacity', 'Offer_Size']]

    # Export to csv if desired:
    if save_to_csv:
        out.to_csv(save_to_csv['wind'], index=False)

    return out



def simulate_data(
    nb_T1_buses :   int,
    nb_T2_buses:    int,
    G_plus = None,
    D_plus = None,
    save_to_csv = False,
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
    full_delta = {k: 1/DataStore.nb_scenarios for k in range(1,DataStore.nb_scenarios+1)}
    
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
    T1_df, T2_df, ld = generate_network(nb_T1_buses, nb_T2_buses, DataStore.p1, DataStore.p2, save_to_csv=save_to_csv)
    T1_bus_ids = list(T1_df.index.unique())
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
    od, sim_G_plus = generate_offer_data(T1_bus_ids, G_plus = G_plus)
    bd = pd.read_csv(paths["bid_data_filename"])
    bd, sim_D_plus = generate_bid_data(T1_bus_ids, D_plus = D_plus, save_to_csv=save_to_csv)

    if scenario_list is not None:
        bd = bd[bd["Scenario"].isin(scenario_list)]
        od = od[od["Scenario"].isin(scenario_list)]

    DataStore.offer_data   = od
    DataStore.bid_data     = bd

    DataStore.G_plus = sim_G_plus
    DataStore.D_plus = sim_D_plus

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

    # Add the CO2 cost to each price:
    carbon_intensity = {gen: np.round(rng.uniform(0.95,1.33)*DataStore.mu) for gen in DataStore.generating_units}
    od['Carbon_Intensity'] = od['Generating_Unit'].map(carbon_intensity)
    od['Upper_Level_Cost'] = od['Offer_Price'] + od['Carbon_Intensity'] * DataStore.c_CO2

    DataStore.lambda_tilde_G = od.set_index(
        ["Generating_Unit","Block"]
    )["Upper_Level_Cost"].to_dict()

    # 3.5b wind
    wd = generate_wind_data(T2_df=T2_df, save_to_csv=save_to_csv)
    DataStore.WPPs = wd.WPP.unique().tolist()
    DataStore.p_max = wd.set_index(["Scenario", "WPP"])["Offer_Size"].to_dict()


    # 3.6 demand
    dc = bd.groupby(["Scenario","Customer"])["Bid_Size"].sum()
    DataStore.d_calc = dc
    DataStore.d_max  = dc.to_dict()
    DataStore.d_min  = (DataStore.d_min_prop * dc).to_dict()

    # 3.7 buses & mappings
    buses = list(set(od.Bus.unique().tolist() + bd.Bus.unique().tolist() + wd.Bus.unique().tolist()))
    DataStore.buses = buses

    DataStore.ref_bus = buses[0]

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

    DataStore.WPPs_at_bus = {
        b: wd[wd["Bus"] == b].WPP.unique().tolist()
        for b in buses
    }

    DataStore.s_m = {
        WPP: bus
        for bus, WPPlist in DataStore.WPPs_at_bus.items()
        for WPP in WPPlist
    }


# NOTE: It might be that I want to do these things together... (SCENARIOS)
def update_wind_data():
    pass

def update_demand_data():
    pass