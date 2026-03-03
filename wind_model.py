# garces_et_al_model_4.py
# garces_et_al_model_3.py
# garces_et_al_model_2.py was a failure.


# IMPORTS
# --------

# Standard libraries
import pandas as pd
import numpy as np

# Optimizer
import xpress as xp

# No need to import the data here; this will be done in the .ipynb file.

def build_model(data, new_lines = [], bilevel = True, 
                alternative_objective = False, alnowibet = False):

    # Create the Xpress problem:
    prob = xp.problem("MILP Formulation")
    var_map = {}

    # UPPER-LEVEL VARIABLES
    # ---------------------
    # "x_k" : binary variable, 1 if line k is built, 0 otherwise.
    x = {(k): xp.var(vartype = xp.binary) for k in data.lines}

    # (Eqn 3) : All existing lines are "open"
    for k in data.existing_lines:
        x[k].lb = 1

    # If the new_lines to be opened have been specified, force these decisions:
    if new_lines:
        for k in new_lines:
            x[k].lb = 1

        for k in data.lines:
            if k not in data.existing_lines:
                if k not in new_lines:
                    x[k].ub=0

    prob.addVariable(x)
    var_map['x'] = x

    # If we are adding an Alnowibet (2023)-style market-power constraint, then
    # we need to add the variables required for the linearized market-power
    # constraint to be written into the model.
    # TODO: Consider if this is really an 'upper level' variables; I guess so...
    if alnowibet == 1:
        Phi_plus = {s: xp.var(vartype = xp.continuous) for s in data.buses}
        Phi_minus = {s: xp.var(vartype = xp.continuous) for s in data.buses}
        prob.addVariable(Phi_plus, Phi_minus)

    if alnowibet == 2:
        z = {w: xp.var(vartype = xp.binary) for w in data.scenarios}
        
        Psi_plus = {(w,k) : xp.var(vartype = xp.continuous) 
                    for w in data.scenarios for k in data.lines}
        
        Psi_minus = {(w,k): xp.var(vartype = xp.continuous) 
                     for w in data.scenarios for k in data.lines}
        
        Xi = {k : xp.var(vartype = xp.continuous) for k in data.lines}

        prob.addVariable(z, Psi_plus, Psi_minus, Xi)


    # PRIMAL LOWER-LEVEL VARIABLES
    # ----------------------------
    # "d_{jh}(w)" : power consumed by hth block of the jth demand in scenario w.
    # N.B. This is indexed as d[w, j, h].
    d = {k: xp.var() for k in data.d_max_jh.keys()}
    var_map['d'] = d

    # "g_{ib}(w)" : power produced by the bth block of the ith gen. unit in scen. w.
    # N.B. This is indexed as g[w, i, b].
    g = {k: xp.var() for k in data.g_max.keys()}
    var_map['g'] = g

    # "f_{k}(w)" : power flow through line k in scenario w.
    # N.B. This is indexed as f[w, k]
    f = {(w,k): xp.var(vartype = xp.continuous, lb = -xp.infinity) 
        for w in data.scenarios for k in data.lines}
    var_map['f'] = f
    # N.B. The index of line_data is just a single numnber for each line.

    # "r_{j}(w)" : load shed by the jth customer in scenario w.
    # N.B. This is indexed as r[w,j].
    r = {(w,j): xp.var(vartype = xp.continuous)
        for w in data.scenarios for j in data.customers}
    var_map['r'] = r

    # "theta_{s}(w)" : voltage angle at bus s in scenario w
    # N.B. This is indexed as theta[w,s].
    theta = {(w, s): xp.var(vartype = xp.continuous, lb = -xp.infinity)
            for w in data.scenarios for s in data.buses}
    var_map['theta'] = theta

    # Add the variables to the problem
    prob.addVariable(d, g, f, r, theta)

    if bilevel:
        # DUAL LOWER-LEVEL VARIABLES
        # --------------------------
        # "lambda_{s}(w)" : dual of the power flow constraint (free)
        # N.B. This is indexed as lambda_s[w,s].
        lambda_s = {(w,s): xp.var(lb = -xp.infinity) 
                    for w in data.scenarios for s in data.buses}
        var_map['lambda_s'] = lambda_s

        # "phi_{k}(w)" : dual of the line flow/voltage angle constraint (free)
        # N.B. This is indexed as phi[w, k]
        phi = {(w,k): xp.var(lb = -xp.infinity) 
            for w in data.scenarios for k in data.lines}
        var_map['phi'] = phi

        # "phi_{k}^{max}(w)" : dual of the forwards line flow limit constraint (>= 0)
        # N.B. This is indexed as phi_max[w,k]
        phi_max = {(w,k): xp.var() for w in data.scenarios for k in data.lines}
        var_map['phi_max'] = phi_max

        # "phi_{k}^{min}(w)" : dual of the backwards line flow limit constraint (<= 0)
        # N.B. This is indexed as phi_min[w,k]
        phi_min = {(w,k): xp.var(lb = -xp.infinity, ub=0) 
                for w in data.scenarios for k in data.lines}
        var_map['phi_min'] = phi_min

        # "varphi_{ib}^{max}(w)" : dual of the [generation from block] <= [Block_Size]
        # constraint (>= 0)
        # N.B. This is indexed as varphi_max[w, i, b]
        varphi_max = {k: xp.var() for k in data.g_max.keys()}
        var_map['varphi_max'] = varphi_max
        # Recall that g_max contains the block sizes of offers and is indexed by 
        # (scenario, generating_unit, block)

        # "beta_{jh}^{max}(w)" : dual of the [consumption from block] <= [Block_Size]
        # constraint (>= 0)
        # N.B. This is indexed as beta_max[w,j,h]
        beta_max = {k: xp.var() for k in data.d_max_jh.keys()}
        var_map['beta_max'] = beta_max
        # Recall that d_max_jh contains the block sizes of bids and is indexed by 
        # (scenario, customer, block)

        # "alpha_{j}^{max}(w)" : dual of the maximum load shed constraint (>= 0)
        # N.B. This is indexed as alpha_max[w,j]
        alpha_max = {(w,j): xp.var() for w in data.scenarios for j in data.customers}
        var_map['alpha_max'] = alpha_max

        # "rho_{j}(w)" : dual of the constraint on minimum consumption for each customer
        # (<= 0)
        # N.B. This is indexed as rho[w,j]
        rho = {(w,j): xp.var(lb = -xp.infinity, ub = 0) 
            for w in data.scenarios for j in data.customers}
        var_map['rho'] = rho

        # "xi_{s}^{max}(w)" : dual of the [voltage angle] <= pi constraint (>= 0)
        # N.B. This is indexed as xi[w,s]
        xi_max = {(w,s): xp.var() for w in data.scenarios for s in data.buses}
        var_map['xi_max'] = rho

        # "xi_{s}^{min}(w)" : dual of the [voltage angle] >= -pi constraint (<= 0)
        # N.B. This is indexed as xi[w,s]
        xi_min = {(w,s): xp.var(lb = -xp.infinity, ub = 0) 
                for w in data.scenarios for s in data.buses}
        var_map['xi_min'] = xi_min

        # "chi_s(w)" : dual of the reference bus voltage angle == 0 constraint (free)
        chi = {w: xp.var(lb = -xp.infinity) for w in data.scenarios}
        var_map['chi'] = chi

        # AUXILIARY VARIABLE
        # ------------------
        # "phi_{k}^{-}(w)" : used to linearize dual constraints (26) and (27) (free)
        phi_minus = {(w,k): xp.var(lb = -xp.infinity, ub = xp.infinity)
                    for w in data.scenarios for k in data.lines}
        var_map['phi_minus'] = phi_minus

        prob.addVariable(lambda_s, phi, phi_max, phi_min, varphi_max, beta_max, 
                        alpha_max, rho, xi_max, xi_min, chi, phi_minus)

    # OBJECTIVE FUNCTION
    # ------------------
    # Calculate the social-welfare benefit and load-shed costs in each scenario:
    sw_ls = {w: xp.Sum(data.lambda_D[k[1],k[2]] * d[k] for k in data.d_max_jh.keys() if k[0] == w) -\
                        xp.Sum(data.lambda_G[k[1],k[2]] * g[k] for k in data.g_max.keys() if k[0] == w) -\
                        xp.Sum(data.c_U[j] * r[w,j] for j in data.customers) for w in data.scenarios}
    
    var_map['sw_ls'] = sw_ls

    # sw_ls = xp.Sum(data.delta[w]*(xp.Sum(data.lambda_D[k[1],k[2]] * d[k] for k in data.d_max_jh.keys() if k[0] == w) -\
    #                     xp.Sum(data.lambda_G[k[1],k[2]] * g[k] for k in data.g_max.keys() if k[0] == w)) for w in data.scenarios)

    # Transmission investment costs:
    tics = xp.Sum(data.c[k]*x[k] for k in data.prospective_lines)
    var_map['tics'] = tics

    # Objective function:

    if not alternative_objective:
        # PROBABILITY-WEIGHTED LOWER LEVEL VALUE - TRANSMISSION INVESTMENT COSTS
        objective = data.sigma*xp.Sum(data.delta[w] * sw_ls[w] for w in data.scenarios) - tics
    
    else:
        # Alternative objective
        print("USING ALTERNATIVE OBJECTIVE")
        objective = - data.sigma*xp.Sum(data.delta[w] * xp.Sum(data.c_U[j] * r[w,j] for j in data.customers) for w in data.scenarios) - tics

    prob.setObjective(objective, sense=xp.maximize)

    # UPPER-LEVEL CONSTRAINTS
    # -----------------------
    # (Eqn 2) : Transmission investment budget constraint
    budget_ctr = xp.Sum(data.c[k]*x[k] for k in data.prospective_lines) <= data.c_max
    prob.addConstraint(budget_ctr)

    # # (Eqn 3) : All existing lines are "open" <- Needs to be done before x added to problem
    # for k in data.existing_lines:
    #     x[k].lb = 1

    # N.B. (Eqn 4), that the x[k] are all binary decisions, is established when
    # the variables are declared and does not need to be written here.

    # LOWER-LEVEL PRIMAL CONSTRAINTS
    # ------------------------------
    # N.B. (Eqn 5), the objective of the lower level is NOT required in this model,
    # as we guarantee an optimal solution with the strong duality constraint.

    # (Eqn 6) : Power flow at each bus
    PF_ctr = {} # dictionary to be indexed by scenario and bus

    for w in data.scenarios:
        for s in data.buses:

            # Calculate amount generated at bus:
            generated_at_bus = xp.Sum(
                                        xp.Sum(g[k] for k in data.g_max.keys() if k[0]==w and k[1]==i) 
                                    for i in data.generators_at_bus[s])

            flows_out_of_bus = data.line_data[data.line_data['From']==s].index
            flow_out = xp.Sum(f[w, k] for k in flows_out_of_bus)

            flows_into_bus = data.line_data[data.line_data['To']==s].index
            flow_in = xp.Sum(f[w, k] for k in flows_into_bus)

            load_shed_at_bus = xp.Sum(r[w,j] for j in data.customers_at_bus[s])

            # Calculate the amount consumed at bus:
            demand_blocks_at_bus = data.bid_data[(data.bid_data['Bus']==s) & (data.bid_data['Scenario']==w)].index
            consumed_at_bus = xp.Sum(d[w, data.bid_data.loc[i,'Customer'], data.bid_data.loc[i,'Block']] 
                                    for i in demand_blocks_at_bus)

            # Assert power flow is balanced.
            PF_ctr[(w,s)] = generated_at_bus - flow_out + flow_in + load_shed_at_bus - consumed_at_bus == 0

    prob.addConstraint(PF_ctr)


    # N.B. (Eqn 7, 8, 9) for Kirchoff's Voltage Law and preventing line overloading
    # are non-linear and replaced in this model by (Eqn 40 and 41).

    # (Eqn 40): Lines are not overloaded
    fmax_linctr = {(w,k): f[w,k] <= x[k] * data.f_max[k] 
                for w in data.scenarios for k in data.lines}

    fmin_linctr = {(w,k): - x[k] * data.f_max[k] <= f[w,k] 
                for w in data.scenarios for k in data.lines}


    # (Eqn 41): "Linearized" KVL constraints
    KVL_linctr_1 = {(w,k): 
                    -(1-x[k]) * data.M <= (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) 
                    for w in data.scenarios for k in data.lines}

    KVL_linctr_2 = {(w,k):
                    (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) <= (1-x[k]) * data.M
                    for w in data.scenarios for k in data.lines}

    prob.addConstraint(fmax_linctr, fmin_linctr, KVL_linctr_1, KVL_linctr_2)


    # (Eqn 10) : generation does not exceed block offered (ATTEMPT 2)
    gen_ctr = {k : g[k] <= data.g_max[k] for k in data.g_max.keys()}
    # Recall that data.g_max is indexed by [Scenario, Generating_Unit, Block]

    prob.addConstraint(gen_ctr)


    # (Eqn 11): consumption does not exceed block bid (ATTEMPT 2)
    con_ctr = {k: d[k] <= data.d_max_jh[k] for k in data.d_max_jh.keys()}
    # Recall that data.d_max_jh is indexed by [Scenario, Generating_Unit, Block]

    prob.addConstraint(con_ctr)


    # (Eqn 12): load shed does not exceed amount of demand from that customer
    shed_ctr = {}

    for w in data.scenarios:
        for j in data.customers:
            shed_ctr[w, j] = r[w, j] <= data.d_max[w,j]

    prob.addConstraint(shed_ctr)


    # (Eqn 13): minimum power always provided (90% of demand in Garces et al.)
    # TODO: This looks like it could be tidied up a bit...
    min_ctr = {}

    for w in data.scenarios:
        for j in data.customers:

            selector = (data.bid_data['Scenario']==w) & (data.bid_data['Customer']==j)
            demand_blocks_at_customer = data.bid_data[selector].index

            min_ctr[w,j] = xp.Sum(d[w,j,data.bid_data.loc[index,'Block']] 
                                for index in demand_blocks_at_customer) >= data.d_min[w,j]
            
    prob.addConstraint(min_ctr)


    # (Eqn 14, 15, 16): Restricting the voltage angle to [-pi, pi]:
    angle_max_ctr = {}
    angle_min_ctr = {}
    ref_bus_ctr = {}

    for w in data.scenarios:
        for s in data.buses:
            angle_max_ctr[w,s] = theta[w,s] <= np.pi
            angle_min_ctr[w,s] = theta[w,s] >= - np.pi
            ref_bus_ctr[w] = theta[w,data.ref_bus] == 0

    prob.addConstraint(angle_max_ctr, angle_min_ctr, ref_bus_ctr)


    # N.B. (Eqn 17, 18, 19) are domain constraints on g, r, and d (all >= 0) which
    # come for free from the Xpress defaults.

    if bilevel:
        # LOWER-LEVEL DUAL CONSTRAINTS
        # ----------------------------
        # (Eqn 22) : dual constraint associated with primal variable g
        g_dualctr = {k: 
                    lambda_s[k[0], data.s_i[k[1]]] + varphi_max[k] >= -data.lambda_G[k[1],k[2]] 
                    for k in data.g_max.keys()
                    }
        # The indexing is a bit awkward here. k = (w, i, b) and we are taking:
        # lambda_s[w,s_i[i]] <- the dual variable for scenario w and the bus where i is
        # varphi_max[w,i,b] <- the dual variable for scenario, i, and block
        # lambda_G[i,b]

        # (Eqn 23) : dual constraint associated with primal variable d
        d_dualctr = {k:
                    - lambda_s[k[0], data.s_j[k[1]]] + beta_max[k] + rho[k[0],k[1]] >= data.lambda_D[k[1],k[2]]
                    for k in data.d_max_jh.keys()
                    }

        # (Eqn 24) : dual constraint associated with primal variable r
        r_dualctr = {(w,j):
                    lambda_s[w,data.s_j[j]] + alpha_max[w,j] >= -data.c_U[j]
                    for w in data.scenarios for j in data.generating_units             
                    }

        # (Eqn 25) : dual constraint associated with primal variable f
        f_dualctr = {(w,k):
                    - lambda_s[w,data.o[k]] + lambda_s[w,data.r[k]] + phi[w,k] + phi_max[w,k] + phi_min[w,k] == 0
                    for w in data.scenarios for k in data.lines
                    }

        # N.B. (Eqn 26 and 27), nonlinear dual constraints, are replaced in this model
        # by (Eqn 42, 43, 44, and 45). This requires the introduction of an auxiliary
        # variable denoted "phi_{k}^{-}(w)" in the paper.

        # (Eqn 42) : 
        theta_lindualctr_1 = {(w,s): - xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.o[k] == s) 
                            + xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.r[k] == s) 
                            + xi_max[w,s] + xi_min[w,s] == 0 
                            for w in data.scenarios for s in data.buses if s != data.ref_bus}

        # (Eqn 43) :
        theta_lindualctr_2 = {w:
                        - xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.o[k] == data.ref_bus) 
                        + xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.r[k] == data.ref_bus)
                        + chi[w] == 0
                        for w in data.scenarios
                        }

        # (Eqn 44) :
        theta_lindualctr_3 = {(w,k): -x[k] * data.Gamma_max <= phi[w,k] - phi_minus[w,k]
                            for w in data.scenarios for k in data.lines}

        theta_lindualctr_4 = {(w,k): phi[w,k] - phi_minus[w,k]<= x[k] * data.Gamma_max
                            for w in data.scenarios for k in data.lines}

        # (Eqn 45) :
        theta_lindualctr_5 = {(w,k): -(1 - x[k]) * data.Gamma_max <= phi_minus[w,k]
                            for w in data.scenarios for k in data.lines}

        theta_lindualctr_6 = {(w,k): phi_minus[w,k] <= (1 - x[k]) * data.Gamma_max
                            for w in data.scenarios for k in data.lines}


        # (Eqn 28 to 34) are domain constraints accounted for in variable declarations.
        # (Eqn 35) : strong duality theorem (SDT) constraint
        # The LHS of the SDT constraint is [social_welfare] - [load shed penalty] in
        # each scenario.
        SDT_ctr = {w :
                    sw_ls[w] == xp.Sum((phi_max[w,k] - phi_min[w,k])*data.f_max[k] for k in data.lines) +\
                                xp.Sum(varphi_max[k] * data.g_max[k] for k in data.g_max.keys() if k[0] == w) +\
                                xp.Sum(beta_max[k] * data.d_max_jh[k] for k in data.d_max_jh.keys() if k[0] == w) +\
                                xp.Sum((alpha_max[w, j] * data.d_max[w,j] + rho[w,j] * data.d_min[w,j]) for j in data.customers) +\
                                xp.Sum(np.pi*(xi_max[w,s] - xi_min[w,s]) for s in data.buses)
                    for w in data.scenarios
                    }
        # QUERY: Should the final sum exclude the reference bus?

        prob.addConstraint(g_dualctr, d_dualctr, f_dualctr, r_dualctr, SDT_ctr, theta_lindualctr_1,
                        theta_lindualctr_2, theta_lindualctr_3, theta_lindualctr_4, 
                        theta_lindualctr_5, theta_lindualctr_6)
        

        # ALNOWIBET MARKET POWER INDICES
        # ------------------------------
        # The expressions given here match with Alnowibet (2023).
        # N.B. Alnowibet uses s for scenarios and i \in N for buses; here
        # I stick with the established notation of w for scenarios and s
        # for buses.
        lambda_s_bar = {s : 
                        xp.Sum(data.delta[w] * (-lambda_s[w,s]) 
                                for w in data.scenarios) 
                        for s in data.buses}
        var_map['lambda_s_bar'] = lambda_s_bar
        
        lambda_bar = xp.Sum(lambda_s_bar[s] for s in data.buses)/len(data.buses)
        var_map['lambda_bar'] = lambda_bar

        if alnowibet == 1:
            Lambda_s = {s: Phi_plus[s] + Phi_minus[s] for s in data.buses}

            # Create non-negative variables for the deviation (I haven't seen
            # reason to add the actual Phi variable, as everything is done with
            # the split variables instead):
                # Phi[s] = Phi_plus[s] - Phi_minus[s]
            alnowibet_split = {s :
                            Phi_plus[s] - Phi_minus[s] == lambda_s_bar[s] - lambda_bar 
                            for s in data.buses}
            
            alnowibet_ctr = xp.Sum(Lambda_s[s] for s in data.buses) <= data.PDI_max * len(data.buses) * lambda_bar

            prob.addConstraint(alnowibet_split, alnowibet_ctr)
            print("Added Alnowibet PDI Market Power constraint.")

        if alnowibet == 2: 
            NUI_ctr1 = xp.Sum(z[w] for w in data.scenarios) == 1

            NUI_ctr2 = xp.Sum(Xi[k] for k in data.lines) <= data.NUI_max * xp.Sum(x[k] * data.f_max[k] for k in data.lines)

            NUI_ctr3 = {k: Psi_plus[w,k] + Psi_minus[w,k] <= Xi[k] for k in data.lines}

            NUI_ctr4 = {(w,k): Xi[k] <= Psi_plus[w,k] + Psi_minus[w,k] + (1 - z[w]) * data.M_Alnowibet 
                        for w in data.scenarios for k in data.lines}

            NUI_ctr5 = {(w,k): f[w,k] == Psi_plus[w,k] - Psi_minus[w,k]
                        for w in data.scenarios for k in data.lines}

            prob.addConstraint(NUI_ctr1, NUI_ctr2, NUI_ctr3, NUI_ctr4, NUI_ctr5)
            var_map["Xi"] = Xi
            var_map["Psi_plus"] = Psi_plus
            var_map["Psi_minus"] = Psi_minus

    return prob, var_map

def run_model(data, model_params = {}, solver_options = {}, wind = False):
    '''
    Builds and solves model using data, model parameters, and solver options.

    Args:
        data: an object, created by the function load_data() in the data module,
            which contains the information required for the specific instance
            to be solved.
        model_params: (OPTIONAL) a dictionary of keyword arguments for building 
            the model.
        solver_options: (OPTIONAL) a dictionary of keyword arguments for
            specifying the controls/behaviour of the solver.
            'outputlog': integer specifying detail level in the solver outputlog

    Returns:
        sol_dict: a dictionary of the values from the model at the optimal soln. 
    '''
    if wind:
        prob, var_map = build_wind_model(data=data, **model_params)
    else:
        prob, var_map = build_model(data=data, **model_params)

    # Control how detailed the output log is:
    solver_output_level = solver_options.get("outputlog")
    if not solver_output_level is None:
        prob.controls.outputlog = solver_output_level
    
    prob.solve()

    sol_dict = {k: prob.getSolution(v) for k, v in var_map.items()}
    return(sol_dict)


def build_LL(data, new_lines=[], alternative_objective=False):
    '''Build the lower level PRIMAL ONLY problem.'''

    # Create the Xpress problem:
    prob = xp.problem("Lower Level Primal Problem")
    var_map = {}

    # UPPER-LEVEL "VARIABLES"
    # -----------------------
    # "x_k" : binary variable, 1 if line k is built, 0 otherwise.
    # Here, in JUST THE LOWER LEVEL, x is actually a dictionary of PARAMETERS.
    x = {(k): 1 if (k in new_lines) or (k in data.existing_lines) else 0 for k in data.lines}
    # var_map['x'] = x

    # PRIMAL LOWER-LEVEL VARIABLES
    # ----------------------------
    # "d_{jh}(w)" : power consumed by hth block of the jth demand in scenario w.
    # N.B. This is indexed as d[w, j, h].
    d = {k: xp.var() for k in data.d_max_jh.keys()}
    var_map['d'] = d

    # "g_{ib}(w)" : power produced by the bth block of the ith gen. unit in scen. w.
    # N.B. This is indexed as g[w, i, b].
    g = {k: xp.var() for k in data.g_max.keys()}
    var_map['g'] = g

    # "f_{k}(w)" : power flow through line k in scenario w.
    # N.B. This is indexed as f[w, k]
    f = {(w,k): xp.var(vartype = xp.continuous, lb = -xp.infinity) 
        for w in data.scenarios for k in data.lines}
    var_map['f'] = f
    # N.B. The index of line_data is just a single numnber for each line.

    # "r_{j}(w)" : load shed by the jth customer in scenario w.
    # N.B. This is indexed as r[w,j].
    r = {(w,j): xp.var(vartype = xp.continuous)
        for w in data.scenarios for j in data.customers}
    var_map['r'] = r

    # "theta_{s}(w)" : voltage angle at bus s in scenario w
    # N.B. This is indexed as theta[w,s].
    theta = {(w, s): xp.var(vartype = xp.continuous, lb = -xp.infinity)
            for w in data.scenarios for s in data.buses}
    var_map['theta'] = theta

    # Add the variables to the problem
    prob.addVariable(d, g, f, r, theta)

    # OBJECTIVE FUNCTION
    # ------------------
    # Calculate the social-welfare benefit and load-shed costs in each scenario:
    sw_ls = {w: xp.Sum(data.lambda_D[k[1],k[2]] * d[k] for k in data.d_max_jh.keys() if k[0] == w) -\
                        xp.Sum(data.lambda_G[k[1],k[2]] * g[k] for k in data.g_max.keys() if k[0] == w) -\
                        xp.Sum(data.c_U[j] * r[w,j] for j in data.customers) for w in data.scenarios}    
    var_map['sw_ls'] = sw_ls

    # Objective function:
    # (Lower level only cares about welfare and load-shedding.)
    if not alternative_objective:
        # PROBABILITY-WEIGHTED LOWER LEVEL VALUE
        objective = data.sigma*xp.Sum(data.delta[w] * sw_ls[w] for w in data.scenarios)
    
    else:
        # Alternative objective
        print("USING ALTERNATIVE OBJECTIVE")
        objective = - data.sigma*xp.Sum(data.delta[w] * xp.Sum(data.c_U[j] * r[w,j] for j in data.customers) for w in data.scenarios)

    prob.setObjective(objective, sense=xp.maximize)

    # UPPER-LEVEL CONSTRAINTS
    # -----------------------
    # These are not required here and will be handled in the loop over feasible
    # transmission decisions.

    # LOWER-LEVEL PRIMAL CONSTRAINTS
    # ------------------------------
    # N.B. (Eqn 5) is the objective of the lower level problem.

    # (Eqn 6) : Power flow at each bus
    PF_ctr = {} # dictionary to be indexed by scenario and bus

    for w in data.scenarios:
        for s in data.buses:

            # Calculate amount generated at bus:
            generated_at_bus = xp.Sum(
                                        xp.Sum(g[k] for k in data.g_max.keys() if k[0]==w and k[1]==i) 
                                    for i in data.generators_at_bus[s])
            # TODO: there must be a better way to do this ^

            flows_out_of_bus = data.line_data[data.line_data['From']==s].index
            flow_out = xp.Sum(f[w, k] for k in flows_out_of_bus)

            flows_into_bus = data.line_data[data.line_data['To']==s].index
            flow_in = xp.Sum(f[w, k] for k in flows_into_bus)

            load_shed_at_bus = xp.Sum(r[w,j] for j in data.customers_at_bus[s])

            # Calculate the amount consumed at bus:
            demand_blocks_at_bus = data.bid_data[(data.bid_data['Bus']==s) & (data.bid_data['Scenario']==w)].index
            consumed_at_bus = xp.Sum(d[w, data.bid_data.loc[i,'Customer'], data.bid_data.loc[i,'Block']] 
                                    for i in demand_blocks_at_bus)

            # Assert power flow is balanced.
            PF_ctr[(w,s)] = generated_at_bus - flow_out + flow_in + load_shed_at_bus - consumed_at_bus == 0

    prob.addConstraint(PF_ctr)


    # N.B. (Eqn 7, 8, 9) for Kirchoff's Voltage Law and preventing line overloading
    # are non-linear and replaced in this model by (Eqn 40 and 41).

    # (Eqn 40): Lines are not overloaded
    fmax_linctr = {(w,k): f[w,k] <= x[k] * data.f_max[k] 
                for w in data.scenarios for k in data.lines}

    fmin_linctr = {(w,k): - x[k] * data.f_max[k] <= f[w,k] 
                for w in data.scenarios for k in data.lines}


    # (Eqn 41): "Linearized" KVL constraints
    KVL_linctr_1 = {(w,k): 
                    -(1-x[k]) * data.M <= (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) 
                    for w in data.scenarios for k in data.lines}

    KVL_linctr_2 = {(w,k):
                    (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) <= (1-x[k]) * data.M
                    for w in data.scenarios for k in data.lines}

    prob.addConstraint(fmax_linctr, fmin_linctr, KVL_linctr_1, KVL_linctr_2)


    # (Eqn 10) : generation does not exceed block offered (ATTEMPT 2)
    gen_ctr = {k : g[k] <= data.g_max[k] for k in data.g_max.keys()}
    # Recall that data.g_max is indexed by [Scenario, Generating_Unit, Block]

    prob.addConstraint(gen_ctr)


    # (Eqn 11): consumption does not exceed block bid (ATTEMPT 2)
    con_ctr = {k: d[k] <= data.d_max_jh[k] for k in data.d_max_jh.keys()}
    # Recall that data.d_max_jh is indexed by [Scenario, Generating_Unit, Block]

    prob.addConstraint(con_ctr)


    # (Eqn 12): load shed does not exceed amount of demand from that customer
    # TODO: Establish why this is necessary... (it's not clear to me how this would
    # end up being a tight constraint...)
    shed_ctr = {}

    for w in data.scenarios:
        for j in data.customers:
            shed_ctr[w, j] = r[w, j] <= data.d_max[w,j]

    prob.addConstraint(shed_ctr)


    # (Eqn 13): minimum power always provided (90% of demand in Garces et al.)
    # TODO: This looks like it could be tidied up a bit...
    min_ctr = {}

    for w in data.scenarios:
        for j in data.customers:

            selector = (data.bid_data['Scenario']==w) & (data.bid_data['Customer']==j)
            demand_blocks_at_customer = data.bid_data[selector].index

            min_ctr[w,j] = xp.Sum(d[w,j,data.bid_data.loc[index,'Block']] 
                                for index in demand_blocks_at_customer) >= data.d_min[w,j]
            
    prob.addConstraint(min_ctr)


    # (Eqn 14, 15, 16): Restricting the voltage angle to [-pi, pi]:
    angle_max_ctr = {}
    angle_min_ctr = {}
    ref_bus_ctr = {}

    for w in data.scenarios:
        for s in data.buses:
            angle_max_ctr[w,s] = theta[w,s] <= np.pi
            angle_min_ctr[w,s] = theta[w,s] >= - np.pi
            ref_bus_ctr[w] = theta[w,data.ref_bus] == 0

    prob.addConstraint(angle_max_ctr, angle_min_ctr, ref_bus_ctr)

    # N.B. (Eqn 17, 18, 19) are domain constraints on g, r, and d (all >= 0) which
    # come for free from the Xpress defaults.

    return prob, var_map

def run_LL(data, model_params = {}, solver_options = {}):
    '''
    Builds and solves LOWER LEVEL MODEL for a given a list of new_lines.
    Uses the PRIMAL CONSTRAINTS ONLY because it calls build_LL().

    Args:
        data: an object, created by the function load_data() in the data module,
            which contains the information required for the specific instance
            to be solved.
        new_lines: (OPTIONAL) a list of new lines whcih are open for this 
            particular lower level problem.
        model_params: (OPTIONAL) a dictionary of keyword arguments for building 
            the model.
        solver_options: (OPTIONAL) a dictionary of keyword arguments for
            specifying the controls/behaviour of the solver.
            'outputlog': integer specifying detail level in the solver outputlog

    Returns:
        sol_dict: a dictionary of the values from the model at the optimal soln. 
    '''

    prob, var_map = build_LL(data=data, **model_params)

    # Control how detailed the output log is:
    solver_output_level = solver_options.get("outputlog")
    if not solver_output_level is None:
        prob.controls.outputlog = solver_output_level
    
    prob.solve()

    sol_dict = {k: prob.getSolution(v) for k, v in var_map.items()}
    return(sol_dict)

def build_LLD(data, new_lines=[], include_primal = True):
    '''Beware: this currently sets the lower-level dual objective to maximize
    phi[1,1] rather than the true dual objective. The purpose is to identify
    if the dual feasible region is unbounded.
    '''

    # Create the Xpress problem:
    prob = xp.problem("Lower Level")
    var_map = {}

    # UPPER-LEVEL "VARIABLES"
    # -----------------------
    # "x_k" : binary variable, 1 if line k is built, 0 otherwise.
    # Here, in JUST THE LOWER LEVEL, x is actually a dictionary of PARAMETERS.
    x = {(k): 1 if (k in new_lines) or (k in data.existing_lines) else 0 for k in data.lines}
    

    # DUAL LOWER-LEVEL VARIABLES
    # --------------------------
    # "lambda_{s}(w)" : dual of the power flow constraint (free)
    # N.B. This is indexed as lambda_s[w,s].
    lambda_s = {(w,s): xp.var(lb = -xp.infinity) 
                for w in data.scenarios for s in data.buses}
    var_map['lambda_s'] = lambda_s

    # "phi_{k}(w)" : dual of the line flow/voltage angle constraint (free)
    # N.B. This is indexed as phi[w, k]
    phi = {(w,k): xp.var(lb = -xp.infinity) 
        for w in data.scenarios for k in data.lines}
    var_map['phi'] = phi

    # "phi_{k}^{max}(w)" : dual of the forwards line flow limit constraint (>= 0)
    # N.B. This is indexed as phi_max[w,k]
    phi_max = {(w,k): xp.var() for w in data.scenarios for k in data.lines}
    var_map['phi_max'] = phi_max

    # "phi_{k}^{min}(w)" : dual of the backwards line flow limit constraint (<= 0)
    # N.B. This is indexed as phi_min[w,k]
    phi_min = {(w,k): xp.var(lb = -xp.infinity, ub=0) 
            for w in data.scenarios for k in data.lines}
    var_map['phi_min'] = phi_min

    # "varphi_{ib}^{max}(w)" : dual of the [generation from block] <= [Block_Size]
    # constraint (>= 0)
    # N.B. This is indexed as varphi_max[w, i, b]
    varphi_max = {k: xp.var() for k in data.g_max.keys()}
    var_map['varphi_max'] = varphi_max
    # Recall that g_max contains the block sizes of offers and is indexed by 
    # (scenario, generating_unit, block)

    # "beta_{jh}^{max}(w)" : dual of the [consumption from block] <= [Block_Size]
    # constraint (>= 0)
    # N.B. This is indexed as beta_max[w,j,h]
    beta_max = {k: xp.var() for k in data.d_max_jh.keys()}
    var_map['beta_max'] = beta_max
    # Recall that d_max_jh contains the block sizes of bids and is indexed by 
    # (scenario, customer, block)

    # "alpha_{j}^{max}(w)" : dual of the maximum load shed constraint (>= 0)
    # N.B. This is indexed as alpha_max[w,j]
    alpha_max = {(w,j): xp.var() for w in data.scenarios for j in data.customers}
    var_map['alpha_max'] = alpha_max

    # "rho_{j}(w)" : dual of the constraint on minimum consumption for each customer
    # (<= 0)
    # N.B. This is indexed as rho[w,j]
    rho = {(w,j): xp.var(lb = -xp.infinity, ub = 0) 
        for w in data.scenarios for j in data.customers}
    var_map['rho'] = rho

    # "xi_{s}^{max}(w)" : dual of the [voltage angle] <= pi constraint (>= 0)
    # N.B. This is indexed as xi[w,s]
    xi_max = {(w,s): xp.var() for w in data.scenarios for s in data.buses}
    var_map['xi_max'] = rho

    # "xi_{s}^{min}(w)" : dual of the [voltage angle] >= -pi constraint (<= 0)
    # N.B. This is indexed as xi[w,s]
    xi_min = {(w,s): xp.var(lb = -xp.infinity, ub = 0) 
            for w in data.scenarios for s in data.buses}
    var_map['xi_min'] = xi_min

    # "chi_s(w)" : dual of the reference bus voltage angle == 0 constraint (free)
    chi = {w: xp.var(lb = -xp.infinity) for w in data.scenarios}
    var_map['chi'] = chi

    # Auxilliary variable phi^{-} not to be used here.

    prob.addVariable(lambda_s, phi, phi_max, phi_min, varphi_max, beta_max, 
                    alpha_max, rho, xi_max, xi_min, chi)


    # LOWER-LEVEL DUAL CONSTRAINTS
    # ----------------------------
    # (Eqn 22) : dual constraint associated with primal variable g
    g_dualctr = {k: 
                lambda_s[k[0], data.s_i[k[1]]] + varphi_max[k] >= -data.lambda_G[k[1],k[2]] 
                for k in data.g_max.keys()
                }
    # The indexing is a bit awkward here. k = (w, i, b) and we are taking:
    # lambda_s[w,s_i[i]] <- the dual variable for scenario w and the bus where i is
    # varphi_max[w,i,b] <- the dual variable for scenario, i, and block
    # lambda_G[i,b]

    # (Eqn 23) : dual constraint associated with primal variable d
    d_dualctr = {k:
                - lambda_s[k[0], data.s_j[k[1]]] + beta_max[k] + rho[k[0],k[1]] >= data.lambda_D[k[1],k[2]]
                for k in data.d_max_jh.keys()
                }

    # (Eqn 24) : dual constraint associated with primal variable r
    r_dualctr = {(w,j):
                lambda_s[w,data.s_j[j]] + alpha_max[w,j] >= -data.c_U[j]
                for w in data.scenarios for j in data.generating_units             
                }

    # (Eqn 25) : dual constraint associated with primal variable f
    f_dualctr = {(w,k):
                - lambda_s[w,data.o[k]] + lambda_s[w,data.r[k]] + phi[w,k] + phi_max[w,k] + phi_min[w,k] == 0
                for w in data.scenarios for k in data.lines
                }


    # (Eqn 26) : dual constraint associated with the primal variable theta
    theta_dualctr_1 = {(w,s): - xp.Sum(data.b_MW[k]*x[k]*phi[w,k] for k in data.lines if data.o[k] == s) 
                         + xp.Sum(data.b_MW[k]*x[k]*phi[w,k] for k in data.lines if data.r[k] == s) 
                         + xi_max[w,s] + xi_min[w,s] == 0 
                         for w in data.scenarios for s in data.buses if s != data.ref_bus}
    
    
    # (Eqn 27) : dual constraint associated with the primal variable theta (at
    #               the reference bus).
    theta_dualctr_2 = {w:
                     - xp.Sum(data.b_MW[k]*x[k]*phi[w,k] for k in data.lines if data.o[k] == data.ref_bus) 
                     + xp.Sum(data.b_MW[k]*x[k]*phi[w,k] for k in data.lines if data.r[k] == data.ref_bus)
                     + chi[w] == 0
                     for w in data.scenarios
                     }

    # SDT constraint not required here (single level)
    # # (Eqn 35) : strong duality theorem (SDT) constraint
    # # The LHS of the SDT constraint is [social_welfare] - [load shed penalty] in
    # # each scenario.
    # SDT_ctr = {w :
    #             sw_ls[w] == xp.Sum((phi_max[w,k] - phi_min[w,k])*data.f_max[k] for k in data.lines) +\
    #                         xp.Sum(varphi_max[k] * data.g_max[k] for k in data.g_max.keys() if k[0] == w) +\
    #                         xp.Sum(beta_max[k] * data.d_max_jh[k] for k in data.d_max_jh.keys() if k[0] == w) +\
    #                         xp.Sum((alpha_max[w, j] * data.d_max[w,j] + rho[w,j] * data.d_min[w,j]) for j in data.customers) +\
    #                         xp.Sum(np.pi*(xi_max[w,s] - xi_min[w,s]) for s in data.buses)
    #             for w in data.scenarios
    #             }

    prob.addConstraint(g_dualctr, d_dualctr, f_dualctr, r_dualctr, 
                       theta_dualctr_1, theta_dualctr_2)
    

    if include_primal:
    # PRIMAL LOWER-LEVEL VARIABLES
    # ----------------------------
        # "d_{jh}(w)" : power consumed by hth block of the jth demand in scenario w.
        # N.B. This is indexed as d[w, j, h].
        d = {k: xp.var() for k in data.d_max_jh.keys()}
        var_map['d'] = d

        # "g_{ib}(w)" : power produced by the bth block of the ith gen. unit in scen. w.
        # N.B. This is indexed as g[w, i, b].
        g = {k: xp.var() for k in data.g_max.keys()}
        var_map['g'] = g

        # "f_{k}(w)" : power flow through line k in scenario w.
        # N.B. This is indexed as f[w, k]
        f = {(w,k): xp.var(vartype = xp.continuous, lb = -xp.infinity) 
            for w in data.scenarios for k in data.lines}
        var_map['f'] = f
        # N.B. The index of line_data is just a single numnber for each line.

        # "r_{j}(w)" : load shed by the jth customer in scenario w.
        # N.B. This is indexed as r[w,j].
        r = {(w,j): xp.var(vartype = xp.continuous)
            for w in data.scenarios for j in data.customers}
        var_map['r'] = r

        # "theta_{s}(w)" : voltage angle at bus s in scenario w
        # N.B. This is indexed as theta[w,s].
        theta = {(w, s): xp.var(vartype = xp.continuous, lb = -xp.infinity)
                for w in data.scenarios for s in data.buses}
        var_map['theta'] = theta

        # Add the variables to the problem
        prob.addVariable(d, g, f, r, theta)


        # LOWER-LEVEL PRIMAL CONSTRAINTS
        # ------------------------------
        # N.B. (Eqn 5) is the objective of the lower level problem.

        # (Eqn 6) : Power flow at each bus
        PF_ctr = {} # dictionary to be indexed by scenario and bus

        for w in data.scenarios:
            for s in data.buses:

                # Calculate amount generated at bus:
                generated_at_bus = xp.Sum(
                                            xp.Sum(g[k] for k in data.g_max.keys() if k[0]==w and k[1]==i) 
                                        for i in data.generators_at_bus[s])
                # TODO: there must be a better way to do this ^

                flows_out_of_bus = data.line_data[data.line_data['From']==s].index
                flow_out = xp.Sum(f[w, k] for k in flows_out_of_bus)

                flows_into_bus = data.line_data[data.line_data['To']==s].index
                flow_in = xp.Sum(f[w, k] for k in flows_into_bus)

                load_shed_at_bus = xp.Sum(r[w,j] for j in data.customers_at_bus[s])

                # Calculate the amount consumed at bus:
                demand_blocks_at_bus = data.bid_data[(data.bid_data['Bus']==s) & (data.bid_data['Scenario']==w)].index
                consumed_at_bus = xp.Sum(d[w, data.bid_data.loc[i,'Customer'], data.bid_data.loc[i,'Block']] 
                                        for i in demand_blocks_at_bus)

                # Assert power flow is balanced.
                PF_ctr[(w,s)] = generated_at_bus - flow_out + flow_in + load_shed_at_bus - consumed_at_bus == 0

        prob.addConstraint(PF_ctr)


        # N.B. (Eqn 7, 8, 9) for Kirchoff's Voltage Law and preventing line overloading
        # are non-linear and replaced in this model by (Eqn 40 and 41).

        # (Eqn 40): Lines are not overloaded
        fmax_linctr = {(w,k): f[w,k] <= x[k] * data.f_max[k] 
                    for w in data.scenarios for k in data.lines}

        fmin_linctr = {(w,k): - x[k] * data.f_max[k] <= f[w,k] 
                    for w in data.scenarios for k in data.lines}


        # (Eqn 41): "Linearized" KVL constraints
        KVL_linctr_1 = {(w,k): 
                        -(1-x[k]) * data.M <= (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) 
                        for w in data.scenarios for k in data.lines}

        KVL_linctr_2 = {(w,k):
                        (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) <= (1-x[k]) * data.M
                        for w in data.scenarios for k in data.lines}

        prob.addConstraint(fmax_linctr, fmin_linctr, KVL_linctr_1, KVL_linctr_2)


        # (Eqn 10) : generation does not exceed block offered (ATTEMPT 2)
        gen_ctr = {k : g[k] <= data.g_max[k] for k in data.g_max.keys()}
        # Recall that data.g_max is indexed by [Scenario, Generating_Unit, Block]

        prob.addConstraint(gen_ctr)


        # (Eqn 11): consumption does not exceed block bid (ATTEMPT 2)
        con_ctr = {k: d[k] <= data.d_max_jh[k] for k in data.d_max_jh.keys()}
        # Recall that data.d_max_jh is indexed by [Scenario, Generating_Unit, Block]

        prob.addConstraint(con_ctr)


        # (Eqn 12): load shed does not exceed amount of demand from that customer
        # TODO: Establish why this is necessary... (it's not clear to me how this would
        # end up being a tight constraint...)
        shed_ctr = {}

        for w in data.scenarios:
            for j in data.customers:
                shed_ctr[w, j] = r[w, j] <= data.d_max[w,j]

        prob.addConstraint(shed_ctr)


        # (Eqn 13): minimum power always provided (90% of demand in Garces et al.)
        # TODO: This looks like it could be tidied up a bit...
        min_ctr = {}

        for w in data.scenarios:
            for j in data.customers:

                selector = (data.bid_data['Scenario']==w) & (data.bid_data['Customer']==j)
                demand_blocks_at_customer = data.bid_data[selector].index

                min_ctr[w,j] = xp.Sum(d[w,j,data.bid_data.loc[index,'Block']] 
                                    for index in demand_blocks_at_customer) >= data.d_min[w,j]
                
        prob.addConstraint(min_ctr)


        # (Eqn 14, 15, 16): Restricting the voltage angle to [-pi, pi]:
        angle_max_ctr = {}
        angle_min_ctr = {}
        ref_bus_ctr = {}

        for w in data.scenarios:
            for s in data.buses:
                angle_max_ctr[w,s] = theta[w,s] <= np.pi
                angle_min_ctr[w,s] = theta[w,s] >= - np.pi
                ref_bus_ctr[w] = theta[w,data.ref_bus] == 0

        prob.addConstraint(angle_max_ctr, angle_min_ctr, ref_bus_ctr)

        # N.B. (Eqn 17, 18, 19) are domain constraints on g, r, and d (all >= 0) which
        # come for free from the Xpress defaults.


    # OBJECTIVE FUNCTION
    # -------------------
    objective = phi[1,1]
    prob.setObjective(objective, sense=xp.maximize)

    return prob, var_map


def build_wind_model(data, new_lines = [], bilevel = True):
    # Create the Xpress problem:
    prob = xp.problem("MILP Formulation")
    var_map = {}

    # UPPER-LEVEL VARIABLES
    # ---------------------
    # "x_k" : binary variable, 1 if line k is built, 0 otherwise.
    x = {(k): xp.var(vartype = xp.binary) for k in data.lines}

    # "x_m^P" : binary variable, 1 if wind power project m is connected, 0 else.
    x_P = {(m): xp.var(vartype = xp.binary) for m in data.WPPs}

    # (Eqn 3) : All existing lines are "open"
    for k in data.existing_lines:
        x[k].lb = 1

    # If the new_lines to be opened have been specified, force these decisions:
    if new_lines:
        for k in new_lines:
            x[k].lb = 1

        for k in data.lines:
            if k not in data.existing_lines:
                if k not in new_lines:
                    x[k].ub=0

    prob.addVariable(x, x_P)
    var_map['x'] = x
    var_map['x_P'] = x_P

    # REMOVED ALNOWIBET CONSTRAINTS


    # PRIMAL LOWER-LEVEL VARIABLES
    # ----------------------------
    # "d_{jh}(w)" : power consumed by hth block of the jth demand in scenario w.
    # N.B. This is indexed as d[w, j, h].
    d = {k: xp.var() for k in data.d_max_jh.keys()}
    var_map['d'] = d

    # "g_{ib}(w)" : power produced by the bth block of the ith gen. unit in scen. w.
    # N.B. This is indexed as g[w, i, b].
    g = {k: xp.var() for k in data.g_max.keys()}
    var_map['g'] = g

    # "f_{k}(w)" : power flow through line k in scenario w.
    # N.B. This is indexed as f[w, k]
    f = {(w,k): xp.var(vartype = xp.continuous, lb = -xp.infinity) 
        for w in data.scenarios for k in data.lines}
    var_map['f'] = f
    # N.B. The index of line_data is just a single numnber for each line.

    # "r_{j}(w)" : load shed by the jth customer in scenario w.
    # N.B. This is indexed as r[w,j].
    r = {(w,j): xp.var(vartype = xp.continuous)
        for w in data.scenarios for j in data.customers}
    var_map['r'] = r

    # "theta_{s}(w)" : voltage angle at bus s in scenario w
    # N.B. This is indexed as theta[w,s].
    theta = {(w, s): xp.var(vartype = xp.continuous, lb = -xp.infinity)
            for w in data.scenarios for s in data.buses}
    var_map['theta'] = theta

    # "p_{m}(w)" : power accepted from wind power plant m in scenario w
    # N.B. This is indexed as p[w,m].
    p = {(w,m): xp.var(vartype = xp.continuous)
         for w in data.scenarios for m in data.WPPs}
    var_map['p'] = p

    # Add the variables to the problem
    prob.addVariable(d, g, f, r, theta, p)

    if bilevel:
        # DUAL LOWER-LEVEL VARIABLES
        # --------------------------
        # "lambda_{s}(w)" : dual of the power flow constraint (free)
        # N.B. This is indexed as lambda_s[w,s].
        lambda_s = {(w,s): xp.var(lb = -xp.infinity) 
                    for w in data.scenarios for s in data.buses}
        var_map['lambda_s'] = lambda_s

        # "phi_{k}(w)" : dual of the line flow/voltage angle constraint (free)
        # N.B. This is indexed as phi[w, k]
        phi = {(w,k): xp.var(lb = -xp.infinity) 
            for w in data.scenarios for k in data.lines}
        var_map['phi'] = phi

        # "phi_{k}^{max}(w)" : dual of the forwards line flow limit constraint (>= 0)
        # N.B. This is indexed as phi_max[w,k]
        phi_max = {(w,k): xp.var() for w in data.scenarios for k in data.lines}
        var_map['phi_max'] = phi_max

        # "phi_{k}^{min}(w)" : dual of the backwards line flow limit constraint (<= 0)
        # N.B. This is indexed as phi_min[w,k]
        phi_min = {(w,k): xp.var(lb = -xp.infinity, ub=0) 
                for w in data.scenarios for k in data.lines}
        var_map['phi_min'] = phi_min

        # "varphi_{ib}^{max}(w)" : dual of the [generation from block] <= [Block_Size]
        # constraint (>= 0)
        # N.B. This is indexed as varphi_max[w, i, b]
        varphi_max = {k: xp.var() for k in data.g_max.keys()}
        var_map['varphi_max'] = varphi_max
        # Recall that g_max contains the block sizes of offers and is indexed by 
        # (scenario, generating_unit, block)

        # "beta_{jh}^{max}(w)" : dual of the [consumption from block] <= [Block_Size]
        # constraint (>= 0)
        # N.B. This is indexed as beta_max[w,j,h]
        beta_max = {k: xp.var() for k in data.d_max_jh.keys()}
        var_map['beta_max'] = beta_max
        # Recall that d_max_jh contains the block sizes of bids and is indexed by 
        # (scenario, customer, block)

        # "alpha_{j}^{max}(w)" : dual of the maximum load shed constraint (>= 0)
        # N.B. This is indexed as alpha_max[w,j]
        alpha_max = {(w,j): xp.var() for w in data.scenarios for j in data.customers}
        var_map['alpha_max'] = alpha_max

        # "rho_{j}(w)" : dual of the constraint on minimum consumption for each customer
        # (<= 0)
        # N.B. This is indexed as rho[w,j]
        rho = {(w,j): xp.var(lb = -xp.infinity, ub = 0) 
            for w in data.scenarios for j in data.customers}
        var_map['rho'] = rho

        # "xi_{s}^{max}(w)" : dual of the [voltage angle] <= pi constraint (>= 0)
        # N.B. This is indexed as xi[w,s]
        xi_max = {(w,s): xp.var() for w in data.scenarios for s in data.buses}
        var_map['xi_max'] = xi_max

        # "xi_{s}^{min}(w)" : dual of the [voltage angle] >= -pi constraint (<= 0)
        # N.B. This is indexed as xi[w,s]
        xi_min = {(w,s): xp.var(lb = -xp.infinity, ub = 0) 
                for w in data.scenarios for s in data.buses}
        var_map['xi_min'] = xi_min

        # "chi_s(w)" : dual of the reference bus voltage angle == 0 constraint (free)
        chi = {w: xp.var(lb = -xp.infinity) for w in data.scenarios}
        var_map['chi'] = chi

        # "gamma_{m}(w)" : dual of the maximum wind production constraint (>= 0)
        # N.B. This is indexed as gamma[w,m].
        gamma = {(w,m): xp.var() for w in data.scenarios for m in data.WPPs}
        var_map['gamma'] = gamma

        # AUXILIARY VARIABLES
        # -------------------
        # "phi_{k}^{-}(w)" : used to linearize dual constraints (26) and (27) (free)
        phi_minus = {(w,k): xp.var(lb = -xp.infinity, ub = xp.infinity)
                    for w in data.scenarios for k in data.lines}
        var_map['phi_minus'] = phi_minus

        # "varsigma_{m}(w)" : used to linearize strong duality constraint in Model 2 (free)
        varsigma = {(w,m): xp.var(lb = -xp.infinity, ub = xp.infinity)
                    for w in data.scenarios for m in data.WPPs}
        var_map['varsigma'] = varsigma

        prob.addVariable(lambda_s, phi, phi_max, phi_min, varphi_max, beta_max, 
                        alpha_max, rho, xi_max, xi_min, chi, gamma, phi_minus, 
                        varsigma)

    # OBJECTIVE FUNCTION
    # ------------------
    # Calculate the social-welfare benefit and load-shed costs in each scenario:
    sw_ls = {w: xp.Sum(data.lambda_D[k[1],k[2]] * d[k] for k in data.d_max_jh.keys() if k[0] == w) -\
                        xp.Sum(data.lambda_tilde_G[k[1],k[2]] * g[k] for k in data.g_max.keys() if k[0] == w) -\
                        xp.Sum(data.c_U[j] * r[w,j] for j in data.customers) for w in data.scenarios}
    
    var_map['sw_ls'] = sw_ls

    wind_cost = {w: xp.Sum(data.kappa * data.p_max[w,m] * x_P[m] for m in data.WPPs) for w in data.scenarios}

    var_map['wind_cost'] = wind_cost

    # sw_ls = xp.Sum(data.delta[w]*(xp.Sum(data.lambda_D[k[1],k[2]] * d[k] for k in data.d_max_jh.keys() if k[0] == w) -\
    #                     xp.Sum(data.lambda_G[k[1],k[2]] * g[k] for k in data.g_max.keys() if k[0] == w)) for w in data.scenarios)

    # Transmission investment costs:
    tics = xp.Sum(data.c[k]*x[k] for k in data.prospective_lines)
    var_map['tics'] = tics

    # NO WIND POWER INVESTMENT COSTS AS THESE ARE COVERED BY THE STRIKE PRICE.

    # Objective function:
    # REMOVED POSSIBILITY TO HAVE AN ALTERNATIVE OBJECTIVE HERE.
    objective = data.sigma*xp.Sum(data.delta[w] * (sw_ls[w] - wind_cost[w]) for w in data.scenarios) - tics
    
    
    prob.setObjective(objective, sense=xp.maximize)


    # UPPER-LEVEL CONSTRAINTS
    # -----------------------
    # (Eqn 2) : Transmission investment budget constraint
    budget_ctr = xp.Sum(data.c[k]*x[k] for k in data.prospective_lines) <= data.c_max
    prob.addConstraint(budget_ctr)

    # # (Eqn 3) : All existing lines are "open" <- Needs to be done before x added to problem
    # for k in data.existing_lines:
    #     x[k].lb = 1

    # N.B. (Eqn 4), that the x[k] are all binary decisions, is established when
    # the variables are declared and does not need to be written here.

    # (Eqn 6) : Power flow at each bus
    PF_ctr = {} # dictionary to be indexed by scenario and bus

    for w in data.scenarios:
        for s in data.buses:

            # Calculate amount generated at bus:
            generated_at_bus = xp.Sum(
                                        xp.Sum(g[k] for k in data.g_max.keys() if k[0]==w and k[1]==i) 
                                    for i in data.generators_at_bus[s])
            
            # Calculate wind power accepted at bus:
            wind_at_bus = xp.Sum(
                xp.Sum(p[w,m] for m in data.WPPs_at_bus[s])
            )


            flows_out_of_bus = data.line_data[data.line_data['From']==s].index
            flow_out = xp.Sum(f[w, k] for k in flows_out_of_bus)

            flows_into_bus = data.line_data[data.line_data['To']==s].index
            flow_in = xp.Sum(f[w, k] for k in flows_into_bus)

            load_shed_at_bus = xp.Sum(r[w,j] for j in data.customers_at_bus[s])

            # Calculate the amount consumed at bus:
            # TODO: MAY HAVE A PROBLEM HERE FOR TYPE 2 BUSES AS SOME OF THESE WILL BE
            # EMPTY OR CAUSE ANOTHER RELATED ERROR
            demand_blocks_at_bus = data.bid_data[(data.bid_data['Bus']==s) & (data.bid_data['Scenario']==w)].index
            consumed_at_bus = xp.Sum(d[w, data.bid_data.loc[i,'Customer'], data.bid_data.loc[i,'Block']] 
                                    for i in demand_blocks_at_bus)

            # Assert power flow is balanced.
            PF_ctr[(w,s)] = generated_at_bus + wind_at_bus - flow_out + flow_in + load_shed_at_bus - consumed_at_bus == 0

    prob.addConstraint(PF_ctr)

    
    # N.B. (Eqn 7, 8, 9) for Kirchoff's Voltage Law and preventing line overloading
    # are non-linear and replaced in this model by (Eqn 40 and 41).

    # (Eqn 40): Lines are not overloaded
    fmax_linctr = {(w,k): f[w,k] <= x[k] * data.f_max[k] 
                for w in data.scenarios for k in data.lines}

    fmin_linctr = {(w,k): - x[k] * data.f_max[k] <= f[w,k] 
                for w in data.scenarios for k in data.lines}


    # (Eqn 41): "Linearized" KVL constraints
    KVL_linctr_1 = {(w,k): 
                    -(1-x[k]) * data.M <= (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) 
                    for w in data.scenarios for k in data.lines}

    KVL_linctr_2 = {(w,k):
                    (1/data.b_MW[k]) * f[w,k] - (theta[w,data.o[k]] - theta[w,data.r[k]]) <= (1-x[k]) * data.M
                    for w in data.scenarios for k in data.lines}

    prob.addConstraint(fmax_linctr, fmin_linctr, KVL_linctr_1, KVL_linctr_2)
    

    # (Eqn 10) : generation does not exceed block offered
    gen_ctr = {k : g[k] <= data.g_max[k] for k in data.g_max.keys()}
    # Recall that data.g_max is indexed by [Scenario, Generating_Unit, Block]

    prob.addConstraint(gen_ctr)


    # (Eqn 11): consumption does not exceed block bid
    con_ctr = {k: d[k] <= data.d_max_jh[k] for k in data.d_max_jh.keys()}
    # Recall that data.d_max_jh is indexed by [Scenario, Generating_Unit, Block]

    prob.addConstraint(con_ctr)


    # (Eqn 12): load shed does not exceed amount of demand from that customer
    shed_ctr = {}

    for w in data.scenarios:
        for j in data.customers:
            shed_ctr[w, j] = r[w, j] <= data.d_max[w,j]

    prob.addConstraint(shed_ctr)


    # (Eqn 13): minimum power always provided (90% of demand in Garces et al.)
    # TODO: This looks like it could be tidied up a bit...
    min_ctr = {}

    for w in data.scenarios:
        for j in data.customers:

            selector = (data.bid_data['Scenario']==w) & (data.bid_data['Customer']==j)
            demand_blocks_at_customer = data.bid_data[selector].index

            min_ctr[w,j] = xp.Sum(d[w,j,data.bid_data.loc[index,'Block']] 
                                for index in demand_blocks_at_customer) >= data.d_min[w,j]
            
    prob.addConstraint(min_ctr)


    # (Eqn 14, 15, 16): Restricting the voltage angle to [-pi, pi]:
    angle_max_ctr = {}
    angle_min_ctr = {}
    ref_bus_ctr = {}

    for w in data.scenarios:
        for s in data.buses:
            angle_max_ctr[w,s] = theta[w,s] <= np.pi
            angle_min_ctr[w,s] = theta[w,s] >= - np.pi
            ref_bus_ctr[w] = theta[w,data.ref_bus] == 0

    prob.addConstraint(angle_max_ctr, angle_min_ctr, ref_bus_ctr)

    # (Model 2 Eqn 4.2) : ensuring wind power not too high
    max_wind_ctr = {}

    for w in data.scenarios:
        for m in data.WPPs:
            max_wind_ctr[w,m] = p[w,m] <= data.p_max[w,m] * x_P[m]


    if bilevel:
        # LOWER-LEVEL DUAL CONSTRAINTS
        # ----------------------------
        # (Eqn 22) : dual constraint associated with primal variable g
        g_dualctr = {k: 
                    lambda_s[k[0], data.s_i[k[1]]] + varphi_max[k] >= -data.lambda_G[k[1],k[2]] 
                    for k in data.g_max.keys()
                    }
        # The indexing is a bit awkward here. k = (w, i, b) and we are taking:
        # lambda_s[w,s_i[i]] <- the dual variable for scenario w and the bus where i is
        # varphi_max[w,i,b] <- the dual variable for scenario, i, and block
        # lambda_G[i,b]

        # (Eqn 23) : dual constraint associated with primal variable d
        d_dualctr = {k:
                    - lambda_s[k[0], data.s_j[k[1]]] + beta_max[k] + rho[k[0],k[1]] >= data.lambda_D[k[1],k[2]]
                    for k in data.d_max_jh.keys()
                    }

        # (Eqn 24) : dual constraint associated with primal variable r
        r_dualctr = {(w,j):
                    lambda_s[w,data.s_j[j]] + alpha_max[w,j] >= -data.c_U[j]
                    for w in data.scenarios for j in data.generating_units             
                    }

        # (Eqn 25) : dual constraint associated with primal variable f
        f_dualctr = {(w,k):
                    - lambda_s[w,data.o[k]] + lambda_s[w,data.r[k]] + phi[w,k] + phi_max[w,k] + phi_min[w,k] == 0
                    for w in data.scenarios for k in data.lines
                    }

        # N.B. (Eqn 26 and 27), nonlinear dual constraints, are replaced in this model
        # by (Eqn 42, 43, 44, and 45). This requires the introduction of an auxiliary
        # variable denoted "phi_{k}^{-}(w)" in the paper.

        # (Eqn 42) : 
        theta_lindualctr_1 = {(w,s): - xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.o[k] == s) 
                            + xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.r[k] == s) 
                            + xi_max[w,s] + xi_min[w,s] == 0 
                            for w in data.scenarios for s in data.buses if s != data.ref_bus}

        # (Eqn 43) :
        theta_lindualctr_2 = {w:
                        - xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.o[k] == data.ref_bus) 
                        + xp.Sum(data.b_MW[k]*(phi[w,k] - phi_minus[w,k]) for k in data.lines if data.r[k] == data.ref_bus)
                        + chi[w] == 0
                        for w in data.scenarios
                        }

        # (Eqn 44) :
        theta_lindualctr_3 = {(w,k): -x[k] * data.Gamma_max <= phi[w,k] - phi_minus[w,k]
                            for w in data.scenarios for k in data.lines}

        theta_lindualctr_4 = {(w,k): phi[w,k] - phi_minus[w,k]<= x[k] * data.Gamma_max
                            for w in data.scenarios for k in data.lines}

        # (Eqn 45) :
        theta_lindualctr_5 = {(w,k): -(1 - x[k]) * data.Gamma_max <= phi_minus[w,k]
                            for w in data.scenarios for k in data.lines}

        theta_lindualctr_6 = {(w,k): phi_minus[w,k] <= (1 - x[k]) * data.Gamma_max
                            for w in data.scenarios for k in data.lines}

        # (Model 2 Eqn 4.8):
        p_lindualctr_1 = {(w,m): lambda_s[w, data.s_m[m]] + gamma[w,m] >= 0
                          for w in data.scenarios for m in data.WPPs}
        
        # (Model 2 Eqn 4.9) :
        p_lindualctr_2 = {(w,m): gamma[w,m] - varsigma[w,m] <= data.Gamma_wind * x_P[m]
                          for w in data.scenarios for m in data.WPPs}
        
        p_lindualctr_3 = {(w,m): -data.Gamma_wind * x_P[m] <= gamma[w,m] - varsigma[w,m] 
                          for w in data.scenarios for m in data.WPPs}
        
        # (Model 2 Eqn 4.10):
        p_lindualctr_4 = {(w,m): varsigma[w,m] <= (1 - x_P[m]) * data.Gamma_wind
                          for w in data.scenarios for m in data.WPPs}
        
        p_lindualctr_5 = {(w,m): -(1 - x_P[m]) * data.Gamma_wind <= varsigma[w,m]
                          for w in data.scenarios for m in data.WPPs}


        # Calculate the social-welfare benefit and load-shed costs in each scenario:
        sw_ls_LL = {w: xp.Sum(data.lambda_D[k[1],k[2]] * d[k] for k in data.d_max_jh.keys() if k[0] == w) -\
                            xp.Sum(data.lambda_G[k[1],k[2]] * g[k] for k in data.g_max.keys() if k[0] == w) -\
                            xp.Sum(data.c_U[j] * r[w,j] for j in data.customers) for w in data.scenarios}
        
        var_map['sw_ls_LL'] = sw_ls_LL

        # (Eqn 28 to 34) are domain constraints accounted for in variable declarations.
        # (Eqn 35) : strong duality theorem (SDT) constraint
        # The LHS of the SDT constraint is [social_welfare] - [load shed penalty] in
        # each scenario.
        SDT_ctr = {w :
                    sw_ls_LL[w] == xp.Sum((phi_max[w,k] - phi_min[w,k])*data.f_max[k] for k in data.lines) +\
                                xp.Sum(varphi_max[k] * data.g_max[k] for k in data.g_max.keys() if k[0] == w) +\
                                xp.Sum(beta_max[k] * data.d_max_jh[k] for k in data.d_max_jh.keys() if k[0] == w) +\
                                xp.Sum((alpha_max[w, j] * data.d_max[w,j] + rho[w,j] * data.d_min[w,j]) for j in data.customers) +\
                                xp.Sum(np.pi*(xi_max[w,s] - xi_min[w,s]) for s in data.buses) +\
                                xp.Sum(data.p_max[w,m] * (gamma[w,m] - varsigma[w,m]) for m in data.WPPs)
                    for w in data.scenarios
                    }

        prob.addConstraint(g_dualctr, d_dualctr, f_dualctr, r_dualctr, SDT_ctr, theta_lindualctr_1,
                        theta_lindualctr_2, theta_lindualctr_3, theta_lindualctr_4, 
                        theta_lindualctr_5, theta_lindualctr_6, p_lindualctr_1,
                        p_lindualctr_2, p_lindualctr_3, p_lindualctr_4, p_lindualctr_5)
        
    return prob, var_map