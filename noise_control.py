import gurobipy as gp
from gurobipy import GRB

import requests
import json
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io as sio


import sys
def check_response(response):
    """Check status code from restful API and return result if no error

    Parameters
    ----------

    response: obj, response object

    Returns
    -------
    result : dict, result from call to restful API

    """
    if isinstance(response, requests.Response):
        status = response.status_code
    if status == 200:
        response = response.json()['payload']
        return response
    print("Unexpected error: {}".format(response.text))
    print("Exiting!")
    sys.exit()
    
    

class NoiseCtr():
    
    def __init__(self, A, B, D, R, dt = 15*60, N = 12*4, ele_price_pat = "dynamic", obj_opt = "ambient_res", build_type = "normal", platform = "local", optimizer = 'gurobi', noise_model = "piecewise_affine"):
        
        
        ################################
        ### dt: 15mins, N: 12hours, ele_price_mat: "dynamic" or "highly dynamic", obj_opt: "ambient_res" or "inverse_penalty", build_type: "normal" or "residential" or "commercial"
        ### platform: "local" or "web"
        ################################
        self.A = A
        self.B = B
        self.D = D
        self.R = R
        self.N = N ## prediction steps
        self.horizon = (N)*dt  ## horizon for geting the prediction
        self.n = A.shape[0]
        self.ti = 0 ### real time building temperature
        self.x = np.zeros(A.shape[0]) ### system state
        self.f = np.array([0, 40, 60, 60])
        self.alpha = np.array([0, 0.2, 0.70, 1])
        self.P_max = 3033
        self.dt = dt ### length of sampling period in seconds
        self.mpc_u = 10 ## current MPC control input
        self.f_nos = 0 ## current noise of HP
        self.output_dict = {}
        self.ti_list = []
        self.input_list = []
        self.f_nos_list = []
        self.temp_list = []
        self.violation_list = []
        self.ele_price_pat = ele_price_pat    #("PriceElectricPowerHighlyDynamic")
        self.platform = platform             ### electricity price pattern: day_night or highly_dynamic
        self.obj_opt = obj_opt
        self.build_type = build_type
        self.t = 0 ### update t as the time instant of the simulation time instant
        self.occupy_time = np.ones(N)
        self.mixed_noise_list = []
        self.verify = True
        self.open_loop = False
        self.delta = 0 ### the parameter for adjusting the threshold of hp noise decibel

        self.noise_model = noise_model
        
        self.amb_con = np.array([])
        self.ele_price = np.array([])
        
        self.energy_cost = 0
        self.noise_cost = 0

        arx_data = np.load("arx_data.npz")
        self.a_y = arx_data["a_y"]
        self.a_to = arx_data["a_to"]
        self.a_sol = arx_data["a_sol"]
        self.a_u = arx_data["a_in"]
               
        self.n_y = len(self.a_y)
        self.n_to = len(self.a_to)
        self.n_sol = len(self.a_sol)
        self.n_u = len(self.a_u)-1

        self.n_max = np.max([self.n_y, self.n_to, self.n_sol, self.n_u])

        self.to_buff_init = np.array([])
        self.sol_buff_init = np.array([])

        self.y_buff = np.array([])
        self.u_buff = np.array([])
        self.to_buff = np.array([])
        self.sol_buff = np.array([])
        
        self.optimizer = optimizer
        self.url_local = "http://127.0.0.1:5000/"
        
        self.boptest_url = "https://api.boptest.net"
        testcase_name = "bestest_hydronic_heat_pump"
        self.test_id = requests.post(f"{self.boptest_url}/testcases/{testcase_name}/select", verify = self.verify).json()["testid"]
        
    
    def init_state(self):
        
        if self.platform == "web":
            check_response(requests.put(f"{self.boptest_url}/scenario/{self.test_id}", verify = self.verify, json = {"electricity_price":self.ele_price_pat}))
            outputs = check_response(requests.put(f"{self.boptest_url}/initialize/{self.test_id}", verify = self.verify, json = {"start_time":1380000,
                                                                "warmup_period":2400}))
                                                                #"time_period":"peak_heat_day"}).json()['payload']["time_period"] ##13860000
        if self.platform == "local":
            check_response(requests.put(f"{self.url_local}/scenario", verify = self.verify, json = {"electricity_price":self.ele_price_pat}))
            outputs = check_response(requests.put(f"{self.url_local}/initialize", verify = self.verify, json = {"start_time":1380000,
                                                                "warmup_period":2400}))
        
        
        self.t = outputs["weaSta_reaWeaCloTim_y"]
        ti = outputs["reaTZon_y"] - 273.15
        to = outputs["weaSta_reaWeaTDryBul_y"] - 273.15
        
        self.ti = ti
        
        self.update_state(ti, to)
        
        self.energy_cost = 0
        self.noise_cost = 0
    
    def update_state(self, ti, to):

        self.ti = ti
        self.x[0] = ti
        self.x[[1,2]] = self.R@np.array([ti, to])

    def initialize_buff(self):

        #####################################
        #####  generate initial buffer ######
        #####################################
        
        _y_buff = []
        _u_buff = []
        _sol_buff = []
        _to_buff = []
        output_dict = {}

        check_response(requests.put(f"{self.url_local}/step", verify = self.verify, json = {"step": self.dt}))
        
        for t in range(self.n_max):
            if self.ti <= 19:
                output_dict = check_response(requests.post(f"{self.url_local}/advance", verify = self.verify, json = {"oveHeaPumY_u": 1, "oveHeaPumY_activate": 1}))
            elif self.ti >= 22:
                output_dict = check_response(requests.post(f"{self.url_local}/advance", verify = self.verify, json = {"oveHeaPumY_u": 0, "oveHeaPumY_activate": 1}))
            else:
                output_dict = check_response(requests.post(f"{self.url_local}/advance", verify = self.verify, json = {"oveHeaPumY_u": 1, "oveHeaPumY_activate": 1}))
 
            _y_buff.append(output_dict["reaTZon_y"] - 273.15)
            _u_buff.append(output_dict["reaPHeaPum_y"])
            _sol_buff.append(output_dict["weaSta_reaWeaHGloHor_y"])
            _to_buff.append(output_dict["weaSta_reaWeaTDryBul_y"] - 273.15)

        
        self.y_buff = np.array(_y_buff[-self.n_y:])
        self.u_buff = np.array(_u_buff[-self.n_u:])
        self.sol_buff_init = np.array(_sol_buff[-self.n_sol:])
        self.to_buff_init = np.array(_to_buff[-self.n_to:])
        
    def get_forecast(self): ### get the prediction of ambient condition and electircity price
        
        inputs = {"point_names":["TDryBul","HGloHor","PriceElectricPowerDynamic","PriceElectricPowerHighlyDynamic"],
                  "horizon": self.horizon,"interval":self.dt}
        
        if self.platform == "web":
            forecast_dict = check_response(requests.put(f"{self.boptest_url}/forecast/{self.test_id}", verify = self.verify, json = inputs ))
            
        if self.platform == "local":
            forecast_dict = check_response(requests.put(f"{self.url_local}/forecast", verify = self.verify, json = inputs ))
        
        to_forecast = np.array(forecast_dict["TDryBul"][1:]) - 273.15
        sol_forecast = np.array(forecast_dict["HGloHor"][1:]) 

        assert len(to_forecast) == self.N
        
        self.amb_con = np.append(to_forecast.reshape(-1,1), sol_forecast.reshape(-1,1), axis = 1)
        
        
        if self.ele_price_pat == 'dynamic':
            self.ele_price = np.array(forecast_dict["PriceElectricPowerDynamic"])
        if self.ele_price_pat == "highly_dynamic":
            self.ele_price = np.array(forecast_dict["PriceElectricPowerHighlyDynamic"])

        #### update data buff for ambient temperature and solar radiation #####
        self.to_buff = np.append(self.to_buff_init, to_forecast)
        self.sol_buff = np.append(self.sol_buff_init, sol_forecast)
            
            
    def real_noise(self, n_hp, n_amb):
        # compute the real noise decibel based on the heat pump decibel and ambient decibel
        n_total = 10*np.log10(10**(n_hp/10) + 10**(n_amb/10))
        
        self.mixed_noise_list.append(n_total) 

        
    def step(self): 
        ### this function is to send out the modulated MPC control input, and get the system response
        
        ### compute the modulated control input sigal
        
        mod_step = int(self.mpc_u/self.P_max * self.dt)
        inputs1 = {"oveHeaPumY_u": 1, "oveHeaPumY_activate": 1}
        inputs2 = {"oveHeaPumY_u": 0, "oveHeaPumY_activate": 1}
        
        ### send the modulated control input
        
        if self.platform == "web":
            if mod_step >= 0.01:
                check_response(requests.put(f"{self.boptest_url}/step/{self.test_id}", verify = self.verify, json = {"step": mod_step}))
                self.output_dict = check_response(requests.post(f"{self.boptest_url}/advance/{self.test_id}", verify = self.verify, json = inputs1))

                self.P_max = self.output_dict["reaPHeaPum_y"]

            if (self.dt - mod_step) >= 0.01:    
                check_response(requests.put(f"{self.boptest_url}/step/{self.test_id}", verify = self.verify, json = {"step": self.dt - mod_step}))
                self.output_dict = check_response(requests.post(f"{self.boptest_url}/advance/{self.test_id}", verify = self.verify, data = inputs2))
            
            
        if self.platform == "local":
            
            if mod_step >= 0.01:
                check_response(requests.put(f"{self.url_local}/step", verify = self.verify, json = {"step": mod_step}))
                self.output_dict = check_response(requests.post(f"{self.url_local}/advance", verify = self.verify, json = inputs1))

                self.P_max = self.output_dict["reaPHeaPum_y"]
                
            if (self.dt - mod_step) >= 0.01:
                check_response(requests.put(f"{self.url_local}/step", verify = self.verify, json = {"step": self.dt - mod_step}))
                self.output_dict = check_response(requests.post(f"{self.url_local}/advance", verify = self.verify, json = inputs2))

        real_P = self.P_max*mod_step/self.dt

        self.t = self.output_dict["weaSta_reaWeaCloTim_y"]
                
        
        ### update current initial condition
        ti = self.output_dict["reaTZon_y"] - 273.15
        to = self.output_dict["weaSta_reaWeaTDryBul_y"] - 273.15
        sol = self.output_dict["weaSta_reaWeaHGloHor_y"]

        self.ti = ti
        self.ti_list.append(ti)
        self.f_nos_list.append(self.f_nos)
        self.update_state(ti, to)

        ###### update the buffer #####
        self.y_buff = np.append(self.y_buff[1:], ti)
        self.u_buff = np.append(self.u_buff[1:], real_P)
        self.to_buff_init = np.append(self.to_buff_init[1:], to)
        self.sol_buff_init = np.append(self.sol_buff_init[1:], sol)
        
        
    def mpc_ctr(self, amb_nos, eta):
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:

                x = m.addMVar(shape = (self.N,), vtype = GRB.CONTINUOUS)
                u = m.addMVar(shape = self.N, vtype = GRB.CONTINUOUS, ub = self.P_max)

                delta = m.addMVar(shape = (self.N,), vtype = GRB.CONTINUOUS, ub = 2)

                for t in range(self.N):
                    if (t <= self.n_y) and (t <= self.n_u):
                        
                        m.addConstr( x[t] == sum(self.a_y[k]*x[t-1-k] for k in range(t)) + sum(self.a_y[t+k]*self.y_buff[self.n_y-1-k] for k in range(self.n_y - t)) +
                                    sum(self.a_to[k]*self.to_buff[self.n_to-1+t-k] for k in range(self.n_to)) + sum(self.a_sol[k]*self.sol_buff[self.n_sol-1+t-k] for k in range(self.n_sol)) +
                                   sum(self.a_u[k]*u[t-k] for k in range(t+1)) + sum(self.a_u[t+1+k]*self.u_buff[self.n_u-1-k] for k in range(self.n_u-t)) )
                    
                    elif (self.n_y > self.n_u) and (t <= self.n_y) and (t > self.n_u):
                        m.addConstr( x[t] == sum(self.a_y[k]*x[t-1-k] for k in range(t)) + sum(self.a_y[t+k]*self.y_buff[self.n_y-1-k] for k in range(self.n_y - t)) +
                                    sum(self.a_to[k]*self.to_buff[self.n_to-1+t-k] for k in range(self.n_to)) + sum(self.a_sol[k]*self.sol_buff[self.n_sol-1+t-k] for k in range(self.n_sol)) +
                                   sum(self.a_u[k]*u[t-k] for k in range(self.n_u+1)) )
                    
                    elif (self.n_y < self.n_u) and (t <= self.n_u) and (t > self.n_y):
                        m.addConstr( x[t] == sum(self.a_y[k]*x[t-1-k] for k in range(self.n_y)) +
                                    sum(self.a_to[k]*self.to_buff[self.n_to-1+t-k] for k in range(self.n_to)) + sum(self.a_sol[k]*self.sol_buff[self.n_sol-1+t-k] for k in range(self.n_sol)) +
                                   sum(self.a_u[k]*u[t-k] for k in range(t+1)) + sum(self.a_u[t+1+k]*self.u_buff[self.n_u-1-k] for k in range(self.n_u - t)) )
                    else:
                        m.addConstr( x[t] == sum(self.a_y[k]*x[t-1-k] for k in range(self.n_y)) +
                                    sum(self.a_to[k]*self.to_buff[self.n_to-1+t-k] for k in range(self.n_to)) + sum(self.a_sol[k]*self.sol_buff[self.n_sol-1+t-k] for k in range(self.n_sol)) +
                                   sum(self.a_u[k]*u[t-k] for k in range(self.n_u+1)) )
                        

                ### set indoor temperature constraints ###
                
                    
                if self.build_type == "residential":
                    for k in range(1, self.N+1):
                        if ( ((k*self.dt+self.t)%(24*3600)) >= 9*3600) and ( ((k*self.dt+self.t)%(24*3600)) <= 17*3600):
                            self.occupy_time[k-1] = 0
                        else:
                            self.occupy_time[k-1] = 1
                
                    
                if self.build_type == "commercial":
                    for k in range(1, self.N+1):
                        if ( ((k*self.dt + self.t)%(24*3600)) >= 8*3600) and ( ((k*self.dt + self.t)%(24*3600)) <= 18*3600):
                            self.occupy_time[k-1] = 1
                        else:
                            self.occupy_time[k-1] = 0
                            
                if self.build_type == "normal":
                    self.occupy_time = np.ones(self.N)
                    
                lower_bound = 16*np.ones(self.N)
                upper_bound = 27*np.ones(self.N)
                
                c3 = m.addConstrs( (lower_bound[t] + 3*self.occupy_time[t] - delta[t] <= x[t] for t in range(self.N) ))
                c4 = m.addConstrs( (x[t] <= upper_bound[t] - 3*self.occupy_time[t] + delta[t] for t in range(self.N) ))

                p = m.addMVar(shape = (self.N,), vtype = GRB.CONTINUOUS, ub = 1)
                f_nos = m.addMVar(shape = (self.N,), vtype = GRB.CONTINUOUS, ub = 100)
                m.addConstrs( (p[t]*self.P_max == u[t] for t in range(self.N)) )  ### can be modified to give better numerical performance: remove self.P_max, change self.P_max to multiply in dynamics
                
                if self.noise_model == "piecewise_affine":
                    lamda = m.addMVar(shape = (self.N, 4), vtype = GRB.CONTINUOUS, ub = 1)
                    z = m.addMVar(shape = (self.N, 3), vtype = GRB.BINARY)    
                    m.addConstrs( (gp.quicksum(lamda[t,:]) == 1 for t in range(self.N)) )
                    m.addConstrs( (sum(z[t,:]) == 1 for t in range(self.N)) )
    
                    m.addConstrs((lamda[t,0] <= z[t,0] for t in range(self.N)))
                    m.addConstrs((lamda[t,1] <= z[t,0] + z[t,1] for t in range(self.N)))
                    m.addConstrs((lamda[t,2] <= z[t,1] + z[t,2] for t in range(self.N)))
                    m.addConstrs((lamda[t,3] <= z[t,2] for t in range(self.N)))
    
                    m.addConstrs( (sum(lamda[t,k]*self.alpha[k] for k in range(4)) == p[t] for t in range(self.N)) )
                    
                    m.addConstrs( (f_nos[t] == sum(lamda[t,k]*self.f[k] for k in range(4)) for t in range(self.N)) )

                elif self.noise_model == "linear":
                    m.addConstrs( (f_nos[t] == (self.f[-1] - self.f[1])*p[t] for t in range(self.N)) )



                M = 1e5
                
                if self.obj_opt == "inverse_penalty":    
                    m.setObjective( sum(self.ele_price[t]*u[t]/1e3 for t in range(self.N))+ eta*sum(1/amb_nos[t]*f_nos[t] for t in range(self.N)) + M*sum(delta), GRB.MINIMIZE )
                    
                elif self.obj_opt == "ambient_res":
                    
                    
                    
                    nos_res = m.addMVar(shape = (self.N), vtype = GRB.CONTINUOUS)

                    ### Linear constraint formulation ########
                    m.addConstrs(( f_nos[t] <= amb_nos[t] + nos_res[t] for t in range(self.N)))

                    #### MILP reformulation of the residule function    
                    # bigM = 1e3
                    # y = m.addMVar(shape = (self.N), vtype = GRB.BINARY)
                    
                    # m.addConstrs((f_nos[t] - (amb_nos[t] - self.delta) <= bigM*y[t] for t in range(self.N)))
                    # m.addConstrs( ((amb_nos[t] - self.delta) - f_nos[t] <= bigM*(1 - y[t]) for t in range(self.N)) )
                    # m.addConstrs( (nos_res[t] >= f_nos[t] - (amb_nos[t] - self.delta) for t in range(self.N)) )
                    # m.addConstrs( (nos_res[t] <= f_nos[t] - (amb_nos[t] - self.delta) + bigM*(1-y[t]) for t in range(self.N) ) )
                    # m.addConstrs( (nos_res[t] <= bigM*y[t] for t in range(self.N)) )
                    
                    
                    m.setObjective( sum(self.ele_price[t]*u[t]/1e3 for t in range(self.N))+ eta*sum(nos_res)+ M*sum(delta), GRB.MINIMIZE )
                    # 0.1*sum((x[t+1,0] - 20)*(x[t+1,0]-20) for t in range(self.N)
                    
                
                m.setParam('OutputFlag', 0)
                #m.setParam("MIPGap", 1e-3)
                
                m.optimize()
                m.display()
                           
                self.mpc_u = u.X[0]
                self.f_nos = f_nos.X[0]

#                plt.plot(x.X)
                # print("noise is: ", f_nos.X)
                # print("control input is: ", u.X)
                
                ### update the real noise of the mixed noise
                self.real_noise(self.f_nos, amb_nos[0])


                ############## energy cost list #############
                if self.open_loop == False:
                    self.energy_cost += self.ele_price[0]*u.X[0]/1e3/4
                elif self.open_loop == True:
                    self.energy_cost += sum(self.ele_price[t]*u.X[t]/1e3/4 for t in range(self.N))

                ################ noise cost list #############
                if self.obj_opt == "inverse_penalty":
                    
                    if self.open_loop == False:
                        self.noise_cost += 1/amb_nos[0]*f_nos.X[0]
                    else:
                        self.noise_cost += sum(1/amb_nos[t] * f_nos.X[t] for t in range(self.N))

                if self.obj_opt == "ambient_res":
                    
                    if self.open_loop == False:
                        self.noise_cost += np.maximum(f_nos.X[0] - (amb_nos[0]-self.delta), 0)
                    elif self.open_loop == True:
                        self.noise_cost += sum(np.maximum(f_nos.X[t] - (amb_nos[t]-self.delta), 0) for t in range(self.N))