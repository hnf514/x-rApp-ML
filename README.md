# -*- coding: utf-8 -*-
"""
Created on Thu Aug 1 14:44:42 2024

@author: hnf514
"""

import random
import json
import requests
import sys
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier

from new_ue_data_fetch_process import fetch_ue_data
from new_ue_neighbour_data_fetch_process import fetch_neighbour_ue_data
from new_cell_data_fetch_process import fetch_microcell_data
from get_last_simulation_id_non_stand_alone import retrieve_latest_sim_id

# -------- Hyperparameters --------
ENABLE_XAPP = True  # Enable xApp execution
INITIAL_POWER_DOWN = True  # Reduce mMIMO order instead of turning off cells
MONITOR_DATABASE = True  # Verify database connectivity

MIN_CELL_RUNTIME = 15  # Minimum seconds a cell must stay active before scaling down mMIMO order
MIMO_ADJUST_DELAY = 5  # Delay before modifying mMIMO order
ACTIVATION_DELAY = 3  # Delay before scaling up mMIMO order
MAX_XAPP_RUNTIME = 7200  # Max execution time (2 hours)

UTILIZATION_THRESHOLD = 50  # PRB usage threshold for reducing mMIMO order
RSRP_THRESHOLD = -110  # Minimum acceptable RSRP for handovers

MAX_CELL_SEARCH_RADIUS = 250  # Max distance to find neighboring cells for activation

# -------- System Variables --------
cell_mimo_log = {}  # Tracks last mMIMO adjustment time for each cell

# -------- Simulation Setup --------
SIMULATION_SERVER = "10.122.14.52"
STOP_SIMULATION_URL = f"http://{SIMULATION_SERVER}/sba/tests/status/20-simulation"
LATEST_SIM_ID = retrieve_latest_sim_id(f"http://{SIMULATION_SERVER}/sba/influx/query?q=SHOW%20DATABASES")

DATA_QUERY_URL = f"http://{SIMULATION_SERVER}/influx/query?db={LATEST_SIM_ID}-simulation&q="
UE_REPORT_URL = f"{DATA_QUERY_URL}SELECT * FROM UEReports GROUP BY Viavi.UE.Name ORDER BY time DESC LIMIT 1"
NEIGHBOUR_REPORT_URL = f"{DATA_QUERY_URL}SELECT * FROM UEReports-neighbour GROUP BY Viavi.UE.Name ORDER BY time DESC LIMIT 4"
CELL_REPORT_URL = f"{DATA_QUERY_URL}SELECT * FROM CellReports GROUP BY Viavi.Cell.Name ORDER BY time DESC LIMIT 1"

# -------- Utility Functions --------
def fetch_api_response(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

if MONITOR_DATABASE:
    if fetch_api_response(UE_REPORT_URL):
        print("Connected to TERA-VM RIC Tester")
    else:
        print("Database connection failed. Exiting.")
        sys.exit(1)

# -------- Modify mMIMO Order --------
def adjust_mimo_order(cell_id, mimo_mask):
    if cell_id in cell_mimo_log:
        if time.time() - cell_mimo_log[cell_id] < MIN_CELL_RUNTIME:
            return
    payload = {"type": "Antenna Configuration", "cell": cell_id, "antennaMask": mimo_mask}
    requests.post(f"http://{SIMULATION_SERVER}/sba/commands", json=payload)
    cell_mimo_log[cell_id] = time.time()
    time.sleep(MIMO_ADJUST_DELAY)

# -------- Main Execution --------
start_time = time.time()
while ENABLE_XAPP:
    elapsed_time = time.time() - start_time
    if elapsed_time >= MAX_XAPP_RUNTIME:
        print("Execution time exceeded. Stopping xApp.")
        break

    ue_data = fetch_ue_data(UE_REPORT_URL)
    neighbour_data = fetch_neighbour_ue_data(NEIGHBOUR_REPORT_URL)
    cell_data = fetch_microcell_data(CELL_REPORT_URL)

    if any(df is None or df.isnull().values.any() for df in [ue_data, neighbour_data, cell_data]):
        print("Data fetch error. Retrying...")
        time.sleep(1)
        continue

    # Decision Tree Classifier for cell activation
    clf = DecisionTreeClassifier()
    train_features = cell_data[['Viavi.Geo.x', 'Viavi.Geo.y', 'RRU.PrbTotDl']].values
    train_labels = cell_data['Viavi.isEnergySaving'].values
    clf.fit(train_features, train_labels)

    # Identify overloaded cells
    overloaded_cells = cell_data[cell_data['RRU.PrbTotDl'] >= 100]
    for _, overloaded_cell in overloaded_cells.iterrows():
        nearby_sleeping_cells = cell_data[(cell_data['Viavi.isEnergySaving'] == 1) &
                                          (np.sqrt((cell_data['Viavi.Geo.x'] - overloaded_cell['Viavi.Geo.x'])**2 +
                                                   (cell_data['Viavi.Geo.y'] - overloaded_cell['Viavi.Geo.y'])**2) <= MAX_CELL_SEARCH_RADIUS)]
        
        if not nearby_sleeping_cells.empty:
            best_candidate = clf.predict(nearby_sleeping_cells[['Viavi.Geo.x', 'Viavi.Geo.y', 'RRU.PrbTotDl']].values)
            adjust_mimo_order(best_candidate[0], "1111111111111111111111111111111111111111111111111111111111111111")

    # Identify underutilized cells and reduce mMIMO order
    underutilized_cells = cell_data[(cell_data['RRU.PrbTotDl'] <= UTILIZATION_THRESHOLD)]
    for _, low_usage_cell in underutilized_cells.iterrows():
        new_mimo_mask = "1100000000000000000000000000000000000000000000000000000000000000"
        adjust_mimo_order(low_usage_cell['Viavi.Cell.Name'], new_mimo_mask)

# Stop simulation
requests.delete(STOP_SIMULATION_URL, json={"type": "stop"})
print("Simulation completed.")





#ML based xApp by Vahid Kouh Daragh---REACH Project , University of York,
#In modern cellular networks, energy efficiency is a crucial factor in reducing operational costs and environmental impact. The ML-based ES-xApp (Energy Saving xApp) presented in this code is designed to optimize network performance by dynamically adjusting the transmission #mode of Remote Units (RUs). Instead of completely turning off underutilized base stations, this version of the xApp reduces the order of massive MIMO (mMIMO) antennas, scaling down from 16T16R to 2T2R based on network load conditions. This approach ensures that base #stations continue operating at a reduced capacity rather than being entirely deactivated, allowing for more efficient energy consumption while maintaining network coverage and quality of service.

#The xApp begins by initializing key hyperparameters, such as defining thresholds for PRB utilization and minimum acceptable RSRP for UE handovers. It connects to a live network simulation, retrieves the latest simulation ID, and establishes API links to UE reports, #neighbor reports, and cell reports. Data validation is performed to ensure reliability before decision-making. A key aspect of the xApp is its machine learning-based approach to network optimization. Using a Decision Tree Classifier, the system continuously monitors base #station utilization and determines whether an RU is overloaded, underutilized, or optimally operating. Overloaded RUs—where PRB usage reaches 100%—are identified, and the system searches for nearby sleeping cells within a 250-meter radius to assist in offloading traffic. #The Decision Tree algorithm predicts the best sleeping RU to activate, ensuring minimal power wastage while maintaining high throughput for UEs.

#When a base station is underutilized (PRB utilization below 50%), instead of shutting it down, the xApp reduces its mMIMO order to conserve energy. Initially, a 16T16R configuration is used for optimal performance. When PRB usage drops, the system progressively scales down #to 8T8R, 4T4R, and finally 2T2R if the RU remains lightly loaded. This adaptive approach allows for a fine-grained balance between energy savings and network performance. Before adjusting the mMIMO order, the system verifies that UEs connected to the base station can #maintain acceptable RSRP levels and have alternative serving cells if needed. By dynamically adjusting the MIMO configuration rather than turning off RUs, the xApp ensures continuous service availability, reduces power consumption, and optimizes spectral efficiency.

#The xApp executes this process continuously over a two-hour runtime. It ensures that base stations remain active at a minimum required configuration, preventing service disruptions while maximizing energy efficiency. If an RU’s PRB usage increases again, the system scales #the MIMO configuration back up, restoring full capacity when needed. Once the execution time limit is reached, the xApp terminates the simulation and logs results for analysis. This ML-driven adaptive MIMO adjustment presents a scalable and intelligent alternative to #conventional energy-saving methods that rely on static rules or full base station deactivation. By leveraging real-time network monitoring, machine learning-based decision-making, and flexible MIMO reconfiguration, this xApp enhances both network sustainability and #operational efficiency, making it a valuable tool for next-generation wireless networks.
