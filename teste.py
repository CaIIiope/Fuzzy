import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Input variables
risk = ctrl.Antecedent(np.arange(0, 11, 1), 'risk')
time_criticality = ctrl.Antecedent(np.arange(0, 11, 1), 'time_criticality')
system_health = ctrl.Antecedent(np.arange(0, 101, 1), 'system_health')

# Output variable
action = ctrl.Consequent(np.arange(0, 11, 1), 'action')

# Fuzzy sets for input variables
risk['low'] = fuzz.trimf(risk.universe, [0, 0, 5])
risk['medium'] = fuzz.trimf(risk.universe, [0, 5, 10])
risk['high'] = fuzz.trimf(risk.universe, [5, 10, 10])

time_criticality['low'] = fuzz.trimf(time_criticality.universe, [0, 0, 5])
time_criticality['medium'] = fuzz.trimf(time_criticality.universe, [0, 5, 10])
time_criticality['high'] = fuzz.trimf(time_criticality.universe, [5, 10, 10])

system_health['poor'] = fuzz.trimf(system_health.universe, [0, 0, 50])
system_health['fair'] = fuzz.trimf(system_health.universe, [0, 50, 100])
system_health['good'] = fuzz.trimf(system_health.universe, [50, 100, 100])

# Fuzzy sets for output variable
action['change_route'] = fuzz.trimf(action.universe, [0, 0, 3])
action['activate_redundancy'] = fuzz.trimf(action.universe, [1, 3, 5])
action['use_occupancy_maps'] = fuzz.trimf(action.universe, [3, 5, 7])
action['change_altitude'] = fuzz.trimf(action.universe, [5, 7, 9])
action['emergency_landing'] = fuzz.trimf(action.universe, [7, 10, 10])

# Define rules for each state
def create_rules_for_state(state):
    if state == "S1":  # Normal operation
        return [
            ctrl.Rule(risk['low'] & time_criticality['low'] & system_health['good'], action['change_route']),
            ctrl.Rule(risk['medium'] & system_health['fair'], action['activate_redundancy']),
            ctrl.Rule(risk['high'] & system_health['good'], action['change_altitude'])
        ]
    elif state == "S2":  # GPS Spoofing
        return [
            ctrl.Rule(risk['medium'] & time_criticality['medium'], action['use_occupancy_maps']),
            ctrl.Rule(risk['high'] & system_health['fair'], action['activate_redundancy']),
            ctrl.Rule(risk['high'] & time_criticality['high'], action['emergency_landing'])
        ]
    elif state == "S3":  # GPS Jamming
        return [
            ctrl.Rule(risk['medium'] & system_health['good'], action['change_altitude']),
            ctrl.Rule(risk['high'] & system_health['fair'], action['activate_redundancy']),
            ctrl.Rule(risk['high'] & time_criticality['high'], action['emergency_landing'])
        ]
    elif state == "S4":  # Engine failure
        return [
            ctrl.Rule(risk['medium'] & system_health['fair'], action['activate_redundancy']),
            ctrl.Rule(risk['high'] & time_criticality['medium'], action['change_route']),
            ctrl.Rule(risk['high'] & time_criticality['high'], action['emergency_landing'])
        ]
    # Add more states and their rules as needed

# Function to get the appropriate action based on the current state and inputs
def get_action(state, risk_level, time_crit, sys_health):
    rules = create_rules_for_state(state)
    action_ctrl = ctrl.ControlSystem(rules)
    action_simulation = ctrl.ControlSystemSimulation(action_ctrl)

    action_simulation.input['risk'] = risk_level
    action_simulation.input['time_criticality'] = time_crit
    action_simulation.input['system_health'] = sys_health
    action_simulation.compute()
    
    action_value = action_simulation.output['action']
    
    if action_value < 2:
        return "Change Route"
    elif action_value < 4:
        return "Activate Redundancy Systems"
    elif action_value < 6:
        return "Use Occupancy Maps"
    elif action_value < 8:
        return "Change Altitude"
    else:
        return "Emergency Landing"

def plot_fuzzy_sets(variable, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, term in variable.terms.items():
        ax.plot(variable.universe, term.mf, label=name)
    ax.set_title(title)
    ax.legend()
    ax.set_ylabel('Membership')
    ax.set_xlabel(variable.label)
    plt.tight_layout()
    plt.show()

# Plot fuzzy sets for each variable
# plot_fuzzy_sets(risk, 'Risk Fuzzy Sets')
# plot_fuzzy_sets(time_criticality, 'Time Criticality Fuzzy Sets')
# plot_fuzzy_sets(system_health, 'System Health Fuzzy Sets')
# plot_fuzzy_sets(action, 'Action Fuzzy Sets')

# Example usage
print(get_action("S1", 3, 2, 80))  # Normal operation, low risk, low time criticality, good system health
print(get_action("S2", 7, 8, 60))  # GPS Spoofing, high risk, high time criticality, fair system health
print(get_action("S4", 8, 9, 30))  # Engine failure, high risk, high time criticality, poor system health