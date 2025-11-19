import random
from town_model import TownModel

def verify():
    print("Initializing model...")
    model = TownModel(N=200, policy_mode="none", seed=42)
    
    # 1. Verify Household Sizes
    print("\nVerifying Household Sizes...")
    home_counts = {}
    for a in model.schedule.agents:
        home_counts[a.home] = home_counts.get(a.home, 0) + 1
    
    sizes = list(home_counts.values())
    min_size = min(sizes)
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)
    
    print(f"Total homes: {len(home_counts)}")
    print(f"Min household size: {min_size}")
    print(f"Max household size: {max_size}")
    print(f"Avg household size: {avg_size:.2f}")
    
    if min_size < 1 or max_size > 4:
        print("FAIL: Household sizes out of range [1, 4]")
    else:
        print("PASS: Household sizes within range [1, 4]")

    # 2. Verify Time Scale & Infection
    print("\nVerifying Time Scale & Infection...")
    # Force an infection
    patient_zero = model.schedule.agents[0]
    patient_zero.health_state = "I"
    patient_zero.days_in_state = 0
    
    # Run for 24 hours (1 day)
    initial_infected = sum(1 for a in model.schedule.agents if a.health_state == "I")
    print(f"Initial infected: {initial_infected}")
    
    for i in range(24):
        model.step()
        
    final_infected = sum(1 for a in model.schedule.agents if a.health_state == "I")
    exposed = sum(1 for a in model.schedule.agents if a.health_state == "E")
    
    print(f"After 24 steps (1 day):")
    print(f"Infected: {final_infected}")
    print(f"Exposed: {exposed}")
    
    # Check if patient zero is still infected (should be, as infectious_hours = 7*24)
    if patient_zero.health_state == "I":
        print("PASS: Patient zero still infected after 24h (expected)")
    else:
        print(f"FAIL: Patient zero state is {patient_zero.health_state}, expected 'I'")

    # 3. Verify Stress
    print("\nVerifying Stress...")
    avg_stress = model.datacollector.get_model_vars_dataframe()["AvgStress"].iloc[-1]
    print(f"Average Stress after 24h: {avg_stress:.4f}")
    if 0 <= avg_stress <= 1:
        print("PASS: Stress within bounds")
    else:
        print("FAIL: Stress out of bounds")

if __name__ == "__main__":
    verify()
