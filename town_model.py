# town_model.py
from mesa import Model, Agent
try:
    from mesa.time import RandomActivation
except ImportError:
    class RandomActivation:
        """
        Minimal fallback scheduler replicating the subset of Mesa's RandomActivation
        the project relies on. It keeps agents in insertion order and exposes the
        `.agents` attribute so the rest of the model logic can iterate over them.
        """

        def __init__(self, model):
            self.model = model
            self.agents = []

        def add(self, agent):
            self.agents.append(agent)

        def step(self):
            self.model.random.shuffle(self.agents)
            for agent in self.agents:
                if hasattr(agent, "step"):
                    agent.step()
from mesa.datacollection import DataCollector
import numpy as np
import random


# ---------- POLICY ----------

class ShutdownPolicy:
    """
    Encodes different policy modes:
    - "none": no shutdown
    - "targeted": close high-risk leisure places (e.g., cafes), keep school/work mostly open
    - "full": strong lockdown: school + most work + leisure closed (except essential)
    """

    def __init__(self, model, mode: str = "none"):
        self.model = model
        self.mode = mode

    def allows_work(self, agent):
        # Service workers are "essential" and always allowed to work
        if agent.role == "service":
            return True

        if self.mode == "none":
            return True

        if self.mode == "targeted":
            # Offices + shops still open under targeted
            return True

        if self.mode == "full":
            # Only essential work allowed; owners + office workers stay home
            return False

        return True

    def allows_school(self, agent):
        if agent.role != "student":
            return False

        if self.mode == "none":
            return True

        if self.mode == "targeted":
            # You can choose: keep schools open or partially open.
            return True

        if self.mode == "full":
            # Full lockdown → schools closed
            return False

        return True

    def allows_leisure_location(self, location_name):
        loc_type = self.model.location_types.get(location_name, "other")

        if self.mode == "none":
            return True

        if self.mode == "targeted":
            # Parks allowed, cafes closed
            if loc_type == "park":
                return True
            if loc_type == "cafe":
                return False
            return True

        if self.mode == "full":
            # No leisure allowed
            return False

        return True

    def allows_business_open(self, agent):
        """
        For small business owners: is their shop allowed to open?
        Here we assume all owner businesses are "cafe" type.
        You can make this more detailed later.
        """
        if agent.role != "owner":
            return True

        if agent.business is None:
            return True

        loc_type = self.model.location_types.get(agent.business, "other")

        if self.mode == "none":
            return True

        if self.mode == "targeted":
            # cafes closed under targeted shutdown
            if loc_type == "cafe":
                return False
            return True

        if self.mode == "full":
            # All non-essential small businesses closed
            return False

        return True


# ---------- AGENT ----------

class TownAgent(Agent):
    def __init__(self, unique_id, model, role, home, work=None, business=None,
                 social_need=0.5, has_dependents=False):
        super().__init__(model)
        self.unique_id = unique_id
        self.role = role  # "student", "office", "service", "owner"
        self.home = home
        self.work = work        # office / school / shop / essential workplace
        self.business = business  # for owners (usually same as work)
        self.location = home

        # Health state
        self.health_state = "S"  # "S", "E", "I", "R"
        self.days_in_state = 0

        # Social / mental state
        self.stress = 0.2        # 0..1
        self.social_need = social_need  # 0..1 (how much they crave interactions)
        self.compliant = True
        self.has_dependents = has_dependents
        self.employed = True     # owners may effectively become "unemployed" if business stays closed

        # For daily stats
        self.today_contacts = set()

    # ---- Movement / routine ----

    def move(self, current_hour):
        policy = self.model.policy

        # Work / school hours
        if self.role in ["office", "service", "owner"]:
            if 9 <= current_hour < 17:  # work block
                if policy.allows_work(self) and self.work is not None:
                    self.location = self.work
                else:
                    self.location = self.home
            else:
                self._leisure_or_home(current_hour)

        elif self.role == "student":
            if 8 <= current_hour < 15:  # school block
                if policy.allows_school(self) and self.work is not None:
                    self.location = self.work  # here work = school
                else:
                    self.location = self.home
            else:
                self._leisure_or_home(current_hour)

        else:
            # Fallback: just stay home or go leisure
            self._leisure_or_home(current_hour)

    def _leisure_or_home(self, current_hour):
        """
        Decide whether to go to a leisure location (park/cafe/etc.)
        or stay home, based on social need, stress and policy.
        """
        policy = self.model.policy

        # Base probability: how social this person is
        base_prob = self.social_need

        # Stress effect: high stress → more likely to break rules (you could flip this later)
        stress_bonus = 0.4 * self.stress
        p_go_out = min(1.0, base_prob + stress_bonus)

        if self.random.random() < p_go_out:
            # Try to pick a leisure location allowed by policy
            allowed = [
                loc for loc in self.model.leisure_locations
                if policy.allows_leisure_location(loc)
            ]
            if allowed:
                self.location = self.random.choice(allowed)
            else:
                self.location = self.home
        else:
            self.location = self.home

    # ---- Health & stress updating ----

    def update_health(self):
        if self.health_state in ["E", "I"]:
            self.days_in_state += 1

        # SEIR progression
        if self.health_state == "E" and self.days_in_state > self.model.incubation_hours:
            self.health_state = "I"
            self.days_in_state = 0
        elif self.health_state == "I" and self.days_in_state > self.model.infectious_hours:
            self.health_state = "R"
            self.days_in_state = 0

    def update_stress(self):
        m = self.model

        # Natural decay
        self.stress = max(0.0, self.stress - m.stress_decay)

        # Isolation stress: fewer contacts than they "want"
        desired_contacts = self.social_need * m.target_contacts
        if len(self.today_contacts) < desired_contacts:
            self.stress = min(1.0, self.stress + m.isolation_stress)

        # Fear stress: interacted with infectious agents
        infected_contacts = sum(1 for a in self.today_contacts if a.health_state == "I")
        if infected_contacts > 0:
            self.stress = min(1.0, self.stress + m.fear_stress)

        # Economic stress: small business owners whose business is closed
        if self.role == "owner":
            if not m.policy.allows_business_open(self):
                self.stress = min(1.0, self.stress + m.economic_stress)

        # Dependents: if they have dependents and high infection around, extra stress
        if self.has_dependents and infected_contacts > 0:
            self.stress = min(1.0, self.stress + m.dependent_stress)

        # Social influence: move towards neighbors' average stress
        if self.today_contacts:
            avg_neighbor_stress = np.mean([a.stress for a in self.today_contacts])
            alpha = m.stress_influence
            self.stress = (1 - alpha) * self.stress + alpha * avg_neighbor_stress

        # Compliance change: very high stress → more likely to stop complying
        if self.stress > 0.8 and self.random.random() < 0.1:
            self.compliant = False


# ---------- MODEL ----------

class TownModel(Model):
    def __init__(self, N=200, policy_mode="none", seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.random.seed(seed)

        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.current_hour = 0

        # Disease parameters
        # Disease parameters (in hours)
        self.incubation_hours = 3 * 24
        self.infectious_hours = 7 * 24
        self.beta = 0.05  # infection prob per contact per infected in location

        # Stress parameters (calibrated for hourly steps)
        self.stress_decay = 0.001
        self.isolation_stress = 0.002
        self.fear_stress = 0.005
        self.economic_stress = 0.005
        self.dependent_stress = 0.002
        self.stress_influence = 0.1
        self.target_contacts = 5

        # Policy
        self.location_types = {}   # name -> type, filled below
        self.policy = ShutdownPolicy(self, policy_mode)

        # --- Locations ---
        # Homes: households of size 1-4
        self.home_locations = []
        self.agent_homes = []
        
        agents_left = N
        home_count = 0
        while agents_left > 0:
            size = random.randint(1, 4)
            if size > agents_left:
                size = agents_left
            
            h_name = f"home_{home_count}"
            self.home_locations.append(h_name)
            self.location_types[h_name] = "home"
            
            # Assign this home to 'size' agents
            for _ in range(size):
                self.agent_homes.append(h_name)
            
            agents_left -= size
            home_count += 1
            
        # Shuffle assignments so roles are mixed in households
        random.shuffle(self.agent_homes)

        # Offices: fewer than agents
        num_offices = max(1, N // 10)
        self.office_locations = [f"office_{i}" for i in range(num_offices)]
        for o in self.office_locations:
            self.location_types[o] = "office"

        # Service workplaces (e.g., grocery, logistics)
        num_service_sites = max(1, N // 20)
        self.service_locations = [f"service_{i}" for i in range(num_service_sites)]
        for s in self.service_locations:
            self.location_types[s] = "service"

        # Schools
        num_schools = max(1, N // 30)
        self.school_locations = [f"school_{i}" for i in range(num_schools)]
        for sc in self.school_locations:
            self.location_types[sc] = "school"

        # Small business / cafes for owners
        num_cafes = max(1, N // 25)
        self.cafe_locations = [f"cafe_{i}" for i in range(num_cafes)]
        for c in self.cafe_locations:
            self.location_types[c] = "cafe"

        # Parks (public leisure)
        num_parks = max(1, N // 25)
        self.park_locations = [f"park_{i}" for i in range(num_parks)]
        for p in self.park_locations:
            self.location_types[p] = "park"

        # Leisure = cafes + parks (for now)
        self.leisure_locations = self.cafe_locations + self.park_locations

        # --- Create agents with roles ---

        # Role distribution (you can tweak these)
        n_students = int(0.3 * N)
        n_office = int(0.4 * N)
        n_service = int(0.2 * N)
        n_owner = N - (n_students + n_office + n_service)

        all_agents = []

        # Helper index for homes
        # home_idx = 0  <-- Removed, using self.agent_homes instead

        # Students
        for _ in range(n_students):
            home = self.agent_homes[len(all_agents)]
            school = random.choice(self.school_locations)
            social_need = random.uniform(0.4, 0.9)
            has_dependents = False
            a = TownAgent(
                unique_id=len(all_agents),
                model=self,
                role="student",
                home=home,
                work=school,
                business=None,
                social_need=social_need,
                has_dependents=has_dependents
            )
            all_agents.append(a)

        # Office workers
        for _ in range(n_office):
            home = self.agent_homes[len(all_agents)]
            office = random.choice(self.office_locations)
            social_need = random.uniform(0.3, 0.8)
            has_dependents = random.random() < 0.3
            a = TownAgent(
                unique_id=len(all_agents),
                model=self,
                role="office",
                home=home,
                work=office,
                business=None,
                social_need=social_need,
                has_dependents=has_dependents
            )
            all_agents.append(a)

        # Service workers (essential)
        for _ in range(n_service):
            home = self.agent_homes[len(all_agents)]
            service_site = random.choice(self.service_locations)
            social_need = random.uniform(0.2, 0.7)
            has_dependents = random.random() < 0.4
            a = TownAgent(
                unique_id=len(all_agents),
                model=self,
                role="service",
                home=home,
                work=service_site,
                business=None,
                social_need=social_need,
                has_dependents=has_dependents
            )
            all_agents.append(a)

        # Small business owners
        for _ in range(n_owner):
            home = self.agent_homes[len(all_agents)]
            cafe = random.choice(self.cafe_locations)
            social_need = random.uniform(0.4, 0.9)
            has_dependents = random.random() < 0.5
            a = TownAgent(
                unique_id=len(all_agents),
                model=self,
                role="owner",
                home=home,
                work=cafe,
                business=cafe,
                social_need=social_need,
                has_dependents=has_dependents
            )
            all_agents.append(a)

        # Register agents with scheduler
        for a in all_agents:
            self.schedule.add(a)

        # Seed a few initial infections
        initially_infected = random.sample(all_agents, k=min(3, len(all_agents)))
        for a in initially_infected:
            a.health_state = "I"
            a.days_in_state = 0

        # --- Data collection ---

        self.datacollector = DataCollector(
            model_reporters={
                "Infected": lambda m: sum(1 for a in m.schedule.agents if a.health_state == "I"),
                "AvgStress": lambda m: np.mean([a.stress for a in m.schedule.agents]),
                "HighStressFrac": lambda m: np.mean([a.stress > 0.8 for a in m.schedule.agents]),
                # By-role examples (optional, useful for plots)
                "AvgStress_Student": lambda m: np.mean(
                    [a.stress for a in m.schedule.agents if a.role == "student"]
                ) if any(a.role == "student" for a in m.schedule.agents) else 0.0,
                "AvgStress_Office": lambda m: np.mean(
                    [a.stress for a in m.schedule.agents if a.role == "office"]
                ) if any(a.role == "office" for a in m.schedule.agents) else 0.0,
                "AvgStress_Service": lambda m: np.mean(
                    [a.stress for a in m.schedule.agents if a.role == "service"]
                ) if any(a.role == "service" for a in m.schedule.agents) else 0.0,
                "AvgStress_Owner": lambda m: np.mean(
                    [a.stress for a in m.schedule.agents if a.role == "owner"]
                ) if any(a.role == "owner" for a in m.schedule.agents) else 0.0,
            }
        )

    def step(self):
        """
        One simulation step = one hour.
        Order:
        1. Everyone decides where to go (move).
        2. We compute co-location → interactions & infections.
        3. Everyone updates health & stress.
        4. We record statistics.
        """
        # Reset daily contacts at the start of the "day" if hour == 0
        if self.current_hour == 0:
            for a in self.schedule.agents:
                a.today_contacts.clear()

        # 1. Movement
        for a in self.schedule.agents:
            a.move(self.current_hour)

        # 2. Group by location, handle interactions & infection
        loc_dict = {}
        for a in self.schedule.agents:
            loc_dict.setdefault(a.location, []).append(a)

        for loc, agents in loc_dict.items():
            if len(agents) < 2:
                continue

            # Record contacts
            for i, ai in enumerate(agents):
                for aj in agents[i + 1:]:
                    ai.today_contacts.add(aj)
                    aj.today_contacts.add(ai)

            # Infection dynamics
            infectious_agents = [a for a in agents if a.health_state == "I"]
            susceptibles = [a for a in agents if a.health_state == "S"]
            for s in susceptibles:
                if not infectious_agents:
                    continue
                # Simple infection probability scaled by number of infectious agents
                prob = 1 - (1 - self.beta) ** len(infectious_agents)
                if self.random.random() < prob:
                    s.health_state = "E"
                    s.days_in_state = 0

        # 3. Health & stress update
        for a in self.schedule.agents:
            a.update_health()
            a.update_stress()

        # 4. Collect data
        self.datacollector.collect(self)

        # Advance time
        self.current_hour = (self.current_hour + 1) % 24
