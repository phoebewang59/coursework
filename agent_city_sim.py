import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------

GRID_SIZE = 30
N_AGENTS = 300
STEPS = 365
RNG_SEED = 42

CENTER = (GRID_SIZE // 2, GRID_SIZE // 2)
INNER_RADIUS = GRID_SIZE * 0.2
MID_RADIUS = GRID_SIZE * 0.35
OUTER_RADIUS = GRID_SIZE * 0.5

BASE_STRESS_DOWNTOWN = 0.5
BASE_STRESS_RESIDENTIAL_INNER = 0.7
BASE_STRESS_RESIDENTIAL_OUTER = 0.4
BASE_STRESS_SUBURB = 0.2
BASE_STRESS_INDUSTRIAL = 0.6

INITIAL_DISORDER = 0.1

A_STRESS_RETURN = 0.1
B_STRESS_FROM_DISORDER = 0.2

DISORDER_FROM_CRIME = 0.1
DISORDER_DECAY = 0.02

ALPHA_STRESS = 1.2
BETA_DISORDER = 1.0
GAMMA_NEIGHBOR_CRIME = 0.8
DELTA_VIOLENCE = 0.6

CONTAGION_FACTOR = 0.2

BROKEN_WINDOWS_STRENGTH = 0.25
SOCIAL_SUPPORT_STRENGTH = 0.25
INTERVENTION_TOP_FRACTION = 0.1

MOVE_PROB = 0.3  # probability an agent moves to neighboring cell each step

# NEW: crowding and social-support weights
CROWD_DENSITY_WEIGHT = 0.15        # how strongly local crowding raises crime risk
SUPPORT_EFFECT_WEIGHT = 0.25       # how strongly local support lowers crime risk
SUPPORT_COOLING = 0.1              # how much support reduces violence_tendency each step



# -----------------------------
# UTILS
# -----------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_distance_grid(center, size):
    y = np.arange(size)
    x = np.arange(size)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    dy = yy - center[0]
    dx = xx - center[1]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist


# -----------------------------
# INITIALIZATION
# -----------------------------

def initialize_environment(grid_size=GRID_SIZE, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    dist = get_distance_grid(CENTER, grid_size)
    base_stress = np.zeros((grid_size, grid_size))

    downtown_mask = dist <= INNER_RADIUS
    base_stress[downtown_mask] = BASE_STRESS_DOWNTOWN

    inner_res_mask = (dist > INNER_RADIUS) & (dist <= MID_RADIUS)
    base_stress[inner_res_mask] = BASE_STRESS_RESIDENTIAL_INNER

    outer_res_mask = (dist > MID_RADIUS) & (dist <= OUTER_RADIUS)
    base_stress[outer_res_mask] = BASE_STRESS_RESIDENTIAL_OUTER

    suburb_mask = dist > OUTER_RADIUS
    base_stress[suburb_mask] = BASE_STRESS_SUBURB

    # Industrial strip at top
    industrial_rows = 3
    industrial_mask = np.zeros_like(base_stress, dtype=bool)
    industrial_mask[:industrial_rows, :] = True
    base_stress[industrial_mask] = BASE_STRESS_INDUSTRIAL

    stress = base_stress + 0.05 * rng.normal(size=base_stress.shape)
    stress = np.clip(stress, 0, 1)

    disorder = np.full((grid_size, grid_size), INITIAL_DISORDER)

    env = {
        "base_stress": base_stress,
        "stress": stress,
        "disorder": disorder,
    }
    return env


def initialize_agents(n_agents=N_AGENTS, grid_size=GRID_SIZE, rng=None):
    """
    Each agent now has:
      - baseline_trouble: trait-level inclination to cause trouble
      - violence_tendency: dynamic state, starts at baseline_trouble
      - susceptibility: how much contagion affects them
      - supportiveness: how prosocial / buffering they are
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    xs = rng.integers(0, grid_size, size=n_agents)
    ys = rng.integers(0, grid_size, size=n_agents)

    # Trait: "born" troublemaker tendency
    baseline_trouble = np.clip(
        rng.normal(loc=0.3, scale=0.15, size=n_agents), 0, 1
    )

    # Violence tendency starts at trait level but will drift via contagion/support
    violence_tendency = baseline_trouble.copy()

    susceptibility = np.clip(
        rng.normal(loc=0.5, scale=0.1, size=n_agents), 0, 1
    )

    # Trait: how supportive/positive the person is (social buffer)
    supportiveness = np.clip(
        rng.normal(loc=0.5, scale=0.2, size=n_agents), 0, 1
    )

    agents = {
        "x": xs,
        "y": ys,
        "baseline_trouble": baseline_trouble,
        "violence_tendency": violence_tendency,
        "susceptibility": susceptibility,
        "supportiveness": supportiveness,
    }
    return agents


# -----------------------------
# NEIGHBORS & INTERVENTIONS
# -----------------------------

def neighbor_average_crime(cell_crime):
    up = np.roll(cell_crime, -1, axis=0)
    down = np.roll(cell_crime, 1, axis=0)
    left = np.roll(cell_crime, -1, axis=1)
    right = np.roll(cell_crime, 1, axis=1)
    return (cell_crime + up + down + left + right) / 5.0


def apply_broken_windows(disorder):
    flat = disorder.flatten()
    cutoff_index = int((1 - INTERVENTION_TOP_FRACTION) * flat.size)
    cutoff_index = max(cutoff_index, 0)
    threshold = np.partition(flat, cutoff_index)[cutoff_index]
    mask = disorder >= threshold
    disorder[mask] -= BROKEN_WINDOWS_STRENGTH
    np.clip(disorder, 0, 1, out=disorder)


def apply_social_support(stress):
    flat = stress.flatten()
    cutoff_index = int((1 - INTERVENTION_TOP_FRACTION) * flat.size)
    cutoff_index = max(cutoff_index, 0)
    threshold = np.partition(flat, cutoff_index)[cutoff_index]
    mask = stress >= threshold
    stress[mask] -= SOCIAL_SUPPORT_STRENGTH
    np.clip(stress, 0, 1, out=stress)


def compute_agent_fields(agents, grid_size=GRID_SIZE):
    """
    Compute:
      - how many agents in each cell (crowding)
      - total supportiveness in each cell (local social support)
    """
    counts = np.zeros((grid_size, grid_size), dtype=int)
    support_sum = np.zeros((grid_size, grid_size), dtype=float)

    np.add.at(counts, (agents["y"], agents["x"]), 1)
    np.add.at(support_sum, (agents["y"], agents["x"]), agents["supportiveness"])

    return counts, support_sum


# -----------------------------
# ONE STEP OF SIMULATION
# -----------------------------

def step(env, agents, prev_cell_crime, intervention="none", rng=None):
    """
    prev_cell_crime = crimes from previous step, used for contagion.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    stress = env["stress"]
    disorder = env["disorder"]
    base_stress = env["base_stress"]

    # 1. Environment: stress & disorder update
    stress = stress + A_STRESS_RETURN * (base_stress - stress) \
             + B_STRESS_FROM_DISORDER * disorder
    np.clip(stress, 0, 1, out=stress)

    disorder = disorder - DISORDER_DECAY
    np.clip(disorder, 0, 1, out=disorder)

    # 2. Cell crime count reset for this step
    cell_crime = np.zeros_like(stress)

    # 3. Agents move a bit
    moves = rng.random(len(agents["x"])) < MOVE_PROB
    dirs = rng.integers(0, 4, size=len(agents["x"]))
    dx = np.zeros_like(agents["x"])
    dy = np.zeros_like(agents["y"])
    dx[dirs == 0] = 1
    dx[dirs == 1] = -1
    dy[dirs == 2] = 1
    dy[dirs == 3] = -1
    agents["x"][moves] = (agents["x"][moves] + dx[moves]) % GRID_SIZE
    agents["y"][moves] = (agents["y"][moves] + dy[moves]) % GRID_SIZE

    # 4. Compute crowding & local social support
    counts, support_sum = compute_agent_fields(agents, grid_size=GRID_SIZE)
    # contagion from last step's crime (neighborhood level)
    neighbor_cr = neighbor_average_crime(prev_cell_crime)

    crimes = np.zeros(len(agents["x"]), dtype=int)

    # 5. Crime decision per agent
    for i in range(len(agents["x"])):
        x = agents["x"][i]
        y = agents["y"][i]
        local_stress = stress[y, x]
        local_disorder = disorder[y, x]
        agent_violence = agents["violence_tendency"][i]

        local_neighbor_crime = neighbor_cr[y, x]

        local_density = counts[y, x]
        local_support_mean = (
            support_sum[y, x] / local_density if local_density > 0 else 0.0
        )

        # crowding effect: more people = more chance of conflict
        crowd_term = CROWD_DENSITY_WEIGHT * max(0, local_density - 1)

        # social support effect: prosocial people dampen crime pressure
        support_term = SUPPORT_EFFECT_WEIGHT * local_support_mean

        crime_pressure = (ALPHA_STRESS * local_stress +
                          BETA_DISORDER * local_disorder +
                          GAMMA_NEIGHBOR_CRIME * local_neighbor_crime +
                          DELTA_VIOLENCE * agent_violence +
                          crowd_term -
                          support_term)

        p_crime = sigmoid(crime_pressure)
        if rng.random() < p_crime:
            crimes[i] = 1
            cell_crime[y, x] += 1

    # 6. Update environment from crime
    disorder = disorder + DISORDER_FROM_CRIME * (cell_crime > 0).astype(float)
    np.clip(disorder, 0, 1, out=disorder)

    # 7. Contagion & social support effects on violence_tendency
    if cell_crime.max() > 0:
        norm_crime = cell_crime / cell_crime.max()
    else:
        norm_crime = cell_crime

    for i in range(len(agents["x"])):
        x = agents["x"][i]
        y = agents["y"][i]
        local_norm_crime = norm_crime[y, x]

        local_density = counts[y, x]
        local_support_mean = (
            support_sum[y, x] / local_density if local_density > 0 else 0.0
        )

        # contagion: exposure to crime increases violence tendency
        agents["violence_tendency"][i] += (
            CONTAGION_FACTOR
            * agents["susceptibility"][i]
            * local_norm_crime
        )

        # social support: being around supportive people cools you down
        agents["violence_tendency"][i] -= SUPPORT_COOLING * local_support_mean

    np.clip(agents["violence_tendency"], 0, 1, out=agents["violence_tendency"])

    # 8. Interventions on environment
    if intervention == "broken_windows":
        apply_broken_windows(disorder)
    elif intervention == "social_support":
        apply_social_support(stress)
    elif intervention == "combined":
        apply_broken_windows(disorder)
        apply_social_support(stress)

    env["stress"] = stress
    env["disorder"] = disorder

    total_crime = crimes.sum()
    return total_crime, cell_crime


# -----------------------------
# RUN SIMULATION
# -----------------------------

def run_simulation(steps=STEPS, intervention="none", seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    env = initialize_environment(rng=rng)
    agents = initialize_agents(rng=rng)

    total_crime_hist = []
    last_cell_crime = np.zeros_like(env["stress"])  # for contagion across time

    for t in range(steps):
        total_crime, cell_crime = step(
            env, agents, last_cell_crime, intervention=intervention, rng=rng
        )
        total_crime_hist.append(total_crime)
        last_cell_crime = cell_crime

    return {
        "env": env,
        "agents": agents,
        "total_crime_hist": np.array(total_crime_hist),
        "last_cell_crime": last_cell_crime,
    }


# -----------------------------
# SIMPLE DEMO + PLOTS
# -----------------------------

if __name__ == "__main__":
    scenarios = ["none", "broken_windows", "social_support", "combined"]
    results = {}

    for scenario in scenarios:
        print(f"Running scenario: {scenario}")
        res = run_simulation(intervention=scenario)
        results[scenario] = res
        print(f"  Mean crime per step: {res['total_crime_hist'].mean():.2f}")

    # 1) Time series: crime over time for each scenario
    plt.figure()
    for scenario in scenarios:
        plt.plot(results[scenario]["total_crime_hist"], label=scenario)
    plt.xlabel("Days")
    plt.ylabel("Total crime events")
    plt.title("Crime over time by intervention")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Final crime heatmap for baseline (no intervention)
    baseline = results["none"]
    plt.figure()
    plt.imshow(baseline["last_cell_crime"])
    plt.colorbar(label="Crimes in last step")
    plt.title("Final crime heatmap (no intervention)")
    plt.tight_layout()
    plt.show()

    # 3) Final crime heatmap for combined intervention (for contrast)
    combined = results["combined"]
    plt.figure()
    plt.imshow(combined["last_cell_crime"])
    plt.colorbar(label="Crimes in last step")
    plt.title("Final crime heatmap (combined intervention)")
    plt.tight_layout()
    plt.show()
