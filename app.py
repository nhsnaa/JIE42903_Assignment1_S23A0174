import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="GA TV Scheduling System", layout="wide")

st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.markdown("Adjust parameters below to run the Genetic Algorithm and generate an optimized schedule.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Genetic Algorithm Parameters")

co_r = st.sidebar.slider(
    "Crossover Rate (CO_R)", 
    min_value=0.0, 
    max_value=0.95, 
    value=0.80, 
    step=0.05
)

mut_r = st.sidebar.slider(
    "Mutation Rate (MUT_R)", 
    min_value=0.01, 
    max_value=0.05, 
    value=0.02, 
    step=0.005
)

generations = st.sidebar.number_input("Generations (GEN)", min_value=10, max_value=500, value=100)
pop_size = st.sidebar.number_input("Population Size (POP)", min_value=10, max_value=200, value=50)

# --- LOAD CSV DATA ---
@st.cache_data
def load_ratings(file_path):
    df = pd.read_csv(file_path)
    program_ratings = {}
    for _, row in df.iterrows():
        program = str(row.iloc[0])
        ratings_list = [float(x) for x in row.iloc[1:].values]
        program_ratings[program] = ratings_list
    return df, program_ratings

try:
    df_display, ratings = load_ratings("modified_ratings.csv")
    st.subheader("📊 Modified Program Ratings Dataset")
    st.dataframe(df_display, use_container_width=True)
except Exception as e:
    st.error("Error loading 'modified_ratings.csv'. Please ensure the file is uploaded in the repository.")
    st.stop()

# --- ALGORITHM LOGIC ---
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))

def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule

def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size=2):
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

# --- RUN GA ON BUTTON CLICK ---
st.subheader("⚙️ Generate Optimized Schedule")

if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing schedule..."):
        all_possible_schedules = initialize_pop(all_programs, all_time_slots)
        initial_best_schedule = finding_best_schedule(all_possible_schedules)

        rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
        genetic_schedule = genetic_algorithm(
            initial_best_schedule, 
            generations=generations, 
            population_size=pop_size, 
            crossover_rate=co_r, 
            mutation_rate=mut_r
        )

        final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]
        total_score = fitness_function(final_schedule)

        schedule_data = []
        for time_slot, program in enumerate(final_schedule):
            hour = all_time_slots[time_slot]
            time_str = f"{hour:02d}:00 - {hour+1:02d}:00"
            rating_val = ratings[program][time_slot]
            schedule_data.append({
                "Time Slot": time_str,
                "Program Scheduled": program,
                "Rating Contribution": round(rating_val, 2)
            })

        result_df = pd.DataFrame(schedule_data)

        st.success(f"Optimal Schedule Generated! Total Ratings Score: {round(total_score, 2)}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(result_df, use_container_width=True)
        with col2:
            st.metric(label="Total Rating Score", value=round(total_score, 2))
            st.write(f"**Crossover Rate (CO_R):** {co_r}")
            st.write(f"**Mutation Rate (MUT_R):** {mut_r}")
