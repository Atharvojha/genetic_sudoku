# backend/ga_solver.py
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

def repair_sequence(seq, allowed):
    """
    Repair a sequence so that it forms a permutation of the allowed numbers.
    It replaces duplicates (or invalid values) with the missing numbers.
    """
    result = seq.copy()
    counts = {}
    for num in result:
        counts[num] = counts.get(num, 0) + 1
    # Find which numbers are missing from the allowed set.
    missing = [num for num in allowed if num not in result]
    # Replace duplicates with missing numbers.
    for i in range(len(result)):
        if result[i] not in allowed:
            if missing:
                result[i] = missing.pop()
        else:
            if counts[result[i]] > 1:
                counts[result[i]] -= 1
                if missing:
                    result[i] = missing.pop()
    return result

def crossover_row(row1, row2, fixed_mask):
    """
    Row-level crossover for a single row.
    Only free positions (where fixed_mask is False) are allowed to be swapped.
    The free-part is recombined (using a random crossover point) and then repaired so that
    the row becomes a valid permutation of the allowed numbers.
    """
    new_row = row1.copy()
    free_indices = np.where(~fixed_mask)[0]
    if len(free_indices) == 0:
        return row1.copy()
    # Extract free parts from each parent.
    parent1_free = [row1[i] for i in free_indices]
    parent2_free = [row2[i] for i in free_indices]
    # Choose a random crossover point.
    if len(free_indices) > 1:
        cp = random.randint(1, len(free_indices) - 1)
        child_free = parent1_free[:cp] + parent2_free[cp:]
    else:
        child_free = parent1_free if random.random() < 0.5 else parent2_free
    # Determine allowed free numbers based on the puzzle's fixed cells.
    allowed = list(set(range(1, 10)) - set(row1[fixed_mask]))
    child_free = repair_sequence(child_free, allowed)
    # Put the free numbers back into the new row.
    for idx, pos in enumerate(free_indices):
        new_row[pos] = child_free[idx]
    return new_row

def mutate_row(row, fixed_mask, mutation_rate):
    """
    Mutation: With a given probability, swap two numbers among free positions.
    """
    free_indices = np.where(~fixed_mask)[0]
    if len(free_indices) >= 2 and random.random() < mutation_rate:
        a, b = random.sample(list(free_indices), 2)
        row[a], row[b] = row[b], row[a]
    return row

def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Tournament selection: select the best candidate among 'tournament_size' randomly chosen candidates.
    """
    selected_index = random.choice(range(len(population)))
    for _ in range(tournament_size - 1):
        candidate = random.choice(range(len(population)))
        if fitnesses[candidate] < fitnesses[selected_index]:
            selected_index = candidate
    # Return a deep copy of the selected candidate.
    return copy.deepcopy(population[selected_index])

def fitness(candidate):
    """
    Fitness function computes a penalty based on duplicates in columns and 3x3 blocks.
    Lower fitness is better; a fitness of 0 corresponds to a perfect Sudoku solution.
    """
    penalty = 0
    # Evaluate penalty in columns.
    for col in range(9):
        col_values = candidate[:, col]
        for num in range(1, 10):
            count = np.sum(col_values == num)
            if count > 1:
                penalty += (count - 1)
    # Evaluate penalty in 3x3 blocks.
    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            block = candidate[block_row:block_row + 3, block_col:block_col + 3].flatten()
            for num in range(1, 10):
                count = np.sum(block == num)
                if count > 1:
                    penalty += (count - 1)
    return penalty

def genetic_algorithm(puzzle, pop_size=100, generations=1000, crossover_rate=0.8, mutation_rate=0.1, verbose=True):
    """
    Main GA loop:
      - Initializes a population based on the provided Sudoku puzzle.
      - Evolves the population over generations using tournament selection, crossover, and mutation.
      - Tracks and optionally plots the best fitness progress over time.
      
    :param puzzle: 9x9 numpy array with given fixed cells (nonzero) and blanks as 0.
    :returns: The best solution (as a numpy array) and a list of best fitness per generation.
    """
    # Convert puzzle to a numpy array (if not already) and get fixed cell mask.
    puzzle_np = np.array(puzzle)
    fixed_mask = puzzle_np != 0

    # Initialize the population.
    population = []
    for _ in range(pop_size):
        candidate = np.zeros((9, 9), dtype=int)
        for i in range(9):
            row = puzzle_np[i].copy()
            free_indices = np.where(row == 0)[0]
            # Allowed numbers for the row are those not already fixed.
            allowed = list(set(range(1, 10)) - set(row))
            random.shuffle(allowed)
            for idx, pos in enumerate(free_indices):
                # Assign the missing numbers randomly.
                row[pos] = allowed[idx] if idx < len(allowed) else random.randint(1, 9)
            candidate[i] = row
        population.append(candidate)

    best_progress = []
    best_solution = None
    best_fit = float('inf')
    for gen in range(generations):
        fitnesses = np.array([fitness(candidate) for candidate in population])
        gen_best_index = np.argmin(fitnesses)
        gen_best_fit = fitnesses[gen_best_index]
        if verbose and gen % 100 == 0:
            print(f"Generation {gen}: Best Fitness = {gen_best_fit}")
        best_progress.append(gen_best_fit)
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_solution = copy.deepcopy(population[gen_best_index])
        if best_fit == 0:
            if verbose:
                print(f"Solution found at generation {gen}!")
            break

        new_population = []
        # Create new population.
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = np.zeros((9, 9), dtype=int)
            for i in range(9):
                row_fixed_mask = fixed_mask[i]
                # Apply crossover with a given probability.
                if random.random() < crossover_rate:
                    child[i] = crossover_row(parent1[i], parent2[i], row_fixed_mask)
                else:
                    child[i] = copy.deepcopy(parent1[i]) if random.random() < 0.5 else copy.deepcopy(parent2[i])
                # Apply mutation.
                child[i] = mutate_row(child[i], row_fixed_mask, mutation_rate)
            new_population.append(child)
        population = new_population

    # Optional: plot GA progress.
    plt.figure()
    plt.plot(best_progress)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Progress over Generations")
    plt.show()

    return best_solution, best_progress

# Uncomment below for quick standalone testing:
# if __name__ == "__main__":
#     # Example puzzle: 0 means blank.
#     puzzle = [
#         [5, 3, 0, 0, 7, 0, 0, 0, 0],
#         [6, 0, 0, 1, 9, 5, 0, 0, 0],
#         [0, 9, 8, 0, 0, 0, 0, 6, 0],
#         [8, 0, 0, 0, 6, 0, 0, 0, 3],
#         [4, 0, 0, 8, 0, 3, 0, 0, 1],
#         [7, 0, 0, 0, 2, 0, 0, 0, 6],
#         [0, 6, 0, 0, 0, 0, 2, 8, 0],
#         [0, 0, 0, 4, 1, 9, 0, 0, 5],
#         [0, 0, 0, 0, 8, 0, 0, 7, 9]
#     ]
#     solution, progress = genetic_algorithm(puzzle)
#     print("Best solution found:")
#     print(solution)
