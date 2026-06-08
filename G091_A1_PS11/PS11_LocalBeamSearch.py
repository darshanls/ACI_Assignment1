import sys
import os
import traceback


# ====================== BEAM DATA STRUCTURE ======================
class Beam:
    """
    Fixed-capacity Beam data structure for Local Beam Search.
    Implements insert operation with appropriate messages when capacity is full or empty.
    This satisfies the assignment requirement for data structure capacity handling.
    """
    def __init__(self, max_size=2):
        """Initialize the Beam with a specified maximum size (capacity) and an empty list to hold the states. """
        self.max_size = max_size
        self.states = []

    def is_full(self):
        """Check if the beam is full by comparing the length of the states list to the maximum size."""
        return len(self.states) >= self.max_size

    def is_empty(self):
        """Check if the beam is empty by checking if the length of the states list is zero."""
        return len(self.states) == 0

    def insert(self, state):
        """Insert state. Shows message if beam capacity is full."""
        if self.is_full():
            print("[BEAM CAPACITY] Beam is FULL (max capacity = {}). Cannot insert more states.".format(self.max_size))
            return False
        self.states.append(state)
        return True

    def delete(self):
        """Delete and return the last state. Shows message if beam is empty."""
        if self.is_empty():
            print("[BEAM CAPACITY] Beam is EMPTY. No states to delete.")
            return None
        return self.states.pop()

    def get_states(self):
        """Return the current list of states in the beam."""
        return self.states

    def set_states(self, new_states):
        """Replace beam with new top-k states (enforces capacity)."""
        if len(new_states) > self.max_size:
            print("[BEAM CAPACITY] Warning: Truncating to max capacity {}".format(self.max_size))
        self.states = new_states[:self.max_size]

    def __len__(self):
        """Return the current number of states in the beam."""
        return len(self.states)


def read_input(file_path):
    """
    Reads the input grid file.
    Expected format:
    Line 1: rows cols
    Line 2: start_row start_col
    Line 3: goal_row goal_col
    Line 4 to end: grid rows (space separated)
    Raises appropriate errors for missing/malformed input.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Input file not found at: {}".format(file_path))

    print("Reading input file from path: {}\n".format(file_path))

    with open(file_path, 'r') as f:
        raw_lines = f.readlines()

    if not raw_lines:
        raise ValueError("Input file is empty!")

    # Show raw file content
    # print("=== RAW CONTENT OF INPUT FILE ===")
    # for i, line in enumerate(raw_lines):
    #     print("Line {}: '{}'".format(i, line.strip()))
    # print("================================\n")

    # Clean lines (remove empty)
    lines = [line.strip() for line in raw_lines if line.strip()]

    if len(lines) < 4:
        raise ValueError("Input file must have at least 4 lines (dimensions, start, goal, and at least one grid row).")

    # Parse metadata
    rows, cols = map(int, lines[0].split())
    start_row, start_col = map(int, lines[1].split())
    goal_row, goal_col = map(int, lines[2].split())

    if len(lines) < 3 + rows:
        raise ValueError("Expected {} grid rows but found only {} data lines.".format(rows, len(lines) - 3))

    # Validate start and goal bounds
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        raise ValueError("Start position ({},{}) is out of grid bounds ({}x{}).".format(start_row, start_col, rows, cols))
    if not (0 <= goal_row < rows and 0 <= goal_col < cols):
        raise ValueError("Goal position ({},{}) is out of grid bounds ({}x{}).".format(goal_row, goal_col, rows, cols))

    print("=== INPUT METADATA ===")
    print("Rows          : {}".format(rows))
    print("Columns       : {}".format(cols))
    print("Start Position: ({}, {})".format(start_row, start_col))
    print("Goal Position : ({}, {})".format(goal_row, goal_col))
    print("======================\n")

    # Read grid
    grid = []
    for i in range(3, 3 + rows):
        row = lines[i].split()
        if len(row) != cols:
            raise ValueError("Grid row {} has {} columns, expected {}.".format(i - 3, len(row), cols))
        grid.append(row)

    # Print the parsed grid for verification
    print("=== PARSED GRID ===")
    print("     " + "  ".join(str(c) for c in range(cols)))
    for r_idx, row in enumerate(grid):
        print("  {}  ".format(r_idx) + "  ".join(row))
    print("===================\n")

    return rows, cols, (start_row, start_col), (goal_row, goal_col), grid


def get_cost(cell):
    """Cost function: returns 3 for high-cost 'C' cells (1 base + 2 penalty) and 1 for normal cells."""
    return 3 if cell == 'C' else 1


def is_valid(r, c, rows, cols, grid):
    """Check if position (r, c) is within grid bounds and not a blocked 'X' cell."""
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'X'


def manhattan(r, c, goal):
    """Heuristic function: Manhattan distance from (r, c) to goal. h(n) = |x_g - x_n| + |y_g - y_n|"""
    return abs(r - goal[0]) + abs(c - goal[1])


def local_beam_search(rows, cols, start, goal, grid, k=2):
    """
    Local Beam Search with beam width k.
    At each iteration:
    - Expand all states in current beam (generate successors Up/Down/Left/Right)
    - Filter invalid (out of bounds, blocked 'X', or already visited in that path)
    - Remove duplicate positions, keep the one with lowest g (traversal cost)
    - Compute h = Manhattan distance for each
    - Compute g = cumulative traversal cost (1 for normal, 3 for 'C' cells)
    - Select top k states with smallest h (tie-break by smaller g)
    Returns: (path, total_cost, iteration_log) or (None, None, log) if no solution.
    """
    log_lines = []  # Collects output for writing to file

    def log(msg):
        """Helper to print and record a message simultaneously."""
        print(msg)
        log_lines.append(msg)

    log("")
    log("=" * 70)
    log("LOCAL BEAM SEARCH EXECUTION (k={})".format(k))
    log("=" * 70)
    log("Heuristic: h(n) = Manhattan Distance = |x_g - x_n| + |y_g - y_n|")
    log("Cost: g(n) = cumulative path cost from start; normal cell cost=1, high-cost 'C' cell cost=3 (1 + 2 penalty)")
    log("Selection: best k states by lowest h, then lowest g as tie-breaker")
    log("=" * 70)
    log("")

    beam = Beam(max_size=k)
    beam.insert((start, 0, [start]))
    iteration = 0
    max_iter = rows * cols * 2  # Safety limit proportional to grid size

    while not beam.is_empty() and iteration < max_iter:
        log("")
        log("=" * 60)
        log("ITERATION {}".format(iteration))
        log("=" * 60)

        current_states = beam.get_states()
        # Display current beam states with h and g
        log("")
        log("[SELECTED BEAM STATES] (k={} best so far):".format(k))
        for (r, c), g, path in current_states:
            h = manhattan(r, c, goal)
            log("  ({},{}) | h={:2d} | g={:2d} | path_len={}".format(r, c, h, g, len(path)))

        # Check if goal reached in current beam
        for (r, c), g, path in current_states:
            if (r, c) == goal:
                log("")
                log("*** GOAL REACHED in beam ***")
                return path, g, log_lines

        # Generate successor states (Up, Down, Left, Right)
        log("")
        log("[GENERATING SUCCESSORS from current beam]:")
        candidates = []
        directions = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
        for (r, c), g, path in current_states:
            visited_in_path = set(path)  # Avoid revisiting cells already in this path
            successors_for_state = []
            for dr, dc, dname in directions:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc, rows, cols, grid) and (nr, nc) not in visited_in_path:
                    cell = grid[nr][nc]
                    cost = get_cost(cell)
                    new_g = g + cost
                    new_path = path + [(nr, nc)]
                    candidates.append(((nr, nc), new_g, new_path))
                    successors_for_state.append(
                        "({},{})[dir={},cell={},cost+={},new_g={}]".format(nr, nc, dname, cell, cost, new_g)
                    )
            if successors_for_state:
                log("  From ({},{}) g={}: {}".format(r, c, g, ", ".join(successors_for_state)))

        if not candidates:
            log("")
            log("No valid successors generated! Search failed.")
            return None, None, log_lines

        log("")
        log("Total raw successors generated: {}".format(len(candidates)))

        # Remove duplicates: keep best (lowest g) for each position
        log("")
        log("[REMOVING DUPLICATES - keep lowest g per position]:")
        best = {}
        for pos, new_g, new_path in candidates:
            if pos not in best or new_g < best[pos][1]:
                best[pos] = (pos, new_g, new_path)
        candidate_list = list(best.values())
        log("  Unique positions after dedup: {}".format(len(candidate_list)))
        for pos, g_val, _ in sorted(candidate_list, key=lambda x: (manhattan(x[0][0], x[0][1], goal), x[1])):
            h_val = manhattan(pos[0], pos[1], goal)
            log("    ({},{}) | h={:2d} | g={:2d}".format(pos[0], pos[1], h_val, g_val))

        # Sort by min heuristic, then min g for tie-breaking
        candidate_list.sort(key=lambda x: (manhattan(x[0][0], x[0][1], goal), x[1]))

        # Select best k states
        log("")
        log("[SELECTING TOP {} BEAM STATES by min h then min g]:".format(k))
        beam.set_states(candidate_list[:k])
        for i, ((r, c), g, _) in enumerate(beam.get_states()):
            h = manhattan(r, c, goal)
            log("  #{}: ({},{}) | h={:2d} | g={:2d}".format(i + 1, r, c, h, g))

        iteration += 1

    if iteration >= max_iter:
        log("")
        log("Max iterations reached ({}) - possible loop or no path.".format(max_iter))
    return None, None, log_lines


def write_output(output_file, log_lines, path, total_cost):
    """
    Writes the full execution trace and final results to the output file.
    Includes: selected beam states, heuristic values, traversal costs,
    final path, total path cost, and total optimized traversal cost.
    """
    with open(output_file, "w") as f:
        # Write full iteration-by-iteration trace
        for line in log_lines:
            f.write(line + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("FINAL RESULT\n")
        f.write("=" * 70 + "\n")

        if path:
            path_str = " -> ".join(["({},{})".format(r, c) for r, c in path])
            f.write("Final Path from Start to Goal:\n{}\n".format(path_str))
            f.write("Total Path Cost (number of steps): {}\n".format(len(path) - 1))
            f.write("Total Optimized Traversal Cost: {}\n".format(total_cost))
        else:
            f.write("No solution found.\n")

        f.write("=" * 70 + "\n")


# ====================== MAIN ======================
# The main block reads the input file, executes the local beam search algorithm,
# displays and writes all required output, with full error handling.
if __name__ == "__main__":
    try:
        # Determine file paths relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)

        input_file = os.path.join(project_dir, "input", "inputPS11.txt")
        output_file = os.path.join(project_dir, "output", "outputPS11.txt")

        print("=" * 70)
        print("FILE PATHS USED:")
        print("Input File Path  -  {}".format(input_file))
        print("Output File Path -  {}".format(output_file))
        print("=" * 70 + "\n")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        rows, cols, start, goal, grid = read_input(input_file)
        print("Grid Size: {} x {}\n".format(rows, cols))

        path, total_cost, log_lines = local_beam_search(rows, cols, start, goal, grid, k=2)

        if path:
            path_str = " -> ".join(["({},{})".format(r, c) for r, c in path])
            print("\n" + "=" * 70)
            print("FINAL RESULT")
            print("=" * 70)
            print("Final Path          : {}".format(path_str))
            print("Total Path Cost     : {} steps".format(len(path) - 1))
            print("Total Traversal Cost: {}".format(total_cost))
            print("=" * 70)

            write_output(output_file, log_lines, path, total_cost)
            print("\nOutput written to: {}".format(output_file))
        else:
            print("\nNo solution found.")
            write_output(output_file, log_lines, None, None)
            print("Output written to: {}".format(output_file))

    except FileNotFoundError as e:
        print("\nFILE ERROR: {}".format(e))
        traceback.print_exc()
    except ValueError as e:
        print("\nINPUT ERROR: {}".format(e))
        traceback.print_exc()
    except Exception as e:
        print("\nUNEXPECTED ERROR: {}".format(e))
        traceback.print_exc()
