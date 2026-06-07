import sys
import os
import traceback


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
    print("=== RAW CONTENT OF INPUT FILE ===")
    for i, line in enumerate(raw_lines):
        print("Line {}: '{}'".format(i, line.strip()))
    print("================================\n")

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

    current_beam = [(start, 0, [start])]  # List of (position, g_cost, path_list)
    iteration = 0
    max_iter = rows * cols * 2  # Safety limit proportional to grid size

    while current_beam and iteration < max_iter:
        log("")
        log("=" * 60)
        log("ITERATION {}".format(iteration))
        log("=" * 60)

        # Display current beam states with h and g
        log("")
        log("[SELECTED BEAM STATES] (k={} best so far):".format(k))
        for (r, c), g, path in current_beam:
            h = manhattan(r, c, goal)
            log("  ({},{}) | h={:2d} | g={:2d} | path_len={}".format(r, c, h, g, len(path)))

        # Check if goal reached in current beam
        for (r, c), g, path in current_beam:
            if (r, c) == goal:
                log("")
                log("*** GOAL REACHED in beam ***")
                return path, g, log_lines

        # Generate successor states (Up, Down, Left, Right)
        log("")
        log("[GENERATING SUCCESSORS from current beam]:")
        candidates = []
        directions = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
        for (r, c), g, path in current_beam:
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
        current_beam = candidate_list[:k]
        for i, ((r, c), g, _) in enumerate(current_beam):
            h = manhattan(r, c, goal)
            log("  #{}: ({},{}) | h={:2d} | g={:2d}".format(i + 1, r, c, h, g))

        iteration += 1
    
    if iteration >= max_iter:                              # If the maximum number of iterations is reached without finding a solution, print a message indicating that the search has reached its limit and return None to indicate that no solution was found, which helps to prevent infinite loops in case of bugs or if the search space is too large to explore within a reasonable time frame.
        print("Max iterations reached - possible loop or no path.")
        pass
    return None, None


# ====================== MAIN ======================
# The main block reads the input file, executes the local beam search algorithm,
# displays and writes all required output, with full error handling.
if __name__ == "__main__":
    try:
        input_file = "input/inputPS11.txt"
        output_file = "output/outputPS11.txt"

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
            print("="*70)
            print(f"Final Path : {path_str}")
            print(f"Total Cost : {total_cost}")
            print("="*70)

            # Write the final path and total cost to the output file for documentation and further analysis.
            # This step ensures that the results of the local beam search are saved in a structured format for future reference or grading purposes.
            with open(output_file, "w") as f:
                f.write("=" * 70 + "\n")
                f.write("FINAL RESULT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Final Path from Start to Goal:\n{path_str}\n")
                f.write(f"Total Path Cost: {total_cost}\n")
                f.write("=" * 70 + "\n")
            print(f"\nOutput written to: {output_file}")
        else:
            print("No solution found.")
            pass

    except FileNotFoundError as e:
        print("\n ERROR OCCURRED!")
        print(f"Error: Input file not found - {e}")
        traceback.print_exc()
    except ValueError as e:
        print("\n ERROR OCCURRED!")
        print(f"Error: Invalid input format - {e}")
        traceback.print_exc()
    except Exception as e:
        print("\nUNEXPECTED ERROR: {}".format(e))
        traceback.print_exc()
