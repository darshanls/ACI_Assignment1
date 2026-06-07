import sys
import traceback

def read_input(file_path):
    """
    Reads the input grid file.
    Expected format:
    Line 1: rows cols
    Line 2: start_row start_col
    Line 3: goal_row goal_col
    Line 4 to end: grid rows (space separated)
    """
    print(f"Reading input file from path: {file_path}\n")
    
    with open(file_path, 'r') as f:   
        raw_lines = f.readlines()     # Read all lines as they are, including empty lines for debugging purposes
    
    # Show exactly what is in the file (including empty lines)
    print("=== RAW CONTENT OF INPUT FILE ===") 
    for i, line in enumerate(raw_lines):     # Debug print to show all lines, including empty ones
        print(f"Line {i}: '{line.strip()}'")
    print("================================\n")
    
    # Clean lines (remove empty)
    lines = [line.strip() for line in raw_lines if line.strip()] # Remove empty lines
    
    if len(lines) < 8:
        raise ValueError("Input file does not have enough lines!") # We expect at least 3 lines for metadata + 5 lines for grid (for a 5x5 grid)
    
    # Parse
    rows, cols = map(int, lines[0].split())  # First line: rows and columns
    start_row, start_col = map(int, lines[1].split())  # Second line: start position
    goal_row, goal_col = map(int, lines[2].split())  # Third line: goal position
    
    print("=== INPUT METADATA ===")
    print(f"Rows          : {rows}")
    print(f"Columns       : {cols}")
    print(f"Start Position: ({start_row}, {start_col})")
    print(f"Goal Position : ({goal_row}, {goal_col})")
    print("======================\n")
    
    # Read grid
    # We expect the grid to start from line index 3 and have 'rows' number of lines
    grid = []
    for i in range(3, 3 + rows):
        row = lines[i].split()
        grid.append(row)
    
    return rows, cols, (start_row, start_col), (goal_row, goal_col), grid # Return all parsed data for further processing


def get_cost(cell):            # Cost function: returns 3 for 'C' cells and 1 for normal cells
    return 3 if cell == 'C' else 1   # This function determines the traversal cost for a given cell type, which is used in the local beam search to calculate g(n) values for pathfinding.


def is_valid(r, c, rows, cols, grid):           # Check if the position (r, c) is within bounds and not an 'X' cell
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'X' # Returns True if the position is valid for traversal, False otherwise


def manhattan(r, c, goal):                      # Heuristic function: Manhattan distance from (r, c) to the goal position
    return abs(r - goal[0]) + abs(c - goal[1])  # Returns the sum of absolute differences in row and column indices to estimate distance to goal


def local_beam_search(rows, cols, start, goal, grid, k=2): 
    """
    Local Beam Search with k=2.
    At each iteration:
    - Expand all states in current beam (generate successors Up/Down/Left/Right)
    - Filter invalid (out of bounds or X)
    - Remove duplicate positions, keep the one with lowest g (traversal cost)
    - Compute h = Manhattan distance for each
    - Compute g = traversal cost (1 for normal, 3 for C high-cost cells)
    - Select top k states with smallest h (tie-break by smaller g)
    Displays all required details for execution flow.
    """
    print("\n" + "="*70)
    print("LOCAL BEAM SEARCH EXECUTION (k=2)")
    print("="*70)
    print("Heuristic: h(n) = Manhattan Distance = |x_g - x_n| + |y_g - y_n|")
    print("Cost: g(n) = path cost from start; normal cell enter cost=1, high-cost 'C' cell enter cost=3 (1+penalty2)")
    print("Selection: best k states by lowest h, then lowest g")
    print("="*70 + "\n")
    current_beam = [(start, 0, [start])]  # (pos, g_cost, path_list)
    iteration = 0
    max_iter = 50  # safety to prevent infinite if any bug

    while current_beam and iteration < max_iter:     # Main loop of the local beam search algorithm, which continues until a solution is found or the maximum number of iterations is reached to prevent infinite loops in case of bugs.
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        # Display current beam states with h and g
        print("\n[SELECTED BEAM STATES] (k=2 best so far):")
        beam_display = []
        for (r, c), g, path in current_beam:
            h = manhattan(r, c, goal)               # Calculate the heuristic value (h) for each state in the current beam to display the current state of the search and how close each state is to the goal.
            beam_display.append(f"  ({r},{c}) | h={h:2d} | g={g:2d} | path_len={len(path)}")  # Append the state information to a list for display, including the position, heuristic value, traversal cost, and path length.
        for line in beam_display:                  # Print the current beam states in a formatted manner to show the selected states for this iteration, their heuristic values, traversal costs, and path lengths for better understanding of the search process.
            print(line)
        
        # Check if goal reached in current beam
        for (r, c), g, path in current_beam:
            if (r, c) == goal:
                print("\n*** GOAL REACHED in beam ***")
                return path, g
        
        # Generate successor states
        print("\n[GENERATING SUCCESSORS from current beam]:")
        candidates = []
        for (r, c), g, path in current_beam:
            successors_for_state = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # Generate successor states by moving up, down, left, and right from the current state, and check if each successor is valid (not out of bounds or an 'X' cell) before calculating its cost and adding it to the list of candidates for the next beam.
                nr, nc = r + dr, c + dc                    # Calculate the new row and column indices for the successor state based on the current position and the direction of movement.
                if is_valid(nr, nc, rows, cols, grid):     # Check if the new position is valid for traversal (within bounds and not an 'X' cell) before proceeding to calculate its cost and add it to the list of candidates for the next iteration of the beam search.
                    cell = grid[nr][nc]                    # Get the type of cell at the new position to determine the cost of entering that cell, which will be used to calculate the new g(n) value for the successor state.
                    cost = get_cost(cell)                  # Calculate the cost of entering the new cell based on its type (1 for normal cells and 3 for 'C' high-cost cells) to update the g(n) value for the successor state, which represents the total traversal cost from the start position to this new state.   
                    new_g = g + cost                       # Update the g(n) value for the successor state by adding the cost of entering the new cell to the g(n) value of the current state, which represents the total traversal cost from the start position to this new state.
                    new_path = path + [(nr, nc)]           # Create a new path list for the successor state by appending the new position to the current path, which will be used to keep track of the path taken from the start position to this successor state.
                    candidates.append(((nr, nc), new_g, new_path))                                         # Add the valid successor state to the list of candidates for the next beam, including its position, updated g(n) value, and path taken to reach it.
                    successors_for_state.append(f"({nr},{nc})[cell={cell},cost+={cost},new_g={new_g}]")    # Append the successor state information to a list for display, including the new position, cell type, cost of entering that cell, and the updated g(n) value for better understanding of the successor generation process.
            if successors_for_state:
                print(f"  From ({r},{c}) g={g}: {', '.join(successors_for_state)}")                 # Print the generated successor states for the current state in the beam, showing the new positions, cell types, costs, and updated g(n) values for each valid successor to provide insight into the search process and how the algorithm is exploring the state space.
        
        if not candidates:                                                                          # If no valid successors are generated from the current beam states, print a message indicating that the search has failed and return None to indicate that no solution was found.
            print("No valid successors! Search failed.")
            return None, None
        
        print(f"\nTotal raw successors generated: {len(candidates)}")
        
        # Remove duplicates: keep best (lowest g) for each position
        print("\n[REMOVING DUPLICATES - keep lowest g per position]:")
        best = {}
        for pos, new_g, new_path in candidates:
            if pos not in best or new_g < best[pos][1]:  # Remove duplicate positions by keeping only the one with the lowest g(n) value for each unique position, which helps to optimize the search by ensuring that we only consider the best path to each position when selecting the next beam states.
                best[pos] = (pos, new_g, new_path)       # Update the best dictionary with the new candidate if it is better than the existing one for that position, ensuring that we maintain the best path to each unique position for the next iteration of the beam search.
        candidate_list = list(best.values())             # Convert the best dictionary values to a list for further processing, which will be used to sort and select the top k beam states for the next iteration of the search.
        print(f"  Unique positions after dedup: {len(candidate_list)}")
        for pos, g_val, _ in sorted(candidate_list, key=lambda x: x[1]): # Print the unique candidate positions after removing duplicates, along with their g(n) values, to show the remaining candidates that will be considered for selection in the next beam, providing insight into how the search is progressing and which states are being retained for further exploration.
            h_val = manhattan(pos[0], pos[1], goal)                      # Calculate the heuristic value (h) for each unique candidate position to display the remaining candidates after deduplication, showing their positions, heuristic values, and g(n) values to provide insight into the search process and how the algorithm is evaluating the remaining states for selection in the next beam.
            print(f"    ({pos[0]},{pos[1]}) | h={h_val:2d} | g={g_val:2d}")   # Print the unique candidate positions after deduplication, showing their positions, heuristic values, and g(n) values to provide insight into the search process and how the algorithm is evaluating the remaining states for selection in the next beam.
        
        # Sort by min heuristic (then min g for tie break)
        candidate_list.sort(key=lambda x: (manhattan(x[0][0], x[0][1], goal), x[1]))
        
        # Select best k
        print(f"\n[SELECTING TOP {k} BEAM STATES by min h then min g]:")
        current_beam = candidate_list[:k]                  # Select the top k candidate states based on the lowest heuristic value (h) and then by the lowest g(n) value for tie-breaking, which will form the new beam for the next iteration of the search, guiding the algorithm towards the most promising states that are closer to the goal while also considering the cost incurred to reach those states.
        for i, ((r, c), g, _) in enumerate(current_beam):  # Print the selected top k beam states for the next iteration, showing their positions, heuristic values, and g(n) values to provide insight into the search process and which states are being retained for further exploration in the next iteration of the beam search.
            h = manhattan(r, c, goal)                      # Calculate the heuristic value (h) for each selected beam state to display the selected states for the next iteration, showing their positions, heuristic values, and g(n) values to provide insight into the search process and how the algorithm is guiding the search towards the goal.
            print(f"  #{i+1}: ({r},{c}) | h={h:2d} | g={g:2d}")   # Print the selected top k beam states for the next iteration, showing their positions, heuristic values, and g(n) values to provide insight into the search process and which states are being retained for further exploration in the next iteration of the beam search.
        
        iteration += 1
    
    if iteration >= max_iter:                              # If the maximum number of iterations is reached without finding a solution, print a message indicating that the search has reached its limit and return None to indicate that no solution was found, which helps to prevent infinite loops in case of bugs or if the search space is too large to explore within a reasonable time frame.
        print("Max iterations reached - possible loop or no path.")
        pass
    return None, None


# ====================== MAIN ======================
# The main block of the code is responsible for reading the input file, executing the local beam search algorithm, and handling the output. 
# It includes error handling to catch and display any exceptions that may occur during execution.
if __name__ == "__main__":
    try:
        input_file = "input/inputPS11.txt"
        output_file = "output/outputPS11.txt"

        print("=" * 70)
        print("FILE PATHS USED:")
        print(f"Input File Path  -  {input_file}")
        print(f"Output File Path -  {output_file}")
        print("=" * 70 + "\n")

        rows, cols, start, goal, grid = read_input(input_file)   # Read the input file to obtain the grid dimensions, start and goal positions, and the grid layout. This function will parse the input file and return the necessary data for the local beam search algorithm to operate on.

        print(f"Grid Size: {rows} x {cols}\n")

        path, total_cost = local_beam_search(rows, cols, start, goal, grid, k=2)  # Execute the local beam search algorithm with a beam width of 2 to find a path from the start position to the goal position on the grid, while considering the costs of traversing different cell types and using the Manhattan distance as a heuristic for guiding the search towards the goal. The function returns the path taken and the total cost incurred if a solution is found, or None if no solution exists.

        # If a path is found, format the path as a string and print the final results, including the path and total cost.
        # Additionally, write the results to an output file for record-keeping.
        
        if path:
            path_str = " -> ".join([f"({r},{c})" for r, c in path])
            print("\n" + "="*70)
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
        print("\n ERROR OCCURRED!")
        print(f"Error: {e}")
        traceback.print_exc()