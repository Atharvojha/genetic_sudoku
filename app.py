# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from ga_solver import genetic_algorithm

app = Flask(__name__, static_folder="../frontend", static_url_path="")

@app.route("/")
def index():
    # Serve the main HTML page located in the frontend folder.
    return send_from_directory("../frontend", "index.html")

@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    puzzle = data.get("puzzle")
    if not puzzle or len(puzzle) != 9:
        return jsonify({"error": "Invalid puzzle format. Expected a 9x9 grid."}), 400

    # Convert the puzzle to a numpy array.
    puzzle_arr = np.array(puzzle)
    # Run the Genetic Algorithm; lower generations for demo purposes.
    best_solution, progress = genetic_algorithm(puzzle_arr, generations=1000, verbose=False)
    # Return the solution as a list of lists.
    return jsonify({"solution": best_solution.tolist(), "progress": progress})

if __name__ == "__main__":
    app.run(debug=True)
