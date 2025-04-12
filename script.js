// frontend/script.js
document.addEventListener("DOMContentLoaded", function() {
  // Dynamically build a 9x9 table for the sudoku puzzle input.
  const tableBody = document.querySelector("table tbody");
  for (let i = 0; i < 9; i++) {
    let row = document.createElement("tr");
    for (let j = 0; j < 9; j++) {
      let cell = document.createElement("td");
      let input = document.createElement("input");
      input.setAttribute("type", "number");
      input.setAttribute("min", "1");
      input.setAttribute("max", "9");
      input.setAttribute("id", `cell-${i}-${j}`);
      cell.appendChild(input);
      row.appendChild(cell);
    }
    tableBody.appendChild(row);
  }

  // Handle Solve button click.
  document.getElementById("solveBtn").addEventListener("click", function(){
    let puzzle = [];
    for(let i = 0; i < 9; i++){
      let row = [];
      for(let j = 0; j < 9; j++){
        let value = document.getElementById(`cell-${i}-${j}`).value;
        // Treat empty inputs as zeros (blanks).
        row.push(value === "" ? 0 : parseInt(value));
      }
      puzzle.push(row);
    }
    fetch("/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ puzzle: puzzle })
    })
    .then(response => response.json())
    .then(data => {
      if(data.error){
        alert(data.error);
      } else {
        displaySolution(data.solution);
        simulateProgress(data.progress);
      }
    });
  });

  // Handle Random Puzzle generation.
  document.getElementById("randomBtn").addEventListener("click", function(){
    // Create a puzzle with ~30 random cells filled.
    let cells = Array.from({ length: 9 }, () => Array(9).fill(0));
    let count = 0;
    while(count < 30){
      let i = Math.floor(Math.random() * 9);
      let j = Math.floor(Math.random() * 9);
      if(cells[i][j] === 0){
        cells[i][j] = Math.floor(Math.random() * 9) + 1;
        count++;
      }
    }
    // Populate the input fields.
    for(let i = 0; i < 9; i++){
      for(let j = 0; j < 9; j++){
        document.getElementById(`cell-${i}-${j}`).value = cells[i][j] === 0 ? "" : cells[i][j];
      }
    }
  });

  // Display the solution as a table.
  function displaySolution(solution){
    let solDiv = document.getElementById("solutionDiv");
    let solGrid = document.getElementById("solutionGrid");
    solGrid.innerHTML = "";
    let table = document.createElement("table");
    table.className = "table table-bordered text-center";
    for(let i = 0; i < 9; i++){
      let tr = document.createElement("tr");
      for(let j = 0; j < 9; j++){
        let td = document.createElement("td");
        td.style.width = "40px";
        td.innerText = solution[i][j];
        tr.appendChild(td);
      }
      table.appendChild(tr);
    }
    solGrid.appendChild(table);
    solDiv.style.display = "block";
  }

  // Simulate GA progress by incrementally filling a progress bar.
  function simulateProgress(progressArray){
    let progressDiv = document.getElementById("progressDiv");
    let progressBar = document.getElementById("progressBar");
    progressDiv.style.display = "block";
    let maxGen = progressArray.length;
    let currentGen = 0;
    let interval = setInterval(function(){
      if(currentGen < maxGen){
        let percent = Math.floor((currentGen / maxGen) * 100);
        progressBar.style.width = percent + "%";
        progressBar.innerText = percent + "%";
        currentGen++;
      } else {
        clearInterval(interval);
        progressBar.style.width = "100%";
        progressBar.innerText = "100%";
      }
    }, 50);
  }
});
