const API_URL = "/predict";
const form = document.getElementById("prediction-form");
const result = document.getElementById("result");
const winner = document.getElementById("winner");
const probability = document.getElementById("probability");
const loading = document.getElementById("loading");

function autocomplete(input, url) {
    let xhr = new XMLHttpRequest();
    xhr.open("GET", url + "?query=" + input.value, true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            let suggestions = JSON.parse(xhr.responseText);
            let datalist = document.getElementById(input.getAttribute("list"));
            datalist.innerHTML = "";
            suggestions.forEach(function (item) {
                let option = document.createElement("option");
                option.value = item;
                datalist.appendChild(option);
            });
        }
    };
    xhr.send();
}

document.getElementById("fighter1").addEventListener("input", function () {
    autocomplete(this, "/autocomplete");
});

document.getElementById("fighter2").addEventListener("input", function () {
    autocomplete(this, "/autocomplete");
});

form.addEventListener("submit", async function (event) {
    event.preventDefault();


    let fighter1 = document.getElementById("fighter1").value;
    let fighter2 = document.getElementById("fighter2").value;

    if (fighter1 !== "" && fighter2 !== "") {
        // Sort fighter names alphabetically
        let fighters = [fighter1, fighter2].sort();

        // loading.style.display = "block"; // Show the loading element
        loading.classList.remove("hidden");

        // let xhr = new XMLHttpRequest();
        // xhr.open("POST", API_URL, true);
        // xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
        // xhr.onload = function () {
        //     loading.style.display = "none"; // Hide the loading element

        //     if (xhr.status === 200) {
        //         let resultData = JSON.parse(xhr.responseText);

        //         winner.innerText = "Predicted Winner: " + resultData.winner;
        //         probability.innerText = "Probability: " + (resultData.probability * 100).toFixed(2) + "%";
        //         result.style.display = "block";
        //     }
        // };
        // xhr.send("fighter1=" + encodeURIComponent(fighters[0]) + "&fighter2=" + encodeURIComponent(fighters[1]));
        
        let res = await fetch(API_URL, {
            mode: "cors",
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body: JSON.stringify({
                fighter1: fighters[0],
                fighter2: fighters[1],
            })
        })
        loading.classList.add("hidden");
        let data = await res.json();
        // console.log(data);
        
        result.classList.remove("hidden");
        winner.classList.remove("hidden");
        probability.classList.remove("hidden");

        winner.innerText = "Predicted Winner: " + data.winner;
        probability.innerText = "Probability: " + (data.probability * 100).toFixed(2) + "%";
    }
});
