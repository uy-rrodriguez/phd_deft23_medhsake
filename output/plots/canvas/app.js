// tSNE or UMAP data
// const ALG = "tsne";
const ALG = "umap";
const colX = `${ALG}0`
const colY = `${ALG}1`

const LABEL_COLOURS = {
    "very easy": "#FDE725",
    "easy": "#5EC962",
    "medium": "#21918C",
    "hard": "#3B528B",
    "very hard": "#440154",
}
const MARGIN = 92;  // Margin betwwen back and foreground
const STROKE = 10;

let canvas = document.getElementById("background");
let ctx = canvas.getContext("2d");
const width = canvas.width - STROKE * 2;
const height = canvas.height - STROKE * 2;

const foundWrapper = document.getElementById("found");
let lastFound = null; // To store last list of hovered items

// Get min and max X and Y to adjust the coordinates to the size of the canvas
var min_x, min_y, max_x, max_y;
for (let item of data) {
    let x = item[colX];
    let y = item[colY];
    if (min_x == null || x < min_x) min_x = x;
    if (min_y == null || y < min_y) min_y = y;
    if (max_x == null || x > max_x) max_x = x;
    if (max_y == null || y > max_y) max_y = y;
}

function getCoordinates(item) {
    const x = (item[colX] - min_x) * width / (max_x - min_x) + STROKE;
    const y = height - (item[colY] - min_y) * height / (max_y - min_y) + STROKE;
    return {"x": x, "y": y};
}

for (let item of data) {
    const coord = getCoordinates(item);
    ctx.fillStyle = LABEL_COLOURS[item.medshake_class];  // + "BB";
    ctx.beginPath();
    ctx.arc(coord.x, coord.y, STROKE, 0, 2 * Math.PI);
    ctx.fill();
}

// Legend
const legX = 10;
let legY = height - 14 * 5;
for (let cls in LABEL_COLOURS) {
    ctx.fillStyle = LABEL_COLOURS[cls];
    ctx.beginPath();
    ctx.arc(legX, legY, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#333333";
    ctx.fillText(cls, legX + 10, legY + 3);
    legY = legY + 20;
}


// Show details on hover
canvas = document.getElementById("foreground");
ctx = canvas.getContext("2d");
canvas.addEventListener("mousemove", function(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    // var mouse_x = event.clientX * 2, mouse_y = event.clientY * 2;

    let found = [];
    for (let item of data) {
        const coord = getCoordinates(item);
        const x = coord.x + MARGIN,
              y = coord.y + MARGIN;
        if (mouseX >= x - STROKE && mouseX <= x + STROKE
                && mouseY >= y - STROKE && mouseY <= y + STROKE) {
            found.push(item);

            // var x = mouse_x;
            // var y = mouse_y;
            // g.strokeStyle = 'blue';
            // g.beginPath();
            // g.arc(x, y, 10, 0, 2*Math.PI);
            // g.stroke();
            // g = fore.getContext('2d');
            // g.fillStyle = 'blue';
            // g.font = '30px Arial';
            // for(var word in selected) {
            //     argmin = selected[word];
            //     g.fillText(argmin, x, y);
            //     y += 30;
            // }
        }
    }
    if (found.length > 0) {
        ctx.fillStyle = "#DDDDDDCC";
        const PAD = 2;
        const LINE_H = 12;
        ctx.fillRect(
            mouseX + LINE_H - PAD, mouseY - LINE_H + PAD,
            180 + PAD, LINE_H * found.length + PAD)

        ctx.fillStyle = "#000000";
        found = found.reverse()
        for (let i = 0; i < found.length; i++) {
            const item = found[i];
            ctx.fillText(
                `ID: ${item.id.substring(0, 10)} | ${item.medshake_class}`,
                mouseX + LINE_H, mouseY + i * LINE_H);
        }
    }
    lastFound = found;
});


function findQuestion(id) {
    for (let q of questions) {
        if (q.id == id) {
            return q;
        }
    }
    return null;
}


// Popup details on double-click
canvas.addEventListener("dblclick", function(event) {
    if (lastFound) {
        const items_str = [];
        for (let item of lastFound) {
            const q = findQuestion(item.id);
            items_str.push(
                "<div>"
                + `<span style="background: ${item.medshake_colour}">&nbsp;</span>`
                + `<b>ID:</b> ${q.id}`
                + `<br><b>Class:</b> ${item.medshake_class}`
                + `<br><b>Topics:</b> ${q.topics}`
                + `<br><b>Year:</b> ${q.year_txt}`
                + `<br><b>Question:</b> ${q.question}`
                + "</div>"
            );
        }
        foundWrapper.innerHTML = items_str.join("");
    }
});
