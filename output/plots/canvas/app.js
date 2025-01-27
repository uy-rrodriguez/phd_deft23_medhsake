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

// Background canvas with plotted items
let background = document.getElementById("background");
let backCtx = background.getContext("2d");
const WIDTH = background.width - STROKE * 2;
const HEIGHT = background.height - STROKE * 2;

// Foreground canvas for selection and hover
const foreground = document.getElementById("foreground");
const foreCtx = foreground.getContext("2d");

// Canvas to highlight selected items
const selection = document.getElementById("selection");
const selCtx = selection.getContext("2d");

// Variables for item selection
const detailsWrapper = document.getElementById("details");
let lastSelected = null;
let multisel = false;
let selStartX = null;
let selStartY = null;
let selEndX = null;
let selEndY = null;


/* ************************************************************************** */
/* Functions                                                                  */
/* ************************************************************************** */

/**
 * Returns the item's coordinates in the canvas.
 *
 * @param {*} item Source item to get coordinates from
 * @returns Object with coordinates "x" and "y".
 */
function getCoordinates(item) {
    const x = (item[colX] - min_x) * WIDTH / (max_x - min_x) + STROKE;
    const y = HEIGHT - (item[colY] - min_y) * HEIGHT / (max_y - min_y) + STROKE;
    return {"x": x, "y": y};
}

/**
 */
/**
 * Draw a red border around selected items. Optionally highlight them.
 *
 * @param {array} items Items to mark as selected
 * @param {*} highlight Whether to highlight them in red
 */
function selectItems(items, highlight = false) {
    if (items && items.length > 0) {
        for (let item of items) {
            const coord = getCoordinates(item);
            if (highlight) {
                selCtx.strokeWidth = 5;
                selCtx.fillStyle = "rgb(200 20 20 / 50%)";
            }
            else {
                selCtx.strokeWidth = 2;
            }
            selCtx.strokeStyle = "#DD4444";
            selCtx.beginPath();
            selCtx.arc(coord.x + MARGIN, coord.y + MARGIN, STROKE, 0, 2 * Math.PI);
            if (highlight) {
                selCtx.fill();
            }
            selCtx.stroke();
        }
    }
}

/**
 * Highlight hovered items (red border and background).
 *
 * @param {array} items Items to highlight
 */
function highlightItems(items) {
    selectItems(items, highlight = true);
}

/**
 * Searches data for an item in the `questions.js` data source.
 *
 * @param {string} id Id of the item required
 * @returns Data found for the item with the given id.
 */
function findQuestion(id) {
    for (let q of questions) {
        if (q.id == id) {
            return q;
        }
    }
    return null;
}


/* ************************************************************************** */
/* Draw all items on load                                                     */
/* ************************************************************************** */

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

// Draw all items
for (let item of data) {
    const coord = getCoordinates(item);
    backCtx.fillStyle = LABEL_COLOURS[item.medshake_class];  // + "BB";
    backCtx.beginPath();
    backCtx.arc(coord.x, coord.y, STROKE, 0, 2 * Math.PI);
    backCtx.fill();
}

// Legend
const legX = 10;
let legY = HEIGHT - 14 * 5;
for (let cls in LABEL_COLOURS) {
    backCtx.fillStyle = LABEL_COLOURS[cls];
    backCtx.beginPath();
    backCtx.arc(legX, legY, 6, 0, 2 * Math.PI);
    backCtx.fill();
    backCtx.fillStyle = "#333333";
    backCtx.fillText(cls, legX + 10, legY + 3);
    legY = legY + 20;
}


/* ************************************************************************** */
/* Show popup on hover                                                        */
/* ************************************************************************** */

foreground.addEventListener("mousemove", function(event) {
    if (multisel) {
        return;
    }

    foreCtx.clearRect(0, 0, foreground.width, foreground.height);

    const rect = foreground.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    let found = [];
    for (let item of data) {
        const coord = getCoordinates(item);
        const x = coord.x + MARGIN,
              y = coord.y + MARGIN;
        if (mouseX >= x - STROKE && mouseX <= x + STROKE
                && mouseY >= y - STROKE && mouseY <= y + STROKE) {
            found.push(item);
        }
    }
    if (found.length > 0) {
        foreCtx.fillStyle = "#DDDDDDCC";
        const PAD = 2;
        const LINE_H = 12;
        foreCtx.fillRect(
            mouseX + LINE_H - PAD, mouseY - LINE_H + PAD,
            180 + PAD, LINE_H * found.length + PAD);

        foreCtx.fillStyle = "#000000";
        found = found.reverse()
        for (let i = 0; i < found.length; i++) {
            const item = found[i];
            foreCtx.fillText(
                `ID: ${item.id.substring(0, 10)} | ${item.medshake_class}`,
                mouseX + LINE_H, mouseY + i * LINE_H);
        }
    }

    // Highlight hovered items
    selCtx.clearRect(0, 0, selection.width, selection.height);
    selectItems(lastSelected);
    highlightItems(found);
});


/* ************************************************************************** */
/* Multi-item selection                                                       */
/* ************************************************************************** */

foreground.addEventListener("mousedown", function(event) {
    if (! multisel) {
        // console.debug("Multisel START")
        // Clear selection context
        selCtx.clearRect(0, 0, selection.width, selection.height);

        const rect = selection.getBoundingClientRect();
        multisel = true;
        selStartX = selEndX = event.clientX - rect.left;
        selStartY = selEndY = event.clientY - rect.top;
    }
});


foreground.addEventListener("mousemove", function(event) {
    if (multisel) {
        // console.debug("Multisel MOVING");
        foreCtx.clearRect(0, 0, foreground.width, foreground.height);

        const rect = selection.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        selEndX = mouseX;
        selEndY = mouseY;

        // Draw a rectangle in the selected area (foreground canvas)
        const minX = Math.min(selStartX, selEndX);
        const maxX = Math.max(selStartX, selEndX);
        const minY = Math.min(selStartY, selEndY);
        const maxY = Math.max(selStartY, selEndY);
        foreCtx.fillStyle = "#333333";
        selCtx.strokeWidth = 3;
        foreCtx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        foreCtx.fillStyle = "rgb(150 150 150 / 20%)";
        foreCtx.fillRect(minX, minY, maxX - minX, maxY - minY);
    }
});


foreground.addEventListener("mouseup", function(event) {
    if (multisel) {
        // console.debug("Multisel END");
        foreCtx.clearRect(0, 0, foreground.width, foreground.height);

        let found = [];
        for (let item of data) {
            const coord = getCoordinates(item);
            const x = coord.x + MARGIN,
                  y = coord.y + MARGIN;
            const minX = Math.min(selStartX, selEndX);
            const maxX = Math.max(selStartX, selEndX);
            const minY = Math.min(selStartY, selEndY);
            const maxY = Math.max(selStartY, selEndY);
            if (minX <= x - STROKE && maxX >= x + STROKE
                    && minY <= y - STROKE && maxY >= y + STROKE) {
                found.push(item);
            }
        }

        // Highlight selected items (selection canvas)
        selectItems(found);

        lastSelected = found;

        multisel = false;
        selStartX = null;
        selStartY = null;
        selEndX = null;
        selEndY = null;
    }
});


// Show details after a multi selection
foreground.addEventListener("mouseup", function(event) {
    if (lastSelected) {
        const items = [];

        // Sort found items by coordinates
        lastSelected.sort(function (a, b) {
            const ca = getCoordinates(a),
                  cb = getCoordinates(b);
            return ca.x - cb.x;  // + ca.y - cb.y;
        });

        for (let item of lastSelected) {
            const q = findQuestion(item.id);
            const div = document.createElement("div");
            div.id = item.id;
            div.innerHTML =
                `<span style="background: ${item.medshake_colour}">&nbsp;</span>`
                + `<b>ID:</b> ${q.id}`
                + `<br><b>Class:</b> ${item.medshake_class}`
                + `<br><b>Topics:</b> ${q.topics}`
                + `<br><b>Year:</b> ${q.year_txt}`
                + `<br><b>Question:</b> ${q.question}`
            ;
            items.push(div);

            // Highlight item when hovering over details on the right
            div.addEventListener("mouseover", function(event) {
                selCtx.clearRect(0, 0, selection.width, selection.height);
                selectItems(lastSelected);
                highlightItems([item]);
            });
            div.addEventListener("mouseout", function(event) {
                selCtx.clearRect(0, 0, selection.width, selection.height);
                selectItems(lastSelected);
            });
        }
        detailsWrapper.replaceChildren(...items);
    }
});
