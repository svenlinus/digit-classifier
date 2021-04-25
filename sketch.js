
let mnist, network, trainImage, networkImg, nodeImg;
let imgIndex = 0;
let show = true;
let epochs = 0;
const maxEpoch = 5;
let inputs = [];

let costPoints = [];
let accuracyPoints = [];
let costs = 0;
let guess;
let correct = 0;
let incorrect = 0;

let displayedNum, displayedProb, displayedProbArr, tempLabel;

let colors;
let maximum = 0;

function setup() {
    textAlign(CENTER, CENTER);
    trainImage = createImage(28, 28);
    createCanvas(900, 400);
    background(220);
    networkImg = loadNetworkImage(64, 64, 10);
    network = new NeuralNetwork(784, 64, 10);

    loadMNIST(function(data) {
        mnist = data;
        console.log(mnist);
    });

    colors = [
        color(100, 100, 100),
        color(204, 2, 2),
        color(230, 146, 56),
        color(241, 194, 49),
        color(105, 168, 79),
        color(69, 130, 141),
        color(62, 132, 198),
        color(102, 78, 167),
        color(166, 77, 120),
        color(255, 255, 255),
    ];
}

function draw() {
    angleMode(DEGREES);
    textAlign(CENTER, CENTER);
    background(220);
    image(networkImg, 440, 3, 400, 400);
    if(nodeImg) image(nodeImg, 440, 0, 410, 400);
    strokeWeight(6);
    stroke(0);
    line(437, 0, 437, 400);
    strokeWeight(1);
    noStroke();
    fill(0);
    for(let i = 0; i < 10; i ++) {
        textSize(20);
        text(i, 870, i*30+68);
    }

    if(mnist && epochs < maxEpoch) {
        for(let i = 0; i < 50; i ++) {
            train(i);
            if(i%5 == 0) {costs += network.cost;}
        }
        
        if(frameCount % 10 == 0) {
            costPoints.push(costs/100);
            costs = 0;
            show = true;
            accuracyPoints.push(correct/(correct+incorrect));
            correct = 0;
            incorrect = 0;
        }
    }
    fill(255);
    stroke(0);
    rect(100, 200, 300, 200);
    beginShape();
    for(let i = 0; i < costPoints.length; i ++) {
        if(costPoints[i] > maximum) maximum = costPoints[i];
        noFill();
        stroke(255, 0, 0);
        const x = map(i, 0, costPoints.length-1, 100, 396);
        const y = map(costPoints[i], 0, maximum, 400, 200);
        vertex(x, y);
    }
    endShape();
    beginShape();
    for(let i = 0; i < accuracyPoints.length; i ++) {
        const x = map(i, 0, accuracyPoints.length-1, 100, 396);
        const y = map(accuracyPoints[i], 0, 1, 400, 200);
        stroke(0, 200, 0);
        vertex(x, y);
    }
    endShape();

    push();
    translate(0, 30);
    textSize(12);
    stroke(255);
    fill(0, 200, 0);
    rect(10, 235, 10, 10);
    text("ACCURACY ", 60, 240);
    fill(255, 0, 0);
    rect(10, 255, 10, 10);
    text("COST", 42, 260);
    pop();
    if(accuracyPoints.length > 0) {
        stroke(255);
        textSize(14);
        push();
        translate(396, map(accuracyPoints[accuracyPoints.length-1], 0, 1, 400, 200));
        fill(0, 200, 0);
        ellipse(0, 0, 8, 8);
        text(round(accuracyPoints[accuracyPoints.length-1]*100)+"%", -14, 14);
        pop();
        push();
        translate(396, map(costPoints[costPoints.length-1], 0, maximum, 400, 200));
        fill(255, 0, 0);
        ellipse(0, 0, 8, 8);
        text(round(costPoints[costPoints.length-1]*100)/100, -14, -14);
        pop();
    }

    //+

    if(costPoints.length == 50) {
        for(let i = 0; i < 10; i ++) {
            let avgCost = 0;
            for(let j = 0; j < 5; j ++) {
                avgCost += costPoints[j+i];
            }
            costPoints.splice(i, 5, avgCost);
        }
    }
    if(accuracyPoints.length == 50) {
        for(let i = 0; i < 10; i ++) {
            let avgAccuracy = 0;
            for(let j = 0; j < 5; j ++) {
                avgAccuracy += accuracyPoints[j+i];
            }
            avgAccuracy = avgAccuracy/5;
            accuracyPoints.splice(i, 5, avgAccuracy);
        }
    }
    

    image(trainImage, 200, 0, 200, 200);
    fill(0);
    noStroke();
    textSize(45);
    if(guess) {
        if(displayedProb) {
            pieChart(displayedProbArr, 100, 100, 80, displayedNum);
        }
    }
    
}


function train(i) {
    inputs = [];
    let outputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const label = mnist.train_labels[imgIndex];
    outputs[label] = 1;

    if(show) trainImage.loadPixels();

    for (let i = 0; i < 784; i++) {
        let bright = mnist.train_images[imgIndex][i];
        inputs.push(bright/255);
        if(show) {
            let index = i * 4;
            trainImage.pixels[index + 0] = bright;
            trainImage.pixels[index + 1] = bright;
            trainImage.pixels[index + 2] = bright;
            trainImage.pixels[index + 3] = 255;
        }
    }

    network.train(inputs, outputs);

    if(i % 5 == 0) {
        const outputMat = network.y;
        const outputArr = Matrix.ConvertMatrixArray(outputMat);
        let highestProb = 0;
        let predictedNumer;
        for(let i = 0; i < outputArr.length; i ++) {
            if(outputArr[i] > highestProb) {
                highestProb = outputArr[i];
                predictedNumer = i;
            }
        }
        guess = {number: predictedNumer, probability: outputArr[predictedNumer]/outputMat.sum(), probArr: outputArr};
        if(guess.number == label) correct ++;
        else incorrect ++;
    }
    
    if(show) {
        nodeImg = loadNodeImage(Matrix.ConvertMatrixArray(network.hiddenLayer[0]), Matrix.ConvertMatrixArray(network.y));
        trainImage.updatePixels();
        displayedNum = guess.number;
        displayedProb = guess.probability;
        displayedProbArr = guess.probArr;
    }

    show = false;
    imgIndex ++;
    if(imgIndex >= 60000) {
        imgIndex = 0;
        epochs ++;
    }
}


function loadNetworkImage(inputs, hidden, outputs) {
    const size = max(max(inputs, hidden), outputs);
    push();
    
    // line(10, 10, 10, 190);
    // fill(0);
    // text("784", 10, 200);
    // line(10, 210, 10, 400);
    for(let i = 0; i < inputs; i ++) {
        for(let j = 0; j < hidden; j ++) {
            const c = random() > 0.5 ? color(255, 100, 100) : color(100, 150, 255);
            stroke(red(c), green(c), blue(c), pow(random(1.32), 20));
            line(0, i*400/size, 200, j*400/size);
        }
    }
    for(let i = 0; i < hidden; i ++) {
        for(let j = 0; j < outputs; j ++) {
            const c = random() > 0.5 ? color(255, 100, 100) : color(100, 150, 255);
            stroke(red(c), green(c), blue(c), pow(random(2), 8));
            line(200, i*400/size, 400, j*30+65);
        }
    }
    pop();
    return get(0, 0, 400, 400);
};

function loadNodeImage(hidden, outputs) {
    for(let i = 0; i < hidden.length; i ++) {
        noStroke();
        fill(hidden[i]*255);
        ellipse(640, i*400/hidden.length+3, 400/hidden.length, 400/hidden.length);
    }
    for(let i = 0; i < outputs.length; i ++) {
        noStroke();
        fill(outputs[i]*255);
        ellipse(840, i*30+68, 20, 20);
    }
    return get(440, 0, 410, 400);
};


function pieChart(arr, x, y, r, greatest) {
    const total = sigma(arr);
    const len = arr.length;
    let angles = [0];
    for(let i = 0; i < len; i ++) {
        let d = r*2;
        if(i == greatest) {d = d*1.1;}
        noStroke();
        fill(colors[i]);
        angles.push(angles[i] + (arr[i]/total)*360);
        arc(x, y, d, d, angles[i], angles[i+1]);
        const theta = angles[i+1] - angles[i];
        if(theta > 30) {
            
            const a = angles[i] + theta/2;
            fill(bright(colors[i]));
            textSize(r/2);
            text(i, x+cos(a)*r*0.6, y+sin(a)*r*0.6);
        }
    }
}



function contrast(clr) {
    return color(255-red(clr), 255-green(clr), 255-blue(clr));
}

function bright(clr) {
    return red(clr)+blue(clr)+green(clr) <= 450 ? color(255) : color(0);
}

function sigma(arr) {
    let sum = 0;
    for(let i = 0; i < arr.length; i ++) {
        sum += arr[i];
    }
    return sum;
}