let data = [
    {
        inputs: [1, 1],
        outputs: [1, 0, 0]
    },
    {
        inputs: [1, 0],
        outputs: [1, 1, 0]
    },
    {
        inputs: [0, 1],
        outputs: [0, 1, 0]
    },
    {
        inputs: [0, 0],
        outputs: [0, 0, 1]
    },
    {
        inputs: [0.5, 0.5],
        outputs: [1, 1, 1]
    },
];
let xor;

function setup() {
    xor = new NeuralNetwork(2, 6, 3);
    createCanvas(400, 400);
    background(200);
    // for(let i = 0; i < 10000; i ++) {
    //     let d = random(data);
    //     xor.train(d.inputs, d.outputs);
    // }
}
function draw() {
    for(let i = 0; i < 100; i ++) {
        let d = random(data);
        xor.train(d.inputs, d.outputs);
    }
    let res = 20;
    for(let x = 0; x < width; x += res) {
      for(let y = 0; y < height; y += res) {
        let xpos = map(x, 0, width, 0, 1);
        let ypos = map(y, 0, height, 0, 1);
        
        noStroke();
        fill(xor.classify([xpos, ypos])[0]*255, xor.classify([xpos, ypos])[1]*255, xor.classify([xpos, ypos])[2]*255);
        rect(x, y, res, res)
      }
    }
}

function mouseClicked() {
    
    xor.train(data[1].inputs, data[1].outputs);
    print(xor.classify([1, 0]));
}