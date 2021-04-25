function derivative(x) {
    return x * (1-x);
}

class NeuralNetwork {
    constructor(input, hidden, output) {                    // ex. 2x6x4x5x2
        this.inputNum = input;
        this.hiddenNum = hidden instanceof Array ? hidden : [hidden];
        this.outputNum = output;
        this.wxh = new Matrix(this.inputNum, this.hiddenNum[0]);   // weights connecting input to hidden nodes    ex. 2x6
        this.why = new Matrix(this.hiddenNum[this.hiddenNum.length-1], this.outputNum);  // weights connecting the last hidden layer to output nodes   ex. 5x2
        this.hidden_weights = [];                                            // arr.[arr.length-1] returns the last item of an array
        this.hidden_bias = [];
        for(let i = 0; i < this.hiddenNum.length-1; i ++) {
          this.hidden_weights.push(new Matrix(this.hiddenNum[i], this.hiddenNum[i+1]));
          this.hidden_bias.push(new Matrix(1, this.hiddenNum[i]));
          this.hidden_bias[i].reset();
          print(this.hidden_weights[i]);
        }
        this.hidden_bias.push(new Matrix(1, this.hiddenNum[this.hiddenNum.length-1]));
        this.hidden_bias[this.hidden_bias.length-1].reset();  
      
        this.output_bias = new Matrix(1, this.outputNum);       // biases for the output nodes    ex. 1x2
        this.output_bias.reset();  
      
        this.hiddenLayer = new Array(this.hiddenNum.length);
      
        this.learningRate = 0.05;
    }

    classify(inputs) {
        this.x = Matrix.ConvertMatrixArray(inputs);     // creates a 1 x n matrix, n being the length of inputs array
        for(let i = 0; i < this.hiddenLayer.length; i ++) {
            if(i < 1) this.hiddenLayer[i] = Matrix.multiply(this.x, this.wxh);    // 1x2 x 2x6
            else  this.hiddenLayer[i] = Matrix.multiply(this.hiddenLayer[i-1], this.hidden_weights[i-1]);   // 1x6 x 6x4
            this.hiddenLayer[i].add(this.hidden_bias[i]);
            this.hiddenLayer[i].activate();
        }
        this.y = Matrix.multiply(this.hiddenLayer[this.hiddenLayer.length-1], this.why);     // output nodes (1 x outputNum) matrix
        this.y.add(this.output_bias);                                           // arr.[arr.length-1] returns the last item of an array
        this.y.activate();
        return Matrix.ConvertMatrixArray(this.y);       // converts output matrix into an array
    }

    train(inputs_array, answer) {
        const inputs = Matrix.ConvertMatrixArray(inputs_array);
        const outputs = Matrix.ConvertMatrixArray(this.classify(inputs_array));
        const hidden = this.hiddenLayer[this.hiddenLayer.length-1];
        
        const target = Matrix.ConvertMatrixArray(answer);
        const error = Matrix.add( target, Matrix.multiply(outputs, -1) );    // the error is equal to the difference between the target and the outputs  ex. 1x2
        this.cost = Matrix.map(error, (n) => { return n*n} ).sum();
        
        
      // Calculate how to change the weights between last hidden and output
        let output_gradients = Matrix.map(outputs, derivative);
        output_gradients.multiply(error);
        output_gradients.multiply(this.learningRate);
        const transposed_hidden = Matrix.transpose(hidden);
        const delta_why = Matrix.multiply(transposed_hidden, output_gradients);
        this.why.add(delta_why);
        this.output_bias.add(output_gradients);

        // the error of the hidden layer
        const why_t = Matrix.transpose(this.why);                           // ex. 3x2 --> 2x3

        const hidden_errors = new Array(this.hiddenLayer.length);
        hidden_errors[hidden_errors.length-1] = Matrix.multiply(error, why_t);                 // ex. 1x2 x 2x5
        for(let i = hidden_errors.length-2; i >= 0; i --) {               // Backpropogate through hidden layers
            let hidden_weights_t = Matrix.transpose(this.hidden_weights[i]);      // 4x5 --> 5x4
            hidden_errors[i] = Matrix.multiply(hidden_errors[i+1], hidden_weights_t);  // 1x5 x 5x4

            let gradient = Matrix.map(this.hiddenLayer[i+1], derivative);
            gradient.multiply(hidden_errors[i+1]);
            gradient.multiply(this.learningRate);
            const transposed_hidden = Matrix.transpose(this.hiddenLayer[i]);
            const delta_whh = Matrix.multiply(transposed_hidden, gradient);
            this.hidden_weights[i].add(delta_whh);
            this.hidden_bias[i+1].add(gradient);
        }
        
        // Calculate how to change the weights between the input and the first hidden
        let hidden_gradients = Matrix.map(this.hiddenLayer[0], derivative);
        hidden_gradients.multiply(hidden_errors[0]);
        hidden_gradients.multiply(this.learningRate);
        const transposed_input = Matrix.transpose(inputs);
        const delta_wxh = Matrix.multiply(transposed_input, hidden_gradients);  

        // adjust the weights and biases according the deltas which is just the gradient for the bias
        this.wxh.add(delta_wxh);
        this.hidden_bias[0].add(hidden_gradients);
        
    }
}