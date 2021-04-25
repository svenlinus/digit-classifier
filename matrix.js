
class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = [];
        
        for(let i = 0; i < this.rows; i ++) {
            this.data.push([]);
            for(let j = 0; j < this.cols; j ++) {
                this.data[i].push(random(-1, 1));
            }
        }
    }
    
    // matrix addition
    add(that) {
        if(that.rows != this.rows || that.cols != this.cols) {
            console.log("rows and cols must match (Addition)");
            return null;
        }
        for(let i = 0; i < this.rows; i ++) {
            for(let j = 0; j < this.cols; j ++) {
                this.data[i][j] += that.data[i][j];
            }
        }
    }

    static add(a, b) {
        if(b.rows != a.rows || b.cols != a.cols) {
            console.log("rows and cols must match (Addition)");
            return null;
        }
        let sum = new Matrix(a.rows, a.cols);
        for(let i = 0; i < sum.rows; i ++) {
            for(let j = 0; j < sum.cols; j ++) {
                sum.data[i][j] = a.data[i][j]+b.data[i][j];
            }
        }
        return sum;
    }

    // matrix multiplication
    multiply(that) {                                            // Hadamard product and scalar
        let product = new Matrix(this.rows, this.cols);
        if(that instanceof Matrix) {
            if(this.rows != that.rows || this.cols != that.cols) {
                console.log("rows and cols must match (Hadamard Product)");
                return null;
            }
            product = new Matrix(this.rows, that.cols);
            for(let i = 0; i < product.rows; i ++) {
                for(let j = 0; j < product.cols; j ++) {
                    product.data[i][j] = this.data[i][j]*that.data[i][j];
                }
            }
        } else {
            for(let i = 0; i < product.rows; i ++) {
                for(let j = 0; j < product.cols; j ++) {
                    product.data[i][j] = this.data[i][j]*that;
                }
            }
        }
        this.data = product.data;
    }
    static multiply(a, b) {                             // Matrix multiplication and scalar
        let product = new Matrix(a.rows, a.cols);
        if(b instanceof Matrix) {
            if(a.cols != b.rows) {
                console.log("cols must match rows (Matrix Multiplication)");
                return null;
            }
            product = new Matrix(a.rows, b.cols);
            for(let i = 0; i < product.rows; i ++) {
                for(let j = 0; j < product.cols; j ++) {
                    let sum = 0;
                    for(let k = 0; k < a.cols; k ++) {
                        sum += a.data[i][k]*b.data[k][j];
                    }
                    product.data[i][j] = sum;
                }
            }
        } else {
            for(let i = 0; i < product.rows; i ++) {
                for(let j = 0; j < product.cols; j ++) {
                    product.data[i][j] = a.data[i][j]*b;
                }
            }
        }
        return product;
    }

    activate() {            // the sigmoid function 1 / (1 + e^-x) will normalize the nodes to a value between 0 and 1
        for(let i = 0; i < this.rows; i ++) {
            for(let j = 0; j < this.cols; j ++) {
                this.data[i][j] = 1 / (1 + Math.pow( Math.E, -this.data[i][j]) );
            }
        }
    }

    static ConvertMatrixArray(data, type) {
        if(data instanceof Array && (type == null || type == "matrix")) {    // double checking if matrix is the desired result
            let mat = new Matrix(1, data.length);
            for(let j = 0; j < data.length; j ++) {
                mat.data[0][j] = data[j];
            }
            return mat;
        } if(data instanceof Matrix && (type == null || type == "array")) {    // double checking if array is the desired result
            let arr = [];
            for(let i = 0; i < data.rows; i ++) {
                for(let j = 0; j < data.cols; j ++) {
                    arr.push(data.data[i][j]);
                }
            }
            return arr;
        }
    }

    static transpose(mat) {
        let result = new Matrix(mat.cols, mat.rows);
        for(let i = 0; i < mat.rows; i ++) {
            for(let j = 0; j < mat.cols; j ++) {
                result.data[j][i] = mat.data[i][j];
            }
        }
        return result;
    }
    
    static map(mat, f) {
        let result = new Matrix(mat.rows, mat.cols);
        for(let i = 0; i < mat.rows; i ++) {
            for(let j = 0; j < mat.cols; j ++) {
                result.data[i][j] = f(mat.data[i][j]);
            }
        }
        return result;
    }
  
    reset() {
        for(let i = 0; i < this.rows; i ++) {
            for(let j = 0; j < this.cols; j ++) {
                this.data[i][j] = 0;
            }
        }
    }

    sum() {
        let sumOfData = 0;
        for(let i = 0; i < this.rows; i ++) {
            for(let j = 0; j < this.cols; j ++) {
                sumOfData += this.data[i][j];
            }
        }
        return sumOfData;
    }

    table() {
        console.table(this.data);
    };
}
