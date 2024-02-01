use nalgebra::{DMatrix, DVector};
/*
Layer struct for neural networks
Contains a vector of weights and a vector of biasesstruct Layer<T: nalgebra::Dim>
*/
enum ActivationFunction {
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
}
struct Layer {
    weights: DMatrix<f64>,
    biases: DVector<f64>,
    activation_fn: ActivationFunction,
}
impl Layer {
    fn new(size: usize, is_input: bool, activation_function: ActivationFunction) -> Layer {
        if (is_input) {
            Layer {
                weights: DMatrix::from_element(size, size, 1.0),
                biases: DVector::from_element(size, 1.0),
                activation_fn: activation_function,
            }
        } else {
            //TODO: Create Layer with random weights and biases
           Layer{
                weights: DMatrix::from_element(size, size, 1.0),
                biases: DVector::from_element(size, 1.0),
                activation_fn: activation_function,
           }
        }
    }
}
