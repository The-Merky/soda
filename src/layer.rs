use nalgebra::{DMatrix, DVector};
use rand::Rng;
/*
Layer struct for neural networks
Contains a vector of weights and a vector of biasesstruct Layer<T: nalgebra::Dim>
*/
pub enum ActivationFunction {
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
}
pub struct Layer {
    weights: DMatrix<f64>,
    biases: DVector<f64>,
    activation_fn: ActivationFunction,
    //Result of activation function
    pub activation_result: DVector<f64>,
} 
impl Layer {
    pub fn new(size: usize, is_input: bool, activation_function: ActivationFunction) -> Layer {
        if (is_input) {
            Layer {
                weights: DMatrix::from_element(size, size, 1.0),
                biases: DVector::from_element(size, 1.0),
                activation_fn: activation_function,
                activation_result :DVector::from_element(size, 1.0),
            }
        } else {
            let mut rng = rand::thread_rng();
            Layer{
            //Random weights between 1 and -1
                weights: DMatrix::from_fn(size, size, |_, _| rng.gen_range(-1.0..1.0)),
                biases: DVector::from_element(size, 1.0),
                activation_fn: activation_function,
            activation_result: DVector::from_element(size, 1.0),
           }
        }
    }
}
