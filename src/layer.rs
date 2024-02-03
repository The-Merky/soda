use core::num;

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
    layer_number: usize,
}
impl Layer {
    pub fn new(size: usize, activation_function: ActivationFunction, layer_number: usize) -> Layer {
        let mut rng = rand::thread_rng();
        Layer {
            //Random weights between 1 and -1
            weights: DMatrix::from_fn(size, size, |_, _| rng.gen_range(-1.0..1.0)),
            biases: DVector::from_element(size, 1.0),
            activation_fn: activation_function,
            activation_result: DVector::from_element(size, 1.0),
            layer_number: layer_number,
        }
    }
}
