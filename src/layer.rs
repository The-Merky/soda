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
    pub biases: DVector<f64>,
    pub activation_fn: ActivationFunction,
    //Result of activation function
    pub activation_result: DMatrix<f64>,
    pub layer_number: usize,
}
impl Layer {
    pub fn new(size: usize, activation_function: ActivationFunction, layer_number: usize, weights:usize) -> Layer {
        let mut rng = rand::thread_rng();
        let weights_field = {
            if weights >1{
                DMatrix::from_fn(size, weights, |_, _| rng.gen_range(-1.0..1.0))
            } else {
                DMatrix::from_element(size, size, 1.0)
            } 

        };
        Layer {
            //Random weights between 1 and -1
            weights: weights_field,
            biases: DVector::from_element(size, 1.0),
            activation_fn: activation_function,
            activation_result: DMatrix::from_element(size, size, 1.0),
            layer_number,
        }
    }
    

    


}
