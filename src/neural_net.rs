use core::panic;


use nalgebra::DMatrix;

use crate::layer::{self, ActivationFunction, Layer};
// Neural Network struct which contains a vector of layers and can backpropagate and feed forward
// TODO backpropagate
pub struct NeuralNet {
    pub layers: Vec<Layer>,
}
impl NeuralNet {
    pub fn new() -> NeuralNet {
        NeuralNet { layers: Vec::new() }
    }
    pub fn add_layer(
        &mut self,
        layer_number: usize,
        layer_size: usize,
        activation_function: ActivationFunction,
    ) {
        if layer_number == 0 {
            self.layers.insert(
                0,
                layer::Layer::new(layer_size, activation_function, layer_number, 1),
            );
            self.verify_and_sort();
        } else {
            self.layers.push(layer::Layer::new(
                layer_size,
                activation_function,
                layer_number,
                self.layers[layer_number - 1].weights.ncols(),
            ));
            self.verify_and_sort();
        }
    }
    pub fn verify_and_sort(&mut self) {
        //Ensure that there are no duplicate layer numbers
        for i in 0..self.layers.len() - 1 {
            if self.layers[i].layer_number == self.layers[i + 1].layer_number {
                panic!("Duplicate layer numbers not allowed");
            }
        }
        //Sort layers
        self.layers.sort_by_key(|layer| layer.layer_number);
    }
    //Activation functions
    fn sigmoid(activation: &DMatrix<f64>) -> DMatrix<f64> {
        activation.map(|x| 1.0 / (1.0 + (-x).exp()))
    }
    fn relu(activation: &DMatrix<f64>) -> DMatrix<f64> {
        activation.map(|x| if x > 0.0 { x } else { 0.0 })
    }
    fn softmax(activation: &DMatrix<f64>) -> DMatrix<f64> {
        let exp_values = activation.map(|x| x.exp());
        let sum_exp: f64 = exp_values.iter().sum();
        exp_values / sum_exp
    }
    pub fn apply_activation_fn(layer: &mut Layer) {
        match layer.activation_fn {
            layer::ActivationFunction::Tanh => {
                layer.activation_result = layer.activation_result.map(|x| x.tanh());
            }
            layer::ActivationFunction::Sigmoid => {
                layer.activation_result = NeuralNet::sigmoid(&layer.activation_result);
            }
            layer::ActivationFunction::Relu => {
                layer.activation_result = NeuralNet::relu(&layer.activation_result);
            }
            layer::ActivationFunction::Softmax => {
                layer.activation_result = NeuralNet::softmax(&layer.activation_result);
            }
        }
    }

    pub fn forward(&mut self, input: &DMatrix<f64>) {
        let mut previous_layer: Option<&Layer> = None;
        for current_layer in &mut self.layers {
            if let Some(prev) = previous_layer {
                current_layer.activation_result =
                    current_layer.weights.clone() * prev.activation_result.clone();
                current_layer.activation_result += current_layer.biases.clone();
                NeuralNet::apply_activation_fn(current_layer);
            } else {
                current_layer.activation_result = (*input).clone();
            }
            previous_layer = Some(current_layer);
        }
    }
    pub fn loss(&mut self, expected: &DMatrix<f64>) ->f64{
    assert_eq!(self.layers[self.layers.len()-1].activation_result.nrows(), expected.nrows(), "Input vectors must have the same dimension.");

    let diff = self.layers[self.layers.len()-1].activation_result.clone() - expected.clone();
    let squared_diff = diff.map(|x| x.powi(2));
    let sum_squared_diff = squared_diff.sum();
    

    sum_squared_diff / expected.nrows() as f64
    }
}
