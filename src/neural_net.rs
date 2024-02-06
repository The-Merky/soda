use core::panic;

use nalgebra::{DMatrix};

use crate::layer::{self, ActivationFunction, Layer};
// Neural Network struct which contains a vector of layers and can backpropagate and feed forward
// TODO backpropagate and feed forward
pub struct NeuralNet {
    layers: Vec<Layer>,
}
impl NeuralNet {
    pub fn new() -> NeuralNet {
        NeuralNet { layers: Vec::new() }
    }
    pub fn add_layer(&mut self, layer_number: usize, layer_size: usize, activation_function: ActivationFunction) {
        if layer_number == 0 {
            self.layers[1] = layer::Layer::new(layer_size, activation_function, layer_number, 1);      
        } else {
            self.layers.push(layer::Layer::new(layer_size, activation_function, layer_number, { if Some(&self.layers[layer_number-1]).is_some(){
                self.layers[layer_number-1].biases.ncols()
            }  else {
                panic!("Previous Layer is not set!");
            }}))
        }
        let mut last_activation = DMatrix::from_element(1, 1, 1);
        for i in &self.layers{
            
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
        activation.map(|x|   if x > 0.0 {
        x
    } else {
        0.0
    })
    }
fn softmax(activation: &DMatrix<f64>) -> DMatrix<f64>{
           let exp_values = activation.map(|x| x.exp());
    let sum_exp: f64 = exp_values.iter().sum();
    exp_values / sum_exp
    }
pub fn apply_activation_fn(layer: &mut Layer){
        match layer.activation_fn{
            layer::ActivationFunction::Tanh=> layer.activation_result = layer.activation_result.map(|x| x.tanh()),
            layer::ActivationFunction::Sigmoid => layer.activation_result = NeuralNet::sigmoid(&layer.activation_result),
            layer::ActivationFunction::Relu => layer.activation_result = NeuralNet::relu(&layer.activation_result),
            layer::ActivationFunction::Softmax => layer.activation_result = NeuralNet::softmax(&layer.activation_result),
        }
    }
}
