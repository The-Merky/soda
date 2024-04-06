use core::panic;

use nalgebra::{DMatrix, DVector};

use crate::layer::{ActivationFunction, Layer};

// Neural Network Struct which contains a vector of layers and can backpropagate and feed forward
// TODO backpropagation
pub struct NeuralNet {
    pub layers: Vec<Layer>,
}
impl Default for NeuralNet {
    fn default() -> Self {
        Self::new()
    }
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
                Layer::new(layer_size, activation_function, layer_number, 1),
            );
            self.verify_and_sort();
        } else {
            self.layers.push(Layer::new(
                layer_size,
                activation_function,
                layer_number,
                self.layers[layer_number - 1].activation_result.nrows(),
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
    fn sigmoid(activation: &DVector<f64>) -> DVector<f64> {
        activation.map(|x| 1.0 / (1.0 + (-x).exp()))
    }
    fn sigmoid_prime(activation: &DVector<f64>) -> DVector<f64> {
        activation.map(|x| 1.0 / (1.0 + (-x).exp()) * (1.0 - (1.0 / (1.0 + (-x).exp()))))
    }
    fn relu(activation: &DVector<f64>) -> DVector<f64> {
        activation.map(|x| if x > 0.0 { x } else { 0.0 })
    }
    fn softmax(activation: &DVector<f64>) -> DVector<f64> {
        let exp_values = activation.map(|x| x.exp());
        let sum_exp: f64 = exp_values.iter().sum();
        exp_values / sum_exp
    }
    pub fn apply_activation_fn(layer: &mut Layer) {
        match layer.activation_fn {
            ActivationFunction::Tanh => {
                layer.activation_result = layer.activation_result.map(|x| x.tanh());
            }
            ActivationFunction::Sigmoid => {
                layer.activation_result = NeuralNet::sigmoid(&layer.activation_result);
            }
            ActivationFunction::Relu => {
                layer.activation_result = NeuralNet::relu(&layer.activation_result);
            }
            ActivationFunction::Softmax => {
                layer.activation_result = NeuralNet::softmax(&layer.activation_result);
            }
        }
    }

    pub fn forward(&mut self, input: &DVector<f64>) {
        let mut previous_layer: Option<&Layer> = None;
        for current_layer in &mut self.layers {
            if previous_layer.is_some() {
                current_layer.activation_result = &current_layer.weights
                    * &previous_layer.unwrap().activation_result
                    + &current_layer.biases;
                NeuralNet::apply_activation_fn(current_layer);
            } else {
                current_layer.activation_result = (*input).clone();
            }
            previous_layer = Some(current_layer);
        }
    }
    pub fn backward(&mut self, expected: &DVector<f64>) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>) {
        let mut w_grad: Vec<DMatrix<f64>> = Vec::new(); //Vector of weights
        let mut b_grad: Vec<DVector<f64>> = Vec::new(); // Vector of biases
        let output_error: DVector<f64> = DVector::from_data(
            self.loss_prime(expected)
                .component_mul(&Self::sigmoid_prime(
                    &self.layers[self.layers.len() - 1].activation_result,
                ))
                .data,
        );
        // Push the gradients of the output layer
        b_grad.push(output_error.clone());
        w_grad.push(
            &output_error
                * &self.layers[self.layers.len() - 1]
                    .activation_result
                    .transpose(),
        );

        let mut delta: DVector<f64> = output_error;
        for layer in self.layers.iter().rev() {
            if layer.layer_number != self.layers.len() - 1 {
                delta = Self::sigmoid_prime(&layer.activation_result).component_mul(
                    &(&self.layers[layer.layer_number + 1].weights.transpose() * &delta),
                );
                b_grad.push(delta.clone());
                w_grad.push(
                    &delta
                        * &self.layers[layer.layer_number.saturating_sub(1)]
                            .activation_result
                            .transpose(),
                );
            }
        }
        (w_grad, b_grad)
    }
    pub fn loss(&mut self, expected: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            self.layers[self.layers.len() - 1].activation_result.nrows(),
            expected.nrows(),
            "Input vectors must have the same dimension."
        );
        // (Sum of all Result - Expected) ^2
        let diff = self.layers[self.layers.len() - 1].activation_result.clone() - expected;
        diff.map(|x| x.powi(2))
    }
    fn loss_prime(&mut self, expected: &DVector<f64>) -> DVector<f64> {
        assert_eq!(
            self.layers[self.layers.len() - 1].activation_result.nrows(),
            expected.nrows(),
            "Input vectors must have the same dimension."
        );
        let diff = self.layers[self.layers.len() - 1].activation_result.clone() - expected;
        diff.map(|x| x * 2.0)
    }
}
