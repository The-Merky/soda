use std::array::from_mut;

use nalgebra::{DMatrix, Matrix};

pub mod layer;
pub mod neural_net;
fn main() {
    let mut n = neural_net::NeuralNet::new();
    n.add_layer(0, 3, layer::ActivationFunction::Tanh);
    n.add_layer(1, 3, layer::ActivationFunction::Tanh);
    n.add_layer(2, 3, layer::ActivationFunction::Softmax);
    n.forward(&nalgebra::DMatrix::from_element(3, 1, 4.0));
    println!("{}", n.layers[2].activation_result);
    n.forward(&nalgebra::DMatrix::from_element(3, 1, 4.0));
    println!("{}", n.layers[2].activation_result);
    println!("{}" , n.loss(nalgebra::DMatrix::from_element(3, 1, 4.0)));
}
