use std::ops::Mul;

pub mod neural_net;
pub mod layer;
fn main() {
    let layer = layer::Layer::new(2, layer::ActivationFunction::Sigmoid, 1);
    let layer2 = layer::Layer::new(2, layer::ActivationFunction::Sigmoid, 2);
    let net = neural_net::NeuralNet::new();
}
