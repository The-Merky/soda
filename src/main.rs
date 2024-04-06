use crate::layer::ActivationFunction;
use nalgebra::DVector;

pub mod layer;
pub mod neural_net;
fn main() {
    let mut n = neural_net::NeuralNet::new();
    n.add_layer(0, 3, ActivationFunction::Relu);
    n.add_layer(1, 12, ActivationFunction::Tanh);
    n.add_layer(2, 12, ActivationFunction::Sigmoid);
    n.add_layer(3, 12, ActivationFunction::Sigmoid);
    n.add_layer(4, 10, ActivationFunction::Softmax);
    println!("{:?}", n.backward(&DVector::from_element(10, 4.0)));
}
