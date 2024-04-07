use crate::layer::ActivationFunction;
use nalgebra::DVector;

pub mod layer;
pub mod neural_net;
fn main() {
    let mut n = neural_net::NeuralNet::new();
    n.add_layer(0, 3, ActivationFunction::Relu);
    n.add_layer(1, 5, ActivationFunction::Tanh);
    n.add_layer(2, 2, ActivationFunction::Softmax);
    println!("{}", n.loss(&DVector::from_element(2, 4.0)).sum());
 }
    println!("{}", n.loss(&DVector::from_element(2, 4.0)).sum());
}
