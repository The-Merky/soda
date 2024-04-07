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
    for i in 1..10 {
        n.forward(&DVector::from_element(3, 2.0));
        let loss = &n.loss(&DVector::from_element(2, 4.0));
        let grads_new = n.backward(loss);
        n.update_params(0.002, grads_new);
    }
    println!("{}", n.loss(&DVector::from_element(2, 4.0)).sum());
}
