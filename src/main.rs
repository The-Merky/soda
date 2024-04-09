use crate::layer::ActivationFunction;
use nalgebra::DVector;

pub mod layer;
pub mod neural_net;
fn main() {
    let mut n = neural_net::NeuralNet::new();
    n.add_layer(0, 3, ActivationFunction::Relu);
    n.add_layer(1, 2, ActivationFunction::Relu);
    n.forward(&DVector::from_element(3, 2.0));
    println!("{}", n.loss(&DVector::from_element(2, 4.0)).sum());
    for i in 1..10000 {
        n.forward(&DVector::from_element(3, 2.0));
        let loss = &n.loss(&DVector::from_element(2, 4.0));
        let grads_new = n.backward(loss);
        if i % 100 == 0 {
            println!("{}", n.loss(&DVector::from_element(2, 4.0)).sum());
        }
        n.update_params(0.0001, grads_new);
    }
    println!("{}", n.loss(&DVector::from_element(2, 4.0)).sum());
}
