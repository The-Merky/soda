use nalgebra::DMatrix;

pub mod layer;
pub mod neural_net;
fn main() {
    let mut n = neural_net::NeuralNet::new();
    n.add_layer(0, 3, layer::ActivationFunction::Relu);
    n.add_layer(1, 3, layer::ActivationFunction::Relu);
    n.add_layer(2, 3, layer::ActivationFunction::Relu);
    n.forward(&DMatrix::from_element(3, 1, 84.2));
    println!("{}", n.layers[2].activation_result);
    println!("{}", n.loss(&DMatrix::from_element(3, 1, 0.5)));
    n.add_layer(3, 3, layer::ActivationFunction::Softmax);
    n.forward(&DMatrix::from_element(3, 1, 4.0));
    println!("{}", n.layers[2].activation_result);
    println!("{}", n.loss(&DMatrix::from_element(3, 1, 0.5)));
}
