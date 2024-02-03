pub mod layer;
fn main() {
    let layer = layer::Layer::new(2, layer::ActivationFunction::Sigmoid, 1);
}
