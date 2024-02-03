use crate::layer::Layer;
// Neural Network struct which contains a vector of layers and can backpropagate and feed forward
// TODO backpropagate and feed forward
pub struct NeuralNet {
    layers: Vec<Layer>,
}
impl NeuralNet {
    pub fn new() -> NeuralNet {
        NeuralNet { layers: Vec::new() }
    }
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
        //Sort layers
        self.layers.sort_by_key(|layer| layer.layer_number);
        //Ensure that there are no duplicate layer numbers
        for i in 0..self.layers.len() - 1 {
            if self.layers[i].layer_number == self.layers[i + 1].layer_number {
                panic!("Duplicate layer numbers not allowed");
            }
        }
    }
}
