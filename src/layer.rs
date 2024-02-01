use nalgebra::{DVector, DMatrix};
/*
Layer struct for neural networks
Contains a vector of weights and a vector of biasesstruct Layer<T: nalgebra::Dim>
*/
struct Layer<T>
{
    weights: DMatrix<T>,
    biases: DVector<T>,
    activation: fn(T) -> T
}




