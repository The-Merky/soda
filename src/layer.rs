
/*
Layer struct for neural networks
Contains a vector of weights and a vector of biasesstruct Layer<T: nalgebra::Dim>
*/
struct Layer<T : nalgebra::DimName>

where
    T: nalgebra::DimName,
{
    weights: SVector<f32, T>,
    biases: SVector<f32, T>,
}

impl<T: nalgebra::DimName> Layer<T>
where
    T: nalgebra::DimName,
{
    fn new(weights: SVector<f32, T>, biases: SVector<f32, T>) -> Self { Self { weights, biases } }
} 

