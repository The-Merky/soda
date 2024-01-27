use std::{clone, ops::{Add, Mul}};

#[derive (Clone)]
struct Value {
    data: f64,
    children: Vec<Value>,
}
impl Add for &Value {
    type Output = Value;
    fn add(self, other: Self) -> Self::Output {
        Value {
            data: (self.data + other.data),
            children: vec![self.clone(), other.clone()],
        }
    }
}
impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: Self) -> Self::Output {
        Value {
            data: (self.data * other.data),
            children: vec![self.clone(), other.clone()],
        }
    }
}
fn main(){

}