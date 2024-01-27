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
impl Value {
    fn new(data: f64) -> Value {
        Value {
            data: data,
            children: vec![],
        }
    }
}
fn main() {
    let a = Value::new(5.0);
    let b = Value::new(8.0);
    let c = &a + &b;
    let d = &a * &b;
    println!("{}", c.data);
    println!("{}", d.data);
}
