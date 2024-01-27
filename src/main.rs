use std::ops::{Add, Mul};
struct Value{
    data:f64,
    children: Vec<Value>
}
impl Add for Value{
    type Output = Value;
   fn add(self, other: Self) -> Self::Output {
       Value{data: (self.data+other.data), children: vec![other, self] }
   } 
} 
impl Mul for Value{
    type Output = Value;
   fn mul(self, other: Self) -> Self::Output {
       Value{data: (self.data*other.data), children: vec![other, self] }
   } 
} 
fn main(){

}
