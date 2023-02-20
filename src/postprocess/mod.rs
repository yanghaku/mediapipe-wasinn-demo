mod detection;
pub mod ops;

pub use detection::*;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Pointer2D<T: Debug + Clone> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Clone)]
pub struct Box2D<T: Debug + Clone> {
    pub p: Pointer2D<T>,
    pub w: T,
    pub h: T,
}

impl<T: Debug + Clone> Box2D<T> {
    pub fn from(slice: impl AsRef<[T]>, stride: usize) -> Self {
        Self {
            p: Pointer2D {
                x: slice.as_ref()[0].clone(),
                y: slice.as_ref()[1 * stride].clone(),
            },
            w: slice.as_ref()[2 * stride].clone(),
            h: slice.as_ref()[3 * stride].clone(),
        }
    }
}
