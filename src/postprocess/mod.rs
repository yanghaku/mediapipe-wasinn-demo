mod detection;
pub mod draw_utils;
pub mod ops;
mod ssd_anchors_generator;

pub use detection::*;
pub use ssd_anchors_generator::*;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Pointer2D<T: Debug + Clone> {
    pub x: T,
    pub y: T,
}

impl Pointer2D<f32> {
    #[inline]
    pub fn transform_with_anchor(self, anchor: &Anchor, x_scale: f32, y_scale: f32) -> Self {
        Self {
            x: self.x / x_scale * anchor.w + anchor.x_center,
            y: self.y / y_scale * anchor.h + anchor.y_center,
        }
    }
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
