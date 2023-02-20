use crate::postprocess::Box2D;
use image::{DynamicImage, GenericImage, Rgba};

pub fn draw_a_box(img: &mut DynamicImage, box2d: &Box2D<f32>) {
    let border_pixel = Rgba::from([255u8, 0u8, 0u8, 1u8]);

    let x_min = (box2d.p.x * img.width() as f32) as u32;
    let x_max = x_min + (box2d.w * img.width() as f32) as u32;
    let y_min = (box2d.p.y * img.height() as f32) as u32;
    let y_max = y_min + (box2d.h * img.height() as f32) as u32;
    for x in x_min..=x_max {
        img.put_pixel(x, y_min, border_pixel.clone());
        img.put_pixel(x, y_max, border_pixel.clone());
    }
    for y in y_min..=y_max {
        img.put_pixel(x_min, y, border_pixel.clone());
        img.put_pixel(x_max, y, border_pixel.clone());
    }
}
