pub trait Sigmoid {
    fn sigmoid_inplace(&mut self);

    fn sigmoid(&self) -> Self;
}

impl Sigmoid for Vec<f32> {
    fn sigmoid_inplace(&mut self) {
        self.iter_mut()
            .for_each(|z| *z = 1f32 / (1f32 + (-(*z)).exp()));
    }

    fn sigmoid(&self) -> Self {
        todo!()
    }
}
