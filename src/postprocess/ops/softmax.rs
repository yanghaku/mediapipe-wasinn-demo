pub trait Softmax {
    fn softmax_inplace(&mut self);

    fn softmax(&self) -> Self;
}

impl Softmax for Vec<f32> {
    fn softmax_inplace(&mut self) {
        let mut sum = 0f32;
        for i in self.iter_mut() {
            *i = i.exp();
            sum += *i;
        }

        for i in self.iter_mut() {
            *i = *i / sum;
        }
    }

    fn softmax(&self) -> Self {
        todo!()
    }
}
