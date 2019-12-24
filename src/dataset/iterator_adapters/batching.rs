pub trait Batching: Iterator{
    fn batching(self, batch_size: usize) -> Batcher<Self> where Self: std::marker::Sized{
        Batcher{
            iterator: self,
            batch_size
        }
    }
}

pub struct Batcher<T: Iterator>{
    iterator: T,
    batch_size: usize,
}

impl<T: Iterator> Iterator for Batcher<T>{
    type Item = Vec<T::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut output = vec![];
        for _i in 0..self.batch_size{
            output.push(self.iterator.next()?);
        }
        Some(output)
    }
}

impl<T: ?Sized> Batching for T where T: Iterator { }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_test() {
        let batch_size = 2;
        let elements = [0, 1, 2, 3, 4, 5];
        let a = elements.iter().batching(batch_size).next().unwrap();
        assert_eq!(a.len(), batch_size)
    }
}
