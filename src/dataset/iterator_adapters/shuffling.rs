use rand::seq::SliceRandom;
use rand::thread_rng;

pub trait Shuffling: Iterator {
    fn shuffling(self, max_elements_to_buffer: usize) -> Shuffler<Self>
    where
        Self: std::marker::Sized,
    {
        Shuffler {
            iterator: self,
            max_elements_to_buffer,
            buffer: vec![],
        }
    }
}

pub struct Shuffler<T: Iterator> {
    iterator: T,
    max_elements_to_buffer: usize,
    buffer: Vec<T::Item>,
}

impl<T: Iterator> Iterator for Shuffler<T> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.fill_buffer_if_needed_and_shuffle();
        self.buffer.pop()
    }
}

impl<T: Iterator> Shuffler<T> {
    fn fill_buffer_if_needed_and_shuffle(&mut self) {
        let mut added_item = false;
        while self.buffer.len() < self.max_elements_to_buffer {
            match self.iterator.next() {
                None => {
                    break;
                }
                Some(element) => {
                    self.buffer.push(element);
                    added_item = true;
                }
            }
        }
        if added_item {
            self.buffer.shuffle(&mut thread_rng());
        }
    }
}

impl<T: ?Sized> Shuffling for T where T: Iterator {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shuffle_test() {
        let elements = [0, 1, 2, 3, 4, 5];
        let a: Vec<_> = elements.iter().shuffling(50).collect();
        let _ = elements.iter().for_each(|e| assert!(a.contains(&e)));
    }
}
