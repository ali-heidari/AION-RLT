use std::fmt::{Debug, Display};

use log::info;

pub struct Footstep {
    text: String,
    precision: usize,
}

impl Footstep {
    pub fn new(precision: usize) -> Self {
        Self {
            text: String::new(),
            precision: precision,
        }
    }

    pub fn add_title(&mut self, title: &str) {
        self.text.push_str(&format!("\n[{}]", title));
    }

    pub fn add_title_with_value<T>(&mut self, title: &str, value: &[T])
    where
        T: Debug,
    {
        self.text.push_str(&format!("\n[{}]\t{:?}", title, value));
    }

    fn create_format<T>(&self, label: &str, value: T, precision: usize) -> String
    where
        T: Display,
    {
        format!("\t{}: {:.prec$}", label, value, prec = precision)
    }

    pub fn add_value_with_precision<T>(&mut self, label: &str, value: T, precision: usize)
    where
        T: Display,
    {
        self.text
            .push_str(&self.create_format(label, value, precision));
    }

    pub fn add_value<T>(&mut self, label: &str, value: T)
    where
        T: Display,
    {
        self.text
            .push_str(&self.create_format(label, value, self.precision));
    }

    pub fn add_values<T>(&mut self, values: Vec<(&str, T)>)
    where
        T: Display,
    {
        for value in values {
            self.add_value(value.0, value.1);
        }
    }

    pub fn print(&self) -> String {
        info!(
            "{}\n-----------------------------------------------------------------",
            self.text
        );
        self.text.clone()
    }
}
