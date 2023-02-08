use csv::Reader;
use linfa::prelude::*;
use linfa::Dataset;
use linfa_trees::DecisionTree;
use ndarray::{Array, Array1, Array2};
use clap::Parser;

#[derive(Parser)]
#[clap(version = "1.0", author = "Zuhang Xu", about = "Train a decision tree model")]
struct Cli {
    //set the path to the file to read
    file: std::path::PathBuf,
}

// build a read_csv function that take in a file path and reads a csv file in src and returns a Dataset
fn read_csv(path: &str) -> Dataset<f32, usize, ndarray::Dim<[usize; 1]>> {
    let mut reader = Reader::from_path(path).unwrap();
    let header: Vec<String> = reader
        .headers()
        .unwrap()
        .iter()
        .map(|header| header.to_string())
        .collect();
    let data = reader.deserialize().map(|r| r.unwrap()).collect();
    let target_index = header.len() - 1;
    let records = get_records(&data, target_index);
    let targets = get_targets(&data, target_index);
    let features = header[0..target_index].to_vec();
    return Dataset::new(records, targets).with_feature_names(features);
}

// get_records returns a 2D array of f32 that are the records of the csv file
fn get_records(data: &Vec<Vec<f32>>, target_index: usize) -> Array2<f32> {
    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
        records.extend_from_slice(&record[0..target_index]);
    }
    return Array::from(records)
        .into_shape((data.len(), target_index))
        .unwrap();
}

// get_targets returns a 1D array of usize that are the targets of the csv file
fn get_targets(data: &Vec<Vec<f32>>, target_index: usize) -> Array1<usize> {
    let mut targets: Vec<usize> = vec![];
    for record in data.iter() {
        targets.push(record[target_index] as usize);
    }
    return Array::from(targets);
}

// main function that parse arguments and reads the csv file, trains the model, and prints the predictions, targets and accuracy
fn main() {
    let args = Cli::parse();
    // split the data into training and testing sets
    let (train, test) = read_csv(&args.file.to_str().unwrap()).split_with_ratio(0.9);
    // Train the model
    let model = DecisionTree::params().fit(&train).unwrap();
    // Predict the test set, and calculate the accuracy
    let predictions = model.predict(&test);
    let accuracy = predictions.confusion_matrix(&test.targets).unwrap().accuracy();

    println!("Predictions:");
    println!("{:?}", predictions);

    println!("Targets:");
    println!("{:?}", test.targets);

    println!("Accuracy: {}", accuracy);
}
