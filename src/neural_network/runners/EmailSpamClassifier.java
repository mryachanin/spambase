package neural_network.runners;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import neural_network.Data;
import neural_network.NeuralNetworkException;

/**
 * Main program for classifying email spam using
 * an artificial neural network.
 * 
 * Usage: java neural_network.runners.EmailSpamClassifier train training_data_filepath validation_data_filepath
 *        java neural_network.runners.EmailSpamClassifier test validation_data_filepath neural_network_save_filepath
 * 
 * @author Michael Yachanin (mry1294)
 */
public class EmailSpamClassifier {	
	/**
	 * Imports all data from a CSV file into an ArrayList.
	 * 
	 * @param filepath: The path of a CSV file.
	 * @return: An ArrayList containing the imported data.
	 * @throws NeuralNetworkException 
	 */
	private static ArrayList<Data> importCSVData(String filepath) throws NeuralNetworkException {
		ArrayList<Data> data = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(filepath ))) {
		    for (String line; (line = br.readLine()) != null; ) {
		        data.add(convertCSVToData(line));
		    }
		} catch (IOException e) {
			throw new NeuralNetworkException("There was an exception while importing CSV data.", e);
		}
		return data;
	}
	
	/**
	 * Converts a CSV datapoint to one parsable by this artificial neural network.
	 * 
	 * @param csvDatapoint: A string containing one line of CSV formatted data.
	 * @return: A Data object with the extracted data.
	 */
	private static Data convertCSVToData(String csvDatapoint) {
		String[] splitData = csvDatapoint.split(",");
		double[] features = new double[splitData.length - 1];
		
		// get features
		// last feature is classification. want < length - 1
		for (int feature = 0; feature < splitData.length - 1; feature++) {
			features[feature] = Double.parseDouble(splitData[feature]);
		}
		
		// get classification
		int[] classification = new int[] {
				Integer.parseInt(splitData[splitData.length - 1])
		};
		
		return new Data(features, classification);
	}
	
	/**
	 * Main Program.
	 */
	public static void main(String[] args) {
		if (args.length != 3) {
			usage();
		}
		
		try {
			ArrayList<Data> data = importCSVData(args[1]);
			switch (args[0].toLowerCase()) {
				case "test":
					new Tester(data).startTest(args[2], true);
					break;
				
				case "train":
					new Thread(new Trainer(data, importCSVData(args[2]))).start();
					break;
					
				default:
					usage();
			}
		} catch (NeuralNetworkException e) {
			System.err.println(e.getMessage());
			System.exit(1);
		}
	}
	
	/**
	 * Prints a usage message to stderr and exits. 
	 */
	private static void usage() {
		System.err.println("Usage: java neural_network.runners.EmailSpamClassifier train training_data_filepath validation_data_filepath");
		System.err.println("       java neural_network.runners.EmailSpamClassifier test validation_data_filepath neural_network_save_filepath");
		System.exit(1);
	}
}
