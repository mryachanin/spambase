package neural_network.runners;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import neural_network.Data;

/**
 * Main program for classifying email spam using
 * an artificial neural network.
 * 
 * Usage: EmailSpamClassifier < train | test >
 * 
 * @author Michael Yachanin (mry1294)
 */
public class EmailSpamClassifier {	
	/**
	 * Imports all data from a CSV file into an ArrayList.
	 * 
	 * @param filepath: The path of a CSV file.
	 * @return: An ArrayList containing the imported data.
	 */
	private static ArrayList<Data> importCSVData(String filepath) {
		ArrayList<Data> data = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(filepath ))) {
		    for (String line; (line = br.readLine()) != null; ) {
		        data.add(convertCSVToData(line));
		    }
		} catch (IOException e) {
			errorAndExit("There was an exception while importing CSV data.");
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
	 * 
	 * @param args: args[0] = "test" or "train" depending on use case.
	 *              args[1] = filepath of CSV data to either test or train with.
	 */
	public static void main(String[] args) {
		if (args.length != 2) {
			usage();
		}
		
		ArrayList<Data> data = importCSVData(args[1]);
		switch (args[0].toLowerCase()) {
			case "test":
				new Thread(new Tester(data)).start();
				break;
			
			case "train":
				new Thread(new Trainer(data)).start();
				break;
				
			default:
				usage();
		}
	}
	
	/**
	 * Prints a usage message to stdout and exits. 
	 */
	private static void usage() {
		errorAndExit("Usage: EmailSpamClassifier <test|train> <filepath to data>");
	}
	
	/**
	 * Prints an error message to stdout and exits.
	 * 
	 * @param error: The error message to print.
	 */
	private static void errorAndExit(String error) {
		System.out.println(error);
		System.exit(1);
	}
}
