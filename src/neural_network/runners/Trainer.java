package neural_network.runners;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import neural_network.NeuralNetwork;
import neural_network.perceptron.Perceptron;


/**
 * Used to train this artificial neural network.
 * 
 * @author Mike Yachanin (mry1294)
 */
public class Trainer extends Runner {

	private final double LEARNING_RATE = .1;
	private final boolean BACKPROP_DEBUG = false;
	private final boolean TEST_DEBUG = false;
	private final Random rand = new Random();
	
	/**
	 * Construct a trainer for an artificial neural network.
	 */
	public Trainer() {
		System.out.println("Would you like to use an existing neural net (y | n)?");
		String ans = in.nextLine();
		if (ans.toLowerCase().equals("y")) {
			loadNeuralNetwork();
		}
		else {
			nnet = new NeuralNetwork(INPUT_COUNT, NUM_HIDDEN_LAYERS, NUM_HIDDEN_PERCEPTRONS, NUM_OUTPUT_PERCEPTRONS);
		}
	}
	
	
	@Override
	public void run() {
		// get English training data filename
		System.out.println("Enter input file with English training data.");
		String englishFilename = in.nextLine();
		//String englishFilename = "tests/e";

		// get Italian training data filename
		System.out.println("Enter input file with Italian training data.");
		String italianFilename = in.nextLine();
		//String italianFilename = "tests/i";
		
		// get Dutch training data filename
		System.out.println("Enter input file with Dutch training data.");
		String dutchFilename = in.nextLine();
		//String dutchFilename = "tests/d";
		
		// get testing data filename
		System.out.println("Enter input file with testing data.");
		String testFilename = in.nextLine();
		//String testFilename = "tests/wikiGeneralTests";
		
		Scanner fin = null;

		// read in English training data
		ArrayList<String> englishLines = new ArrayList<String>();
		try {
			fin = new Scanner(new File(englishFilename));
			while (fin.hasNextLine()) {
				englishLines.add(fin.nextLine());
			}
		} catch (FileNotFoundException e) { fileNotFound(englishFilename); }

		// read in Italian training data
		ArrayList<String> italianLines = new ArrayList<String>();
		try {
			fin = new Scanner(new File(italianFilename));
			while (fin.hasNextLine()) {
				italianLines.add(fin.nextLine());
			}
		} catch (FileNotFoundException e) { fileNotFound(italianFilename); }

		// read in Dutch training data
		ArrayList<String> dutchLines = new ArrayList<String>();
		try {
			fin = new Scanner(new File(dutchFilename));
			while (fin.hasNextLine()) {
				dutchLines.add(fin.nextLine());
			}
		} catch (FileNotFoundException e) { fileNotFound(dutchFilename); }
		
		// read in testing data
		ArrayList<String> testLines = new ArrayList<String>();
		try {
			fin = new Scanner(new File(testFilename));
			while (fin.hasNextLine()) {
				testLines.add(fin.nextLine());
			}
		} catch (FileNotFoundException e) { fileNotFound(testFilename); }

		fin.close();


		// start training
		double testingError;
		double trainingError;
		do {
			String language = null;
			String inputText = null;

			// choose language and input text to use at random
			switch (rand.nextInt(3)) {
				// use English
				case 0:
					language = "english";
					inputText = englishLines.get(rand.nextInt(englishLines.size()));
					break;
	
				// use Italian
				case 1:
					language = "italian";
					inputText = italianLines.get(rand.nextInt(italianLines.size()));
					break;
	
				// use Dutch
				case 2:
					language = "dutch";
					inputText = dutchLines.get(rand.nextInt(dutchLines.size()));
					break;
	
				// shouldn't happen
				default:
					throw new IndexOutOfBoundsException("Something weird happened in the training language selection switch statement.");
			}
		

			// get input values to feed into neural network
			double[] inputs = getInputs(inputText);

			// feed inputs through neural network
			double[] predictedOutputs = nnet.run(inputs);
			
			// compute what the output should be
			int[] actualOutput = getLanguageVector(language);
			

			if (BACKPROP_DEBUG) {
				System.out.print("Predicted Output (training): ");
				for (double i : predictedOutputs) {
					System.out.print(i + " ");
				}
				System.out.print("\nActual Output (training): ");
				for (int i : actualOutput) {
					System.out.print(i + " ");
				}
				System.out.println();
			}

			
			// BACKPROPAGATION START
			double[] newOutWeights = new double[nnet.NUM_HIDDEN_NEURONS];
			double[] newHiddenWeights = new double[nnet.NUM_INPUTS];
			double[] error = new double[nnet.NUM_HIDDEN_NEURONS];

			Perceptron[] outputNeurons = nnet.getOutputNeurons();
			Perceptron[] hiddenNeurons = nnet.getHiddenNeurons();

			trainingError = 0;
			testingError = 0;

			// update output weights
			for (int i=0; i < nnet.NUM_OUTPUT_NEURONS; i++) {
				double outError = actualOutput[i] - predictedOutputs[i];
				double outTransfer = outputNeurons[i].getOutput();
				double outModifiedError = outError * outTransfer * (1 - outTransfer);
				double[] thisOutputsWeights = outputNeurons[i].getWeights();

				// for each hidden neuron connected to this output neuron
				for (int j=0; j < nnet.NUM_HIDDEN_NEURONS; j++) {
					newOutWeights[j] = LEARNING_RATE * outModifiedError * hiddenNeurons[j].getOutput();
					error[j] += outModifiedError * thisOutputsWeights[j];
				}
				outputNeurons[i].updateWeights(newOutWeights);

				trainingError += outError * outError;
			}

			// update hidden weights
			for (int j=0; j < nnet.NUM_HIDDEN_NEURONS; j++) {
				double hiddenTransfer = hiddenNeurons[j].getOutput();
				double hiddenError = LEARNING_RATE * error[j] * hiddenTransfer * (1 - hiddenTransfer);

				// for each input connect to this hidden neuron
				for (int k=0; k < nnet.NUM_INPUTS; k++) {
					newHiddenWeights[k] = hiddenError * inputs[k];
				}
				hiddenNeurons[j].updateWeights(newHiddenWeights);
			}

			if (BACKPROP_DEBUG) 
				System.out.println("Training error: " + trainingError);

			// BACKPROPAGATION END
			
			
			// TEST NEURAL NETWORK START
			for (int l=0; l < testLines.size(); l+=2) {
				// get input values and feed them into the neural network
				double[] predictedTestOutputs = nnet.run(getInputs(testLines.get(l)));
				
				// compute what the output should be
				actualOutput = getLanguageVector(testLines.get(l+1));
				
				if (TEST_DEBUG) {
					System.out.print("Predicted Output (testing): ");
					for (double i : predictedOutputs) {
						System.out.print(i + " ");
					}
					System.out.print("\nActual Output (testing): ");
					for (int i : actualOutput) {
						System.out.print(i + " ");
					}
					System.out.println();
				}
				
				// compute testing error
				double singleTestError = 0;
				for (int i=0; i < nnet.NUM_OUTPUT_NEURONS; i++) {
					singleTestError += (actualOutput[i] - predictedTestOutputs[i]) * (actualOutput[i] - predictedTestOutputs[i]);
				}
				singleTestError /= nnet.NUM_OUTPUT_NEURONS;
				testingError += singleTestError;
			}
			testingError /= testLines.size() / 2;
			
			if (TEST_DEBUG)
				System.out.println(testingError);
			
			// TEST NEURAL NETWORK END
		} while (testingError > .05);
		
		
		// prompt to save neural net for future use
		System.out.println("Would you like to save the neural net for future use (y | n)?");
		if (in.nextLine().toLowerCase().equals("y")) {
			saveNeuralNetwork();
		}
	}
	
	
	/**
	 * Saves the current neural network to a file - nnet.save.
	 */
	private void saveNeuralNetwork() {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("nnet.save"));
			oos.writeObject(nnet);
			oos.close();
		} catch (IOException e) {
			System.out.println("Error while saving neural network to a file.");
			System.exit(1);
		}
	}
	
	/**
	 * Transforms the string representation of a language into a vector.
	 * 
	 * @param language: string representation of a language
	 * @return: vector representation of the language passed in
	 */
	private int[] getLanguageVector(String language) {
		switch (language) {
			case "english":
				return new int[] { 1, 0, 0 };
	
			case "italian":
				return new int[] { 0, 1, 0 };
	
			case "dutch":
				return new int[] { 0, 0, 1 };
			
			default:
				throw new IllegalArgumentException("Something weird happened in the training output computation switch statement.");
		}
	}
	
	
	/**
	 * Prints an error usage message regarding the file not found to stdout and exits. 
	 */
	private void fileNotFound(String filename) {
		System.out.println("File not found: " + filename);
		System.exit(1);
	}
}
