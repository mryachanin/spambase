package neural_network.runners;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import neural_network.Data;
import neural_network.HiddenLayer;
import neural_network.NeuralNetwork;
import neural_network.perceptron.Perceptron;


/**
 * Used to train this artificial neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class Trainer extends Runner {

	private final double LEARNING_RATE = .1;
	private final double GOAL_TEST_ERROR_RATE = .10;
	private final boolean BACKPROP_DEBUG = false;
	private final boolean TEST_DEBUG = false;
	private final boolean TRAIN_ERROR_DEBUG = false;
	private final boolean TEST_ERROR_DEBUG = false;
	private final Random rand = new Random();
	private final Scanner in = new Scanner(System.in);
	
	/**
	 * Construct a trainer for an artificial neural network.
	 */
	public Trainer(ArrayList<Data> data) {
		super(data);
		System.out.println("Would you like to use an existing neural net (y | n)?");
		if (in.nextLine().toLowerCase().equals("y")) {
			loadNeuralNetwork();
		}
		else {
			nnet = new NeuralNetwork(INPUT_COUNT, NUM_HIDDEN_LAYERS, NUM_HIDDEN_PERCEPTRONS, NUM_OUTPUT_PERCEPTRONS);
		}
	}
	
	
	@Override
	public void run() {
		// set aside some test data
		ArrayList<Data> testData = new ArrayList<>();
		for (int testDatapoint = 0; testDatapoint < data.size() / 10; testDatapoint++) {
			int randomDatapoint = rand.nextInt(data.size());
			testData.add(data.get(randomDatapoint));
			data.remove(randomDatapoint);
		}
		
		// start training
		double testingError, trainingError;
		do {
			trainingError = 0;
			testingError = 0;
			for (int iter = 0; iter < data.size(); iter++) {
				Data datapoint = data.get(rand.nextInt(data.size()));
	
				// input values to feed into neural network
				double[] inputs = datapoint.getInputs();
	
				// feed inputs through neural network to get prediction
				double[] predictedOutputs = nnet.run(datapoint);
				
				// what the classification should be
				int[] actualOutputs = datapoint.getClassification();
	
				if (BACKPROP_DEBUG) {
					System.out.print("Predicted Output (training): ");
					for (double i : predictedOutputs) {
						System.out.print(i + " ");
					}
					System.out.print("\nActual Output (training): ");
					for (int i : actualOutputs) {
						System.out.print(i + " ");
					}
					System.out.println();
				}
	
				// BACKPROPAGATION START
				HiddenLayer[] hiddenLayers = nnet.getHiddenLayers();
				
				Perceptron[] outputPerceptrons = nnet.getOutputPerceptrons();
				Perceptron[] perceptronsConnectedToOutputs = hiddenLayers[hiddenLayers.length - 1].getHiddenPerceptrons();
				double[] newOutWeights = new double[perceptronsConnectedToOutputs.length];
				double[] outputError = new double[perceptronsConnectedToOutputs.length];
				
				// update output weights
				for (int i = 0; i < nnet.NUM_OUTPUT_PERCEPTRONS; i++) {
					double outError = actualOutputs[i] - predictedOutputs[i];
					double outTransfer = predictedOutputs[i];
					double outModifiedError = outError * outTransfer * (1 - outTransfer);
					double[] outputWeights = outputPerceptrons[i].getWeights();
	
					// for each hidden neuron connected to this output neuron
					for (int j = 0; j < perceptronsConnectedToOutputs.length; j++) {
						newOutWeights[j] = LEARNING_RATE * outModifiedError * perceptronsConnectedToOutputs[j].getOutput();
						outputError[j] += outModifiedError * outputWeights[j];
					}
					outputPerceptrons[i].updateWeights(newOutWeights);
	
					trainingError += outError * outError;
				}
	
				for (int hiddenLayerIndex = hiddenLayers.length - 1; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
					HiddenLayer hiddenLayer = hiddenLayers[hiddenLayerIndex];
					int prevIndex = hiddenLayerIndex - 1;
					int numInputs = hiddenLayerIndex > 0 ? hiddenLayers[prevIndex].getNumPerceptrons() : nnet.NUM_INPUTS;
					double[] newHiddenWeights = new double[numInputs];
					
					// update hidden weights
					for (int j = 0; j < hiddenLayer.getNumPerceptrons(); j++) {
						double hiddenTransfer = hiddenLayer.getHiddenPerceptrons()[j].getOutput();
						double hiddenError = LEARNING_RATE * outputError[j] * hiddenTransfer * (1 - hiddenTransfer);
	
						// for each input connect to this hidden neuron
						for (int k=0; k < nnet.NUM_INPUTS; k++) {
							newHiddenWeights[k] = hiddenError * inputs[k];
						}
						hiddenLayer.getHiddenPerceptrons()[j].updateWeights(newHiddenWeights);
					}
				}
				
	
				if (TRAIN_ERROR_DEBUG) 
					System.out.println("Training error: " + trainingError);
				// BACKPROPAGATION END
			}
			
			// TEST NEURAL NETWORK START
			for (int testDatapointIndex = 0; testDatapointIndex < testData.size(); testDatapointIndex++) {
				Data testDatapoint = testData.get(testDatapointIndex);
				
				// get input values and feed them into the neural network
				double[] predictedTestOutputs = nnet.run(testDatapoint);
				
				// compute what the output should be
				int[] actualOutputs = testDatapoint.getClassification();
				
				if (TEST_DEBUG) {
					System.out.print("Predicted Output (testing): ");
					for (double i : predictedTestOutputs) {
						System.out.print(i + " ");
					}
					System.out.print("\nActual Output (testing): ");
					for (int i : actualOutputs) {
						System.out.print(i + " ");
					}
					System.out.println();
				}
				
				// compute testing error
				double singleTestError = 0;
				for (int i = 0; i < nnet.NUM_OUTPUT_PERCEPTRONS; i++) {
					singleTestError += (actualOutputs[i] - predictedTestOutputs[i]) * (actualOutputs[i] - predictedTestOutputs[i]);
				}
				singleTestError /= nnet.NUM_OUTPUT_PERCEPTRONS;
				testingError += singleTestError;
			}
			testingError /= testData.size();
			
			if (TEST_ERROR_DEBUG)
				System.out.println("Testing error: " + testingError);
			// TEST NEURAL NETWORK END
			
		} while (testingError > GOAL_TEST_ERROR_RATE);
		
		
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
}
