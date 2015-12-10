package neural_network.runners;

import java.util.ArrayList;

import neural_network.Data;
import neural_network.HiddenLayer;
import neural_network.NeuralNetwork;
import neural_network.NeuralNetworkException;
import neural_network.Perceptron;


/**
 * Used to train this artificial neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class Trainer extends Runner implements Runnable {

	private final double LEARNING_RATE = .1;
	private final double GOAL_TEST_ERROR_RATE = .04;
	private final boolean TEST_ERROR_DEBUG = false;
	private final Tester tester;
	
	/**
	 * Construct a trainer for an artificial neural network.
	 */
	public Trainer(ArrayList<Data> data) {
		super(data);
		tester = new Tester(data);
	}
	
	/**
	 * Start generating and training neural networks.
	 */
	public void run() {
		while (true) {
			// create a new neural net with random weights
			// loops a bunch of times (infinite) in case poor weights are selected
			nnet = new NeuralNetwork(INPUT_COUNT, NUM_HIDDEN_LAYERS, NUM_HIDDEN_PERCEPTRONS, NUM_OUTPUT_PERCEPTRONS);
			
			// allocate 10 folds for 10-fold cross validation
			ArrayList<ArrayList<Data>> folds = new ArrayList<ArrayList<Data>>();
			for (int foldNum = 0; foldNum < 10; foldNum++) {
				ArrayList<Data> fold = new ArrayList<>();
				for (int datapoint = foldNum; datapoint < data.size(); datapoint += 10) {
					fold.add(data.get(datapoint));
				}
				folds.add(fold);
			}
			
			// train on each fold.
			for (int foldNum = 0; foldNum < 10; foldNum++) {
				trainFold(foldNum, folds);
				nnet.reset();
			}
		}
	}
	
	/**
	 * Trains a neural network with given folds of data, keeping one separate to test with.
	 * 
	 * @param testFoldIndex : The index of the fold to use as test data.
	 * @param folds : All folds to use to train/test a neural network.
	 */
	public void trainFold(int testFoldIndex, ArrayList<ArrayList<Data>> folds) {
		double testingError;
		double lowestTestingError = Double.POSITIVE_INFINITY;
		int iteration = 0;
		do {
			// TRAIN NEURAL NETWORK START
			for (int foldIndex = 0; foldIndex < 10; foldIndex++) {
				// iterate over all folds except fold containing test data
				if (foldIndex == testFoldIndex) {
					continue;
				}
				ArrayList<Data> fold = folds.get(foldIndex);
				for (int datapointIndex = 0; datapointIndex < fold.size(); datapointIndex++) {
					Data datapoint = fold.get(datapointIndex);
		
					// input values to feed into neural network
					double[] inputs = datapoint.getInputs();
		
					// feed inputs through neural network to get prediction
					double[] predictedOutputs = nnet.run(datapoint);
					
					// what the classification should be
					int[] expectedOutputs = datapoint.getClassification();
		
					// BACKPROPAGATION START
					HiddenLayer[] hiddenLayers = nnet.getHiddenLayers();
					Perceptron[] outputPerceptrons = nnet.getOutputPerceptrons();
					Perceptron[] perceptronsConnectedToOutputs = hiddenLayers[hiddenLayers.length - 1].getHiddenPerceptrons();
					
					// update output weights
					double[] outputError = new double[perceptronsConnectedToOutputs.length];
					for (int i = 0; i < nnet.NUM_OUTPUT_PERCEPTRONS; i++) {
						double expected = expectedOutputs[i];
						double actual = predictedOutputs[i];
						
						double outModifiedError = actual * (1 - actual) * (expected - actual);
						double[] outputWeights = outputPerceptrons[i].getWeights();
						double[] newOutputWeights = new double[perceptronsConnectedToOutputs.length];
		
						// for each hidden perceptron connected to this output neuron
						for (int j = 0; j < perceptronsConnectedToOutputs.length; j++) {
							newOutputWeights[j] = LEARNING_RATE * outModifiedError * perceptronsConnectedToOutputs[j].getOutput();
							outputError[j] += outModifiedError * outputWeights[j];
						}
						outputPerceptrons[i].updateWeights(newOutputWeights);
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
					// BACKPROPAGATION END
				}
			}
			// TRAIN NEURAL NETWORK END
			
			// TEST NEURAL NETWORK START
			testingError = 0;
			ArrayList<Data> testFold = folds.get(testFoldIndex);
			for (int datapointIndex = 0; datapointIndex < testFold.size(); datapointIndex++) {
				Data testDatapoint = testFold.get(datapointIndex);
				
				// get input values and feed them into the neural network
				double[] predictedTestOutputs = nnet.run(testDatapoint);
				
				// compute what the output should be
				int[] actualOutputs = testDatapoint.getClassification();
				
				// compute testing error
				double singleTestError = 0;
				for (int i = 0; i < nnet.NUM_OUTPUT_PERCEPTRONS; i++) {
					// squaring ensures number is positive
					singleTestError += (actualOutputs[i] - predictedTestOutputs[i]) * (actualOutputs[i] - predictedTestOutputs[i]);
				}
				singleTestError /= nnet.NUM_OUTPUT_PERCEPTRONS;
				testingError += singleTestError;
			}
			testingError /= testFold.size();
			
			if (testingError < lowestTestingError) {
				lowestTestingError = testingError;
			}
			
			iteration++;
			if (TEST_ERROR_DEBUG) {
				System.out.println("Iteration: " + iteration);
				System.out.println("Testing error: " + testingError);
			}
			// TEST NEURAL NETWORK END
			
			// probably won't get much better after 1K iterations
			if (iteration > 1000) {
				System.out.printf("Fold: %d - Lowest testing error acheived: %.3f%%%n", 
						(testFoldIndex + 1), (100 * lowestTestingError));
				return;
			}
		} while (testingError > GOAL_TEST_ERROR_RATE);
		
		// test against validation data
		// only save if validation error < .1
		double validationError = tester.startTest(nnet, false);
		if (TEST_ERROR_DEBUG) {
			System.out.println("Validation error: " + validationError);
		}
		if (validationError < .1) {
			try {
				saveNeuralNetwork(String.format("nnet_fold-%d_validationerror-%.4f_testerror-%.4f_iter-%d.save",
						(testFoldIndex + 1), validationError, testingError, iteration));
			} catch (NeuralNetworkException e) {
				System.err.println(e.getMessage());
				System.exit(1);
			}
			System.out.println("Fold " + testFoldIndex + " took " + iteration + " iterations to compute.");
		}
	}
}
