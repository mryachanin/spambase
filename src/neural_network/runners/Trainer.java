package neural_network.runners;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import neural_network.Data;
import neural_network.HiddenLayer;
import neural_network.NeuralNetwork;
import neural_network.Perceptron;


/**
 * Used to train this artificial neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class Trainer extends Runner {

	private final double LEARNING_RATE = .1;
	private final double GOAL_TEST_ERROR_RATE = .15;
	private final boolean TRAIN_ERROR_DEBUG = false;
	private final boolean TEST_ERROR_DEBUG = false;
	
	/**
	 * Construct a trainer for an artificial neural network.
	 */
	public Trainer(ArrayList<Data> data) {
		super(data);
		nnet = new NeuralNetwork(INPUT_COUNT, NUM_HIDDEN_LAYERS, NUM_HIDDEN_PERCEPTRONS, NUM_OUTPUT_PERCEPTRONS);
	}
	
	private class TrainOneFold extends Thread {
		private int testFoldIndex;
		private ArrayList<ArrayList<Data>> folds;
		
		public TrainOneFold(int testFoldIndex, ArrayList<ArrayList<Data>> folds) {
			this.testFoldIndex = testFoldIndex;
			this.folds = folds;
		}
		
		@Override
		public void run() {
			double testingError, trainingError;
			int iteration = 0;
			do {
				// TRAIN NEURAL NETWORK START
				trainingError = 0;
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
						int[] actualOutputs = datapoint.getClassification();
			
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
				
				if (TEST_ERROR_DEBUG)
					System.out.println("Testing error: " + testingError);
				// TEST NEURAL NETWORK END
				
				iteration++;
			} while (testingError > GOAL_TEST_ERROR_RATE);
			
			
			// prompt to save neural net for future use
			testingError *= 100000;
			testingError = Math.round(testingError);
			testingError /= 100000;
			saveNeuralNetwork("nnet-fold-" + (testFoldIndex + 1) + "-error-" + testingError);
			System.out.println("Fold " + testFoldIndex + " took " + iteration + " iterations to compute.");
		}
	}
	
	@Override
	public void run() {
		// allocate 10 folds for 10-fold cross validation
		ArrayList<ArrayList<Data>> folds = new ArrayList<ArrayList<Data>>();
		for (int foldNum = 0; foldNum < 10; foldNum++) {
			ArrayList<Data> fold = new ArrayList<>();
			for (int datapoint = foldNum; datapoint < data.size(); datapoint += 10) {
				fold.add(data.get(datapoint));
			}
			folds.add(fold);
		}
		
		for (int foldNum = 0; foldNum < 5; foldNum++) {
			new TrainOneFold(foldNum, folds).start();
		}
	}
	
	
	/**
	 * Saves the current neural network to a file - nnet.save.
	 */
	private void saveNeuralNetwork(String filename) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
			oos.writeObject(nnet);
			oos.close();
		} catch (IOException e) {
			System.out.println("Error while saving neural network to a file.");
			System.exit(1);
		}
		System.out.println("Neural net successfully saved as: " + filename);
	}
}
