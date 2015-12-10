package neural_network.runners;

import java.util.ArrayList;

import neural_network.Data;
import neural_network.NeuralNetwork;
import neural_network.NeuralNetworkException;

/**
 * Used to test an artificial neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class Tester extends Runner {
	
	/**
	 * Construct a tester for an artificial neural network.
	 */
	public Tester(ArrayList<Data> data) {
		super(data);
	}
	
	/**
	 * Test a neural network against validation data.
	 * 
	 * @param neuralNetFilepath : Filepath to the neural network to test.
	 * @param debug : Should a confusion matrix be printed?
	 * @return : The validation error rate.
	 * @throws NeuralNetworkException 
	 */
	public double startTest(String neuralNetFilepath, boolean debug) throws NeuralNetworkException {
		NeuralNetwork nnet = loadNeuralNetwork(neuralNetFilepath);
		return startTest(nnet, debug);
	}
	
	/**
	 * Test a neural network against validation data.
	 * 
	 * @param nnet : The neural network to test.
	 * @param debug : Should a confusion matrix be printed?
	 * @return : The validation error rate.
	 */
	public double startTest(NeuralNetwork nnet, boolean debug) {
		this.nnet = nnet;
		
		int TP = 0, TN = 0, FP = 0, FN = 0;
		for (Data datapoint : data) {
			// get input values and feed them into the neural network
			double[] predictedTestOutputs = nnet.run(datapoint);

			// compute what the output should be
			int[] actualOutputs = datapoint.getClassification();

			// compute testing error
			for (int i = 0; i < nnet.NUM_OUTPUT_PERCEPTRONS; i++) {
				int predicted = (int) Math.round(predictedTestOutputs[i]);
				int actual = actualOutputs[i];

				if (actual == 1 && predicted == 1) {
					TP++;
				}
				else if (actual == 1 && predicted == 0) {
					FN++;
				}
				else if (actual == 0 && predicted == 1) {
					FP++;
				}
				else {
					TN++;
				}
			}
		}
		double testingError = ((double)(FP + FN) / nnet.NUM_OUTPUT_PERCEPTRONS) / data.size();
		
		if (debug) {
			double accuracy = 100 * (1 - testingError);
			System.out.printf("Accuracy: %.4f%%%n", accuracy);
			
			// print confusion matrix
			System.out.println("    T    F  ");
			System.out.printf("T | %d  %d\n", TP, FP);
			System.out.printf("F | %d  %d\n", FN, TN);
		}
		return testingError;
	}
}
