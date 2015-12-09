package neural_network.runners;

import java.util.ArrayList;

import neural_network.Data;

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
		loadNeuralNetwork();
	}
	
	@Override
	public void run() {
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
		testingError *= 100000;
		testingError = Math.round(testingError);
		testingError /= 1000;
		System.out.println("Accuracy: " + (100 - testingError) + "%");
		System.out.println("    T    F  ");
		System.out.printf("T | %d  %d\n", TP, FP);
		System.out.printf("F | %d  %d\n", FN, TN);
	}
}
