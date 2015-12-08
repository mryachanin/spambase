package neural_network.runners;

import java.util.ArrayList;

import neural_network.Data;

/**
 * Used to test an artificial neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class Tester extends Runner {
	
	private final boolean DEBUG = true;
	
	/**
	 * Construct a tester for an artificial neural network.
	 */
	public Tester(ArrayList<Data> data) {
		super(data);
		loadNeuralNetwork();
	}
	
	@Override
	public void run() {
		// get input values and feed them into neural network
		for (Data datapoint : data) {
			double[] predictedOutputs = nnet.run(datapoint);
			
			if (DEBUG) {
				for (int output = 0; output < predictedOutputs.length; output++) {
					System.out.println(String.format("Output $1%d: $2%d", output, predictedOutputs[output]));
				}
			}
		}
	}
}
