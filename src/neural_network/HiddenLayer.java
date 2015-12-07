package neural_network;

import java.io.Serializable;

import neural_network.perceptron.Perceptron;

/**
 * Represents a hidden layer in an artificial neural network.
 * 
 * @author Mike Yachanin (mry1294)
 */
public class HiddenLayer implements Serializable {

	private static final long serialVersionUID = -587112015006963582L;
	
	private Perceptron[] perceptrons;
	
	/**
	 * Construct a hidden layer for an artificial neural network.
	 * 
	 * @param numPerceptrons: The number of perceptrons in this hidden layer.
	 * @param numInputs: The number of inputs to each perceptron.
	 */
	public HiddenLayer(int numPerceptrons, int numInputs) {
		// initialize the perceptrons in this hidden layer
		for (int perceptron = 0; perceptron < numPerceptrons; perceptron++) {
			perceptrons[perceptron] = new Perceptron(numInputs);
		}
	}
	
	/**
	 * Returns the number of perceptrons in this hidden layer.
	 * 
	 * @return: The number of perceptrons in this hidden layer.
	 */
	public int getNumPerceptrons() {
		return perceptrons.length;
	}
	
	/**
	 * Returns an array of the perceptrons in this hidden layer.
	 * 
	 * @return: The perceptrons in this hidden layer.
	 */
	public Perceptron[] getHiddenPerceptrons() {
		return perceptrons;
	}
}