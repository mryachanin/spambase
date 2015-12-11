package neural_network;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Represents a perceptron in an artificial neural network.
 *  
 * @author Michael Yachanin (mry1294)
 */
public class Perceptron implements Serializable {
	
	private static final long serialVersionUID = 6743181867327070353L;
	
	private final int NUM_INPUTS;
	private double[] weights;
	private final double[] initialWeights;
	private double transferValue;
	
	/**
	 * Construct a perceptron.
	 *  
	 * @param numInputs: The number of inputs connected to this perceptron.
	 */
	public Perceptron(int numInputs) {
		NUM_INPUTS = numInputs;
		weights = new double[NUM_INPUTS];
		
		Random rand = new Random();
		for (int i = 0; i < NUM_INPUTS; i++) {
			// initialize weights between to values -1 and 1
			weights[i] = Math.pow(-1, (rand.nextInt(2) + 1)) * rand.nextDouble();
		}
		initialWeights = Arrays.copyOf(weights, numInputs);
		transferValue = -1;
	}
	
	/**
	 * Aggregates the weighted inputs connected to this perceptron.
	 * 
	 * @param inputs: An array of inputs to aggregate.
	 * 
	 * @throws IllegalArgumentException: input array length must be equal to the 
	 * number of inputs given upon instantiation.
	 */
	private double[] generateWeightedInputs(double[] inputs) {
		if (inputs.length != NUM_INPUTS) {
			String errorStr = "The number of given inputs is not equal to the number of expected inputs.";
			throw new IllegalArgumentException(errorStr);
		}
		
		double[] weightedInputs = new double[inputs.length];
		for (int i = 0; i < inputs.length; i++) {
			weightedInputs[i] = inputs[i] * weights[i];
		}
		return weightedInputs;
	}
	
	/**
	 * Aggregates the weighted inputs connected to this perceptron.
	 */
	private double aggregateWeightedInputs(double[] weightedInputs) {
		double sum = 0;
		for (double d : weightedInputs) {
			sum += d;
		}
		return sum / weightedInputs.length;
	}
	
	/**
	 * Implements a sigmoid function that converts the weighted inputs' sum
	 * to a value between 0 and 1.
	 */
	private double normalize(double sum) {
		return 1 / (1 + Math.pow(Math.E, -sum));
	}
	
	/**
	 * Returns an output based on the aggregation and transformation
	 * of the weighted inputs passed in.
	 *  
	 * @param inputs: An array of inputs to aggregate.
	 * @return: Boolean output based on value generated by transfer function.
	 */
	public double getOutput(double[] inputs) {
		double[] weightedInputs = generateWeightedInputs(inputs);
		double sum = aggregateWeightedInputs(weightedInputs);
		transferValue = normalize(sum);
		return transferValue;
	}
	
	/**
	 * Returns the output of this perceptron.
	 * 
	 * @return: The output from this perceptron.
	 */
	public double getOutput() {
		if (transferValue == -1) {
			throw new IllegalStateException("This perceptron has not fired yet.");
		}
		return transferValue;
	}
	
	/**
	 * Returns the weights of the inputs connected to this perceptron.
	 * 
	 * @return: The weights of the inputs connected to this perceptron.
	 */
	public double[] getWeights() {
		return weights;
	}
	
	/**
	 * Updates the weights of the inputs connected to this perceptron.
	 * Used during backpropagation to train the perceptron.
	 *  
	 * @param weightDeltas: An array of deltas to add to the current input weights.
	 */
	public void updateWeights(double[] weightDeltas) {
		for (int i=0; i < weights.length; i++) {
			weights[i] += weightDeltas[i];
		}
	}
	
	/**
	 * Resets this perceptron to its initial state.
	 */
	public void reset() {
		weights = initialWeights;
		transferValue = -1;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(weights);
	}
}
