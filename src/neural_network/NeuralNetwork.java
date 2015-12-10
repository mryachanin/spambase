package neural_network;

import java.io.Serializable;

import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;

/**
 * Represents an artificial neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class NeuralNetwork implements Serializable {
	
	private static final long serialVersionUID = 5987541387089871958L;
	
	private final HiddenLayer[] hiddenLayers;
	private final Perceptron[] outputPerceptrons;
	public final int NUM_INPUTS;
	public final int NUM_HIDDEN_LAYERS;
	public final int NUM_OUTPUT_PERCEPTRONS;
	
	private double[] lastInputs;
	
	/**
	 * Construct the artificial neural network.
	 * 
	 * @param numInputs: The number of inputs.
	 * @param numHiddenLayers: The number of hidden layers in this neural network.
	 * @param numHiddenPerceptrons: The number of perceptrons in each hidden layer.
	 * @param numOutputPerceptrons: The number of output perceptrons.
	 */
	public NeuralNetwork(int numInputs, int numHiddenLayers, int numHiddenPerceptrons, int numOutputPerceptrons) {
		if (numHiddenLayers <= 0) {
			throw new ValueException("The number of hidden layers must be greater than zero.");
		}
		NUM_INPUTS = numInputs;
		NUM_HIDDEN_LAYERS = numHiddenLayers;
		NUM_OUTPUT_PERCEPTRONS = numOutputPerceptrons;
		lastInputs = null;
		
		// Initialize hidden layers
		hiddenLayers = new HiddenLayer[NUM_HIDDEN_LAYERS];
		hiddenLayers[0] = new HiddenLayer(numHiddenPerceptrons, numInputs);
		for (int hiddenLayer = 1; hiddenLayer < NUM_HIDDEN_LAYERS; hiddenLayer++) {
			hiddenLayers[hiddenLayer] = new HiddenLayer(numHiddenPerceptrons, numHiddenPerceptrons);
		}
		
		// Initialize output perceptrons
		outputPerceptrons = new Perceptron[NUM_OUTPUT_PERCEPTRONS];
		for (int outputPerceptron = 0; outputPerceptron < NUM_OUTPUT_PERCEPTRONS; outputPerceptron++) {
			outputPerceptrons[outputPerceptron] = new Perceptron(numHiddenPerceptrons);
		}
	}
	
	/**
	 * Returns an array of the output perceptrons.
	 * 
	 * @return: The output perceptrons in this network.
	 */
	public Perceptron[] getOutputPerceptrons() {
		return outputPerceptrons;
	}
	
	/**
	 * Returns an array of the hidden layers in this network.
	 *  
	 * @return: The hidden layers in this network.
	 */
	public HiddenLayer[] getHiddenLayers() {
		return hiddenLayers;
	}
	
	/**
	 * Returns an array of the last inputs passed through this network.
	 *  
	 * @return: The last inputs passed through this network.
	 */
	public double[] getLastInputs() {
		if (lastInputs == null) {
			throw new IllegalStateException("This neural network has not been passed any inputs yet");
		}
		return lastInputs;
	}
	
	/**
	 * Resets all perceptrons in this neural network to their original random weights.
	 */
	public void reset() {
		for (HiddenLayer hiddenLayer : hiddenLayers) {
			hiddenLayer.reset();
		}
		
		for (Perceptron outputPerceptron : outputPerceptrons) {
			outputPerceptron.reset();
		}
		lastInputs = null;
	}
	
	/**
	 * Runs an array of inputs through this network.
	 *  
	 * @param inputs: The inputs to use.
	 * @return: The output of this neural network given an array of inputs.
	 */
	public double[] run(Data inputs) {
		double[] lastOutputs = inputs.getInputs();
		
		for (int hiddenLayer = 0; hiddenLayer < NUM_HIDDEN_LAYERS; hiddenLayer++) {
			Perceptron[] hiddenPerceptrons = hiddenLayers[hiddenLayer].getHiddenPerceptrons();
			double[] nextOutputs = new double[hiddenPerceptrons.length];
			for (int hiddenPerceptron = 0; hiddenPerceptron < hiddenPerceptrons.length; hiddenPerceptron++) {
				nextOutputs[hiddenPerceptron] = hiddenPerceptrons[hiddenPerceptron].getOutput(lastOutputs);
			}
			lastOutputs = nextOutputs;
		}
		
		double[] output = new double[NUM_OUTPUT_PERCEPTRONS];
		for (int outputPerceptron = 0; outputPerceptron < NUM_OUTPUT_PERCEPTRONS; outputPerceptron++) {
			output[outputPerceptron] = outputPerceptrons[outputPerceptron].getOutput(lastOutputs);
		}
		
		lastInputs = inputs.getInputs();
		return output;
	}
}
