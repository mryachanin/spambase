package neural_network;

/**
 * Represents a datapoint to be used with this artificial neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class Data {
	private double[] inputs;
	private int[] classification;
	
	/**
	 * Construct a datapoint.
	 * 
	 * @param inputs: An array of features to be used as inputs.
	 * @param classification: An array representing the bitwise 
	 * classification of this datapoint.
	 */
	public Data(double[] inputs, int[] classification) {
		this.inputs = inputs;
		this.classification = classification;
	}
	
	/**
	 * Returns the inputs given by this datapoint.
	 * 
	 * @return: The inputs given by this datapoint.
	 */
	public double[] getInputs() {
		return inputs;
	}
	
	/**
	 * Returns the classification of this datapoint.
	 * 
	 * @return: The classification of this datapoint.
	 */
	public int[] getClassification() {
		return classification;
	}
}
