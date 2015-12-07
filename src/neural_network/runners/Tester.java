package neural_network.runners;

/**
 * Used to test an artificial neural network.
 * 
 * @author Mike Yachanin (mry1294)
 */
public class Tester extends Runner {
	
	private final boolean DEBUG = false;
	
	/**
	 * Construct a tester for an artificial neural network.
	 */
	public Tester() {
		loadNeuralNetwork();
	}
	
	@Override
	public void run() {
		System.out.println("Enter data to classify: ");
		String text = in.nextLine();
		
		// get input values and feed them into neural network
		double[] predictedOutputs = nnet.run(getInputs(text));
		
		if (DEBUG) {
			for (int output = 0; output < predictedOutputs.length; outputs++) {
				System.out.println(String.format("Output $1%d: $2%d", output, predictedOutputs[output]))
			}
		}
		
		// print best prediction
		if (predictedOutputs[0] > predictedOutputs[1] && predictedOutputs[0] > predictedOutputs[2])
			System.out.println("English");
		else if (predictedOutputs[1] > predictedOutputs[2])
			System.out.println("Italian");
		else
			System.out.println("Dutch");
	}
}
