package neural_network.runners;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

import neural_network.Data;
import neural_network.NeuralNetwork;


/**
 * Provides generic functionality for neural net runners.
 *  
 * @author Michael Yachanin (mry1294)
 */
public abstract class Runner implements Runnable {
	
	protected final int INPUT_COUNT;
	protected final int NUM_HIDDEN_LAYERS = 1;
	protected final int NUM_HIDDEN_PERCEPTRONS;
	protected final int NUM_OUTPUT_PERCEPTRONS = 1;
	protected NeuralNetwork nnet;
	protected final ArrayList<Data> data;
	
	
	/**
	 * Constructor a runner.
	 */
	protected Runner(ArrayList<Data> data) {
		// Assume at least one datapoint.
		INPUT_COUNT = data.get(0).getInputs().length;
		NUM_HIDDEN_PERCEPTRONS = 21;
		this.data = data;
	}

	
	/**
	 * Sets the neural network contained by this object
	 * 
	 * @param nnet: neural network to use
	 */
	protected void setNeuralNetwork(NeuralNetwork nnet) {
		this.nnet = nnet;
	}
	
	
	/**
	 * Loads a saved neural network from file - nnet.save.
	 */
	protected void loadNeuralNetwork() {
		ObjectInputStream ois = null;
		
		// open neural net save file for reading
		try {
			ois = new ObjectInputStream(new FileInputStream(new File("nnet.save")));
		} catch (IOException e) {
			System.out.println("Error opening neural net save file (nnet.save)");
			System.exit(1);
		}
		
		// read from neural net save file
		try {
			nnet = (NeuralNetwork) ois.readObject();
		} catch (ClassNotFoundException | IOException e) {
			System.out.println("Error reading neural net save file (nnet.save)");
			System.exit(1);
		}
	}
}
