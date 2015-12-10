package neural_network.runners;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import neural_network.Data;
import neural_network.NeuralNetwork;
import neural_network.NeuralNetworkException;


/**
 * Provides generic functionality for neural net runners.
 *  
 * @author Michael Yachanin (mry1294)
 */
public abstract class Runner {
	
	protected final int INPUT_COUNT;
	protected final int NUM_HIDDEN_LAYERS = 1;
	protected final int NUM_HIDDEN_PERCEPTRONS = 5;
	protected final int NUM_OUTPUT_PERCEPTRONS = 1;
	protected NeuralNetwork nnet;
	protected final ArrayList<Data> data;
	
	/**
	 * Constructor a runner.
	 */
	protected Runner(ArrayList<Data> data) {
		// Assume at least one datapoint.
		INPUT_COUNT = data.get(0).getInputs().length;
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
	 * Loads a saved neural network from file.
	 * @return : The neural network loaded.
	 * @throws NeuralNetworkException 
	 */
	protected NeuralNetwork loadNeuralNetwork(String filename) throws NeuralNetworkException {
		ObjectInputStream ois = null;
		
		// open neural net save file for reading
		try {
			ois = new ObjectInputStream(new FileInputStream(new File(filename)));
		} catch (IOException e) {
			throw new NeuralNetworkException(String.format("Error opening neural net save file: %s", filename), e);
		}
		
		// read from neural net save file
		NeuralNetwork nnet = null;
		try {
			nnet = (NeuralNetwork) ois.readObject();
		} catch (ClassNotFoundException | IOException e) {
			throw new NeuralNetworkException(String.format("Error reading file: %s", filename), e);
		} finally {
			try {
				ois.close();
			} catch (IOException e) {
				throw new NeuralNetworkException("Error closing file input stream used to load neural network.", e);
			}
		}
		return nnet;
	}
	
	/**
	 * Saves the current neural network to a file - nnet.save.
	 * @throws NeuralNetworkException
	 */
	protected void saveNeuralNetwork(String filename) throws NeuralNetworkException {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
			oos.writeObject(nnet);
			oos.close();
		} catch (IOException e) {
			throw new NeuralNetworkException("Error while saving neural network to a file.", e);
		}
		System.out.println("Neural net successfully saved as: " + filename);
	}
}
