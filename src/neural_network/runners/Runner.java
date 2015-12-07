package neural_network.runners;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.Scanner;

import neural_network.NeuralNetwork;


/**
 * Provides generic functionality for neural net runners.
 *  
 * @author Mike Yachanin (mry1294)
 */
public abstract class Runner implements Runnable {
	
	protected final int INPUT_COUNT = 31;
	protected final int NUM_HIDDEN_LAYERS = 1;
	protected final int NUM_HIDDEN_PERCEPTRONS = 0;
	protected final int NUM_OUTPUT_PERCEPTRONS = 1;
	protected NeuralNetwork nnet;
	protected final Scanner in;
	
	
	/**
	 * Constructor a runner.
	 */
	protected Runner() {
		in = new Scanner(System.in);
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
			this.nnet = (NeuralNetwork) ois.readObject();
		} catch (ClassNotFoundException | IOException e) {
			System.out.println("Error reading neural net save file (nnet.save)");
			System.exit(1);
		}
	}
	
	
	/**
	 * Parses a text segment into inputs for a neural net.
	 * 
	 * @param text: text to parse inputs from
	 * @return: double array containing parsed values
	 */
	protected double[] getInputs(String text) {
		double[] inputs = new double[INPUT_COUNT];
		
		HashMap<Character, Integer> charCounts = getCharCounts(text);
		double validChars = getNumValidChars(text);
		
		int count = 0;
		
		// first 26 inputs = English alphabet char frequencies
		for (int c='a'; c <= 'z'; c++) {
			if (charCounts.get(new Character((char)c)) != null) 
				inputs[count++] = (double)charCounts.get(new Character((char)c)) / validChars;
		}
		
		// next 5 inputs = accented vowels 
		char[] accentedVowels = new char[] { 'à', 'è', 'ì', 'ò', 'ù' };
		for (char c : accentedVowels) {
			if (charCounts.get(c) != null)
				inputs[count++] = (double)charCounts.get(c) / validChars * 100;
		}
		
		return inputs;
	}
	
	
	/**
	 * Returns a hashmap containing the count of characters in a given string
	 * 
	 * @param text: text segment
	 * @return: count of chars in the string passed in
	 */
	private HashMap<Character, Integer> getCharCounts(String text) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		text = text.toLowerCase();
		for (int i=0; i < text.length(); i++) {
			char nextChar = text.charAt(i);
			if (map.containsKey(nextChar)) 
				map.put(nextChar, map.get(nextChar) + 1);
			else 
				map.put(nextChar, 1);
		}
		return map;
	}
	
	
	/**
	 * Returns the number of relevant characters in a string.
	 * 
	 * @param text: string of text to parse
	 * @return: number of relevant characters in a give string
	 */
	private int getNumValidChars(String text) {
		int sum = 0;
		text = text.toLowerCase();
		for (int i=0; i < text.length(); i++) {
			char c = text.charAt(i);
			if ((c >= 'a' && c <= 'z')
				|| c == 'à' || c == 'è' || c == 'ì' || c == 'ò' || c == 'ù')
				sum++;
		}
		return sum;
	}
	
	/**
	 * Returns a hashmap containing the count of words in a given string
	 * 
	 * @param text: text segment
	 * @return: count of words in the string passed in
	 */
/* Not used per specs
	private HashMap<String, Integer> getWordCounts(String text) {
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		String[] words = text.toLowerCase().split(" ");
		for (String s : words) {
			if (map.containsKey(s))
				map.put(s, map.get(s) + 1);
			else
				map.put(s, 1);
		}
		return map;
	}
*/
}
