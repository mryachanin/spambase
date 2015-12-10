package neural_network;

/**
 * An exception to be thrown from code implementing this neural network.
 * 
 * @author Michael Yachanin (mry1294)
 */
public class NeuralNetworkException extends Exception {

	private static final long serialVersionUID = -4047091667710496178L;

	/**
	 * Construct a new neural network exception.
	 * 
	 * @param message : A description of the error.
	 * @param cause : The stacktrace.
	 */
	public NeuralNetworkException(String message, Throwable cause) {
		super(message, cause);
	}
	
	/**
	 * Construct a new neural network exception.
	 * 
	 * @param message : A description of the error.
	 */
	public NeuralNetworkException(String message) {
		super(message);
	}
}
