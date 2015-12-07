package neural_network.runners;

/**
 * Main program for classifying email spam using
 * an artificial neural network.
 * 
 * Usage: EmailSpamClassifier < train | test >
 * 
 * @author Mike Yachanin (mry1294)
 */
public class EmailSpamClassifier {	
	
	/**
	 * Main Program.
	 * 
	 * @param args: args[0] = "test" or "train" depending on use case
	 */
	public static void main(String[] args) {
		// check for correct usage
		if (args.length != 1) {
			usage();
		}
		
		switch (args[0].toLowerCase()) {
			case "test":
				new Thread( new Tester()).start();
				break;
			
			case "train":
				new Thread( new Trainer()).start();
				break;
				
			default:
				usage();
		}
	}
	
	
	/**
	 * Prints a usage message to stdout and exits. 
	 */
	private static void usage() {
		System.out.println("Usage: EmailSpamClassifier < test | train >");
		System.exit(1);
	}
}
