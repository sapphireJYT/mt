import java.io.IOException;
import java.util.LinkedList;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

public class Align {
	static public LinkedList<Option> options = new LinkedList<Option>();
	static public String path = "data/alignments"; // default file path to store the alignment results 

	public static void main(String[] args) throws IOException {
		// Parse the command line.
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Align.options);
		
		String data = "data/hansards";	// default training data file
		if (CommandLineUtilities.hasArg("d"))
			data = CommandLineUtilities.getOptionValue("d");
		
		String english = data + ".e";	// English filename
		String french = data + ".f";	// French filename
		
		int num_sentences = 100000; 	// default number of sentences to use for training of alignment
		if (CommandLineUtilities.hasArg("n"))
			CommandLineUtilities.getIntOptionValue("n");
		
		// Read the training data
		DataReader data_reader = new DataReader(english);
		String[] e_data = data_reader.readData(num_sentences);
		// String[] e_data = data_reader.readFullData();
		data_reader = new DataReader(french);
		String[] f_data = data_reader.readData(num_sentences);
		// String[] f_data = data_reader.readFullData();
		data_reader.close();	
		
		Dictionary.e_sen = e_data;
		Dictionary.f_sen = f_data;
		
		// Train and align the data with IBM Model1
		IBM1 m1 = new IBM1();
		m1.train();
		// m1.align();
		
		// IBM2 m2 = new IBM2();
		// m2.train();
		// m2.align();
		
		FastIBM2 fast_ibm2 = new FastIBM2();
		fast_ibm2.train();
		fast_ibm2.align();
	}

	private static void createCommandLineOptions() {
		registerOption("d", "String", true, "Training data file.");
		registerOption("n", "int", true, "Number of sentences to use for training and alignment.");
	}
	
	private static void  registerOption(String option_name, String arg_name, boolean has_arg, String description) {
		OptionBuilder.withArgName(arg_name);
		OptionBuilder.hasArg(has_arg);
		OptionBuilder.withDescription(description);
		Option option = OptionBuilder.create(option_name);
		Align.options.add(option);
	}
    
}