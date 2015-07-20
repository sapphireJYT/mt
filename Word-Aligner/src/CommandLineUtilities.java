/**********************************************
 * 
 * Based on JHU CS475 CommandLineUtilities.java 
 * 
 **********************************************/

import java.io.*;
import java.util.*;

import org.apache.commons.cli.*;

public class CommandLineUtilities {
	
	private static CommandLine command_line = null;
	private static Properties properties = null;
	
	public static void initCommandLineParameters(String[] args, LinkedList<Option> specified_options) {
		Options options = new Options();
		if (specified_options != null)
			for (Option option : specified_options)
				options.addOption(option);
		
		Option option = null;
		
		OptionBuilder.withArgName("file");
		OptionBuilder.hasArg();
		OptionBuilder.withDescription("A file containing command line parameters.");
		option = OptionBuilder.create("parameter_file");
		
		options.addOption(option);
		
		CommandLineParser command_line_parser = new GnuParser();
		CommandLineUtilities.properties = new Properties();
		
		try {
			CommandLineUtilities.command_line = command_line_parser.parse(options, args);
		}catch (ParseException e) {
			System.out.println("Error: " + e.getClass() + ": " + e.getMessage());
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("parameters:", options);
			System.exit(0);
		}
		
		if (CommandLineUtilities.hasArg("parameter_file")) {
			String parameter_file = CommandLineUtilities.getOptionValue("parameter_file");
			try {
				properties.load(new FileInputStream(parameter_file));
			}catch (IOException e) {
				System.err.println("Problem reading parameter file: " + parameter_file);
			}
		}	
	}
	
	public static boolean hasArg(String option) {
		if (CommandLineUtilities.command_line.hasOption(option) 
				|| CommandLineUtilities.properties.containsKey(option))
			return true;
		return false;
	}
	
	public static String[] getOptionValues(String option) {
		String arguments_to_parse = null;
		if (CommandLineUtilities.command_line.hasOption(option))
			arguments_to_parse = CommandLineUtilities.command_line.getOptionValue(option);
		if (CommandLineUtilities.properties.containsKey(option)) 
			arguments_to_parse = (String)CommandLineUtilities.properties.getProperty(option);
		
		return arguments_to_parse.split(":");
	}
	
	public static String getOptionValue(String option) {
		if (CommandLineUtilities.command_line.hasOption(option)) 
			return CommandLineUtilities.command_line.getOptionValue(option);
		if (CommandLineUtilities.properties.containsKey(option))
			return (String)CommandLineUtilities.properties.getProperty(option);
		return null;
	}
	
	public static int getIntOptionValue(String option) {
		String value = CommandLineUtilities.getOptionValue(option);
		if (value != null)
			return Integer.parseInt(value);
		return -1;
	}
	
	public static double getDoubleOptionValue(String option) {
		String value = CommandLineUtilities.getOptionValue(option);
		if (value != null)
			return Double.parseDouble(value);
		return -1;
	}

}