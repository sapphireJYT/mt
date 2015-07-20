import java.util.HashMap;

/*
 * Data structures for Word Alignment Models;
 */

public class Dictionary {
	static public String[] e_sen;		 // English sentences
	static public String[] f_sen;		 // French sentences
	static public String[] align; 		 // Alignment function
	static public HashMap<String, Double> t; // Word translation probabilities 
}