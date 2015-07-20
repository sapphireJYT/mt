import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

public class IBM1 {
	private String[] e_sen;			// English sentences
	private String[] f_sen;			// French sentences
	private int n;				// Number of sentences used to train and align
	private HashMap<String, Double> t;	// Word translation probabilities
	
	public IBM1() throws IOException {
		this.e_sen = Dictionary.e_sen;
		this.f_sen = Dictionary.f_sen;
		this.n = e_sen.length;
	}
	
	public void train() {
		HashMap<String, Double> count_ef = new HashMap<String, Double>();
		HashMap<String, Double> count_f = new HashMap<String, Double>();
		
		// Initialize t(e|f) uniformly
		this.t = new HashMap<String, Double>();
		for (int k=0; k<n; k++) { 
			String[] e_words = e_sen[k].split(" ");
			String[] f_words = f_sen[k].split(" ");
			for (String e : e_words) {
				for (String f : f_words) {
					String ef = e +"_"+ f;
					if (!count_ef.containsKey(ef)) 
						count_ef.put(ef, 1.0);
					else
						count_ef.put(ef, count_ef.get(ef)+1.0);
					
					if(!count_f.containsKey(f))
						count_f.put(f, 1.0);
					else
						count_f.put(f, count_f.get(f)+1.0);
				}
			}
		}
		for (String ef : count_ef.keySet()) {
			String f = ef.split("_")[1];
			t.put(ef, count_ef.get(ef)/count_f.get(f));
		}
		
		for(int r=0; r<5; r++) {
			// Initialize 
			for (int k=0; k<n; k++) {
				// for each sentence pair
				String[] e_words = e_sen[k].split(" ");
				String[] f_words = f_sen[k].split(" ");
				for (String f : f_words) {
					count_f.put(f, 0.0);
					for (String e : e_words) {
						String ef = e +"_"+ f; 
						count_ef.put(ef, 0.0);
					}
				}	
			}
			
			for (int k=0; k<n; k++) { 
				String[] e_words = e_sen[k].split(" ");
				String[] f_words = f_sen[k].split(" ");
				for (String e : e_words) {
					// Compute normalization
					double Z = 0.0;
					for (String f : f_words) {
						String ef = e +"_"+ f;
						Z += t.get(ef);
					}
					// Collect Counts
					for (String f : f_words) {
						String ef = e +"_"+ f;
						double c = t.get(ef) / Z;
						count_ef.put(ef, count_ef.get(ef)+c);
						count_f.put(f, count_f.get(f)+c);
					}
				}	
			}
			
			// Estimate probabilities
			for (int k=0; k<n; k++) { 
				String[] e_words = e_sen[k].split(" ");
				String[] f_words = f_sen[k].split(" ");
				for (String f : f_words) {
					for (String e : e_words) {
						String ef =  e +"_"+ f; 
						t.put(ef, count_ef.get(ef)/count_f.get(f));
					}
				}
			}
		}
		Dictionary.t = new HashMap<String, Double>(t);
	}
	
	public void align() throws IOException {
		File file = new File(Align.path);
		if(file.exists())
			file.delete();
		file.createNewFile();
		BufferedWriter writer = new BufferedWriter(new FileWriter(file)); // Write the alignments to file
		
		for (int k=0; k<n; k++) { 
			String[] e_words = e_sen[k].split(" ");
			String[] f_words = f_sen[k].split(" ");
			for (int i=0; i<f_words.length; i++) {
				double best_pr = 0.0;
				for (int j=0; j<e_words.length; j++)  {
					String ef = e_words[j] +"_"+ f_words[i];
					if (t.get(ef) > best_pr)
						best_pr = t.get(ef);
				}
				for (int j=0; j<e_words.length; j++)  {
					String ef = e_words[j] +"_"+ f_words[i];
					if (t.get(ef) == best_pr)
						writer.write(String.valueOf(i) + "-" + String.valueOf(j) + " ");
				} 
			}
			writer.newLine();
		}
		writer.close();
	}
    
}