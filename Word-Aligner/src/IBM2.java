import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

public class IBM2 {
	private String[] e_sen;			// English sentences
	private String[] f_sen;			// French sentences
	private int n;				// Number of sentences used to train and align
	private HashMap<String, Double> t;	// Word translation probabilities
	private HashMap<String, Double> a;	// Alignment probability distribution

	public IBM2() {
		this.e_sen = Dictionary.e_sen;
		this.f_sen = Dictionary.f_sen;
		this.n = e_sen.length;
		this.t = new HashMap<String, Double>(Dictionary.t);	// Carry over t(e|f) from Model1
		this.a = new HashMap<String, Double>();
	}
	
	public void train() {
		HashMap<String, Double> count_ef = new HashMap<String, Double>();
		HashMap<String, Double> count_f = new HashMap<String, Double>();
		HashMap<String, Double> count_a_ij = new HashMap<String, Double>();
		HashMap<String, Double> count_a_j = new HashMap<String, Double>();
		
		// Initialize a
		for (int k=0; k<n; k++) { 
			String[] e_words = e_sen[k].split(" ");
			String[] f_words = f_sen[k].split(" ");
			int le = e_words.length;
			int lf = f_words.length;
			for (int i=0; i<lf; i++) { // French word at position i
				for (int j=0; j<le; j++) { // English word at position j
					String ijl = String.valueOf(i) + "_" + String.valueOf(j)
							+ "_" + String.valueOf(le) + "_" + String.valueOf(lf);  
					a.put(ijl, 1/(double)(lf+1));	// a(i|j, l_e, l_f)
				}
			}
		}
		
		for(int r=0; r<5; r++) {
			// Initialize 
			for (int k=0; k<n; k++) {
				// for each sentence pair
				String[] e_words = e_sen[k].split(" ");
				String[] f_words = f_sen[k].split(" ");
				int le = e_words.length;
				int lf = f_words.length;
				for (int i=0; i<lf; i++) { // French word at position i
					String f = f_words[i];
					count_f.put(f, 0.0);
					for (int j=0; j<le; j++) { // English word at position j
						String ef = e_words[j] + "_" + f;
						String jl =  String.valueOf(j) + "_" + String.valueOf(le) + "_" + String.valueOf(lf);
						String ijl = String.valueOf(i) + "_" + jl;
						count_ef.put(ef, 0.0);
						count_a_ij.put(ijl, 0.0);
						count_a_j.put(jl, 0.0);
					}		
				}	
			}
			
			for (int k=0; k<n; k++) { 
				String[] e_words = e_sen[k].split(" ");
				String[] f_words = f_sen[k].split(" ");
				int le = e_words.length;
				int lf = f_words.length;
				for (int j=0; j<le; j++ ) {
					// Compute normalization
					double Z = 0.0;
					for (int i=0; i<lf; i++) {
						String ef = e_words[j] +"_"+ f_words[i];
						String ijl = String.valueOf(i) + "_" + String.valueOf(j)
								+ "_" + String.valueOf(le) + "_" + String.valueOf(lf); 
						Z += t.get(ef) * a.get(ijl);
					}
					// Collect Counts
					String jl =  String.valueOf(j) + "_" + String.valueOf(le) + "_" + String.valueOf(lf);
					for (int i=0; i<lf; i++) {
						String ijl = String.valueOf(i) + "_" + jl;
						String f = f_words[i];
						String ef = e_words[j] +"_"+ f;
						double c = t.get(ef)*a.get(ijl) / Z;
						count_ef.put(ef, count_ef.get(ef)+c);
						count_f.put(f, count_f.get(f)+c);
						count_a_ij.put(ijl, count_a_ij.get(ijl)+c);
						count_a_j.put(jl, count_a_j.get(jl)+c);
					}
				}	
			}
			
			// Estimate probabilities
			for (int k=0; k<n; k++) { 
				String[] e_words = e_sen[k].split(" ");
				String[] f_words = f_sen[k].split(" ");
				int le = e_words.length;
				int lf = f_words.length;
				for (int j=0; j<le; j++ ) {
					String jl =  String.valueOf(j) + "_" + String.valueOf(le) + "_" + String.valueOf(lf);
					for (int i=0; i<lf; i++) {
						String ijl = String.valueOf(i) + "_" + jl;
						String f = f_words[i];
						String ef = e_words[j] +"_"+ f;
						t.put(ef, count_ef.get(ef)/count_f.get(f));
						a.put(ijl, count_a_ij.get(ijl)/count_a_j.get(jl));
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
			int le = e_words.length;
			int lf = f_words.length;
			for (int i=0; i<lf; i++) {
				String f = f_words[i];
				double best_pr = 0.0;
				for (int j=0; j<le; j++)  {
					String ef = e_words[j] +"_"+ f;
					String ijl = String.valueOf(i) + "_" + String.valueOf(j)
							+ "_" + String.valueOf(le) + "_" + String.valueOf(lf); 
					double pr = t.get(ef) * a.get(ijl);
					if (pr > best_pr)
						best_pr = pr;
				}
				for (int j=0; j<le; j++)  {
					String ef = e_words[j] +"_"+ f;
					String ijl = String.valueOf(i) + "_" + String.valueOf(j)
							+ "_" + String.valueOf(le) + "_" + String.valueOf(lf); 
					double pr = t.get(ef) * a.get(ijl);
					if (pr == best_pr)
						writer.write(String.valueOf(i) + "-" + String.valueOf(j) + " ");
				} 
			}
			writer.newLine();
		}
		writer.close();
	}
    
}