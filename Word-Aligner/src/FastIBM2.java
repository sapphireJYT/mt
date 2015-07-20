import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

/*
 * A log-linear reparameterization of IBM Model 2
 */

public class FastIBM2 {
	private double p0 = 0.08;		// Null alignment probability
	private double lambda = 20;		// Precision
	
	private String[] e_sen;			// English sentences
	private String[] f_sen;			// French sentences
	private int n;				// Number of sentences used to train and align
	private HashMap<String, Double> t;	// Word translation probabilities
	private HashMap<String, Double> H;	// Diagonal distances
	private HashMap<String, Double> Z;	// Normalization constants
	private HashMap<String, Double> A;	// Alignment probability distributions
	
	public FastIBM2() {
		this.e_sen = Dictionary.e_sen;
		this.f_sen = Dictionary.f_sen;
		this.n = e_sen.length;
		this.t = new HashMap<String, Double>(Dictionary.t);	// Carry over t(e|f) from Model1
	}
	
	private void computeHZA() {
		this.H = new HashMap<String, Double>();
		this.Z = new HashMap<String, Double>();
		this.A = new HashMap<String, Double>();	
		for (int k=0; k<n; k++) {
			String[] e = e_sen[k].split(" ");
			String[] f = f_sen[k].split(" ");
			int le = e.length;
			int lf = f.length;
			for (int j=0; j<le; j++) {
				String jl = String.valueOf(j) + "_" + String.valueOf(le) + "_" + String.valueOf(lf);
				Z.put(jl, 0.0);
				for (int i=0; i<lf; i++) {
					String ijl = String.valueOf(i) + "_" + jl;
					double h = -Math.abs((double)i/(double)lf - (double)j/(double)le);
					H.put(ijl, h);
					Z.put(jl, Z.get(jl)+Math.exp(lambda*h));
					double a = i==0 ? p0 : (1.0-p0)*(Math.exp(lambda*h)/Z.get(jl));
					A.put(ijl, a);
				}
			}
		}
	}
	
	public void train() {
		HashMap<String, Double> count_ef = new HashMap<String, Double>();
		HashMap<String, Double> count_f = new HashMap<String, Double>();
		HashMap<String, Double> count_A_ij = new HashMap<String, Double>();
		HashMap<String, Double> count_A_j = new HashMap<String, Double>();
		
		// Initialize H, Z, A
		computeHZA();
		
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
						count_A_ij.put(ijl, 0.0);
						count_A_j.put(jl, 0.0);
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
						Z += t.get(ef) * A.get(ijl);
					}
					// Collect Counts
					String jl =  String.valueOf(j) + "_" + String.valueOf(le) + "_" + String.valueOf(lf);
					for (int i=0; i<lf; i++) {
						String ijl = String.valueOf(i) + "_" + jl;
						String f = f_words[i];
						String ef = e_words[j] +"_"+ f;
						double c = t.get(ef) * A.get(ijl) / Z;
						count_ef.put(ef, count_ef.get(ef)+c);
						count_f.put(f, count_f.get(f)+c);
						count_A_ij.put(ijl, count_A_ij.get(ijl)+c);
						count_A_j.put(jl, count_A_j.get(jl)+c);
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
					    A.put(ijl, count_A_ij.get(ijl)/count_A_j.get(jl));
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
					double pr = t.get(ef) * A.get(ijl);
					if (pr > best_pr)
						best_pr = pr;
				}
				for (int j=0; j<le; j++)  {
					String ef = e_words[j] +"_"+ f;
					String ijl = String.valueOf(i) + "_" + String.valueOf(j)
							+ "_" + String.valueOf(le) + "_" + String.valueOf(lf); 
					double pr = t.get(ef) * A.get(ijl);
					if (pr == best_pr)
						writer.write(String.valueOf(i) + "-" + String.valueOf(j) + " ");
				} 
			}
			writer.newLine();
		}
		writer.close();
	}
    
}