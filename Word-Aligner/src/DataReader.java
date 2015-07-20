import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class DataReader {
	
	private Scanner scanner;

	public DataReader (String filename) throws FileNotFoundException {
		this.scanner = new Scanner(new BufferedInputStream(new FileInputStream(filename)));
	}
	
	public String[] readData(int num_sentences) {
		
		String[] text = new String[num_sentences];
		int n = 0;
		while (n<num_sentences && scanner.hasNextLine()) {
			String line = scanner.nextLine();
			if (line.trim().length() == 0)
				continue;
			text[n] = line;
			n ++;
		}
		return text;
	}
	
	public String[] readFullData() {
		ArrayList<String> data = new ArrayList<String>();
		while (scanner.hasNextLine()) {
			String line = scanner.nextLine();
			if (line.trim().length() == 0)
				continue;
			data.add(line);
		}
		
		String[] text = new String[data.size()];
		for (int i=0; i<data.size(); i++)
			text[i] = data.get(i);
		
		return text;
	}
	
	public HashMap<String, Double> collectAlignProbs() {
		HashMap<String, Double> align_pr = new HashMap<String, Double>();
		Dictionary.align = new String[Dictionary.e_sen.length];
		int n = 0;
		while (scanner.hasNext()) {
			String line = scanner.nextLine();
			if (line.trim().length() == 0) 
				continue;
			Dictionary.align[n] = line;
			String[] aligns = line.split(" ");
			int I = Dictionary.e_sen[n].length();
			int norm = 0;	// normalization constant
			for (int l=1; l<=I; l++) {
				norm += l;
			}
			for (int k=1; k<aligns.length; k++) {
				String a0 = aligns[k-1];
				String a1 = aligns[k];
				int i0 = Integer.parseInt(a0.split("-")[1]);	// position of English word e_(i-1)
				int i1 = Integer.parseInt(a1.split("-")[1]);	// position of English word e_i
				int s = Math.abs(i1-i0);	// jump width 
				double pr = (double)s / Math.abs((double)(norm-I*s));	// p(a_i|a_(i-1),I)
				String key = String.valueOf(i1) +"_"+ String.valueOf(i0);
				if (!align_pr.containsKey(key))
					align_pr.put(key, pr);
				else
					align_pr.put(key, (align_pr.get(key)+pr)/2.0);
			}
			n ++;
		}
		return align_pr;
	}
	
	public void close() {
		this.scanner.close();
	}
	
}
