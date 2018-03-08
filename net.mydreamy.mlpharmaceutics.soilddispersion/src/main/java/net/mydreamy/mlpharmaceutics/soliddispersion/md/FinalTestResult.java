package net.mydreamy.mlpharmaceutics.soliddispersion.md;

import net.mydreamy.mlpharmaceutics.soliddispersion.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		
		//best Model testing
		Prediction.prediction("SD", "md/trainingset.csv", 200, "src/main/resources/md/bestModel.bin", false);
		Prediction.prediction("SD", "md/testingset.csv", 20, "src/main/resources/md/bestModel.bin", false); 
		
	}
}
 