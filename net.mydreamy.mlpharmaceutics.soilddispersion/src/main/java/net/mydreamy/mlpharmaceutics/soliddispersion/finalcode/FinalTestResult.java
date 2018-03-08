package net.mydreamy.mlpharmaceutics.soliddispersion.finalcode;

import net.mydreamy.mlpharmaceutics.soliddispersion.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		
		//best Model testing
		Prediction.prediction("SD", "final/trainingset.csv", 200, "src/main/resources/final/bestModel.bin", false);
		Prediction.prediction("SD", "final/testingset.csv", 20, "src/main/resources/final/bestModel.bin", false); 
		Prediction.prediction("SD", "final/extrascaledtestset.csv", 20, "src/main/resources/final/bestModel.bin", false); 
		
	}
}
 