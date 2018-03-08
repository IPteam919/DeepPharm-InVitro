package net.mydreamy.mlpharmaceutics.soliddispersion.mdfis;

import net.mydreamy.mlpharmaceutics.soliddispersion.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		
		//best Model testing
		Prediction.prediction("SD", "mdfis/trainingset.csv", 200, "src/main/resources/mdfis/bestModel.bin", false);
		Prediction.prediction("SD", "mdfis/testingset.csv", 20, "src/main/resources/mdfis/bestModel.bin", false); 
		
	}
}
 