package net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.md;

import net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		

		//best Model testing
		Prediction.prediction("OFDF", "md/trainingset.csv", 100, "src/main/resources/md/bestModel.bin", false);
		Prediction.prediction("OFDF", "md/testingset.csv", 20, "src/main/resources/md/bestModel.bin", false); 
		
	}
}
