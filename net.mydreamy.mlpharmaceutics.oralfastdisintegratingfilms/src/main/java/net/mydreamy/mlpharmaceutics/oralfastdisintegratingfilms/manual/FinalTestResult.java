package net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.manual;

import net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		

		//best Model testing
		Prediction.prediction("OFDF", "manual/trainingset.csv", 100, "src/main/resources/manual/bestModel.bin", false);
		Prediction.prediction("OFDF", "manual/testingset.csv", 20, "src/main/resources/manual/bestModel.bin", false); 
		
	}
}
