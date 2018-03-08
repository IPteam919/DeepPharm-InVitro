package net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.finalcode;

import net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		
		Prediction p = new Prediction();
		//best Model testing
		p.prediction("OFDF-final", "final/trainingset.csv", 100, "src/main/resources/final/bestModel.bin", false);
		p.prediction("OFDF-final", "final/testingset.csv", 20, "src/main/resources/final/bestModel.bin", false); 
		p.prediction("OFDF-final", "final/extrascaledtestset.csv", 20, "src/main/resources/final/bestModel.bin", false); 
		
	}
}
