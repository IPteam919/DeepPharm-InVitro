package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.dnn;

import net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.base.Prediction;

public class FinalTestResult {
	
	public static void main(String[] args) {
		
		Prediction.prediction("OFDT", "final/trainingset.csv", 200, "src/main/resources/final/bestModel.bin.bak", true);
		Prediction.prediction("OFDT", "final/testingset.csv", 20, "src/main/resources/final/bestModel.bin.bak", true); 
		Prediction.prediction("OFDT", "final/extrascaledtestset.csv", 20, "src/main/resources/final/bestModel.bin.bak", true); 
	}
}
