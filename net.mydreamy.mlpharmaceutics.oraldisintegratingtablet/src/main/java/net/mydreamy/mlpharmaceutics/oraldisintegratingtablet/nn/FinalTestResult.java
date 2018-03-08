package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.nn;

import net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.base.Prediction;

public class FinalTestResult {
	
	public static void main(String[] args) {
		
		Prediction.prediction("OFDT", "nn/trainingset.csv", 200, "src/main/resources/nn/bestModel.bin", true);
		Prediction.prediction("OFDT", "nn/testingset.csv", 20, "src/main/resources/nn/bestModel.bin", true); 
		Prediction.prediction("OFDT", "nn/extrascaledtestset.csv", 20, "src/main/resources/nn/bestModel.bin", true); 
	}
}
