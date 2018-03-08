package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.nn.mf;

import net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.base.Prediction;

public class FinalTestResult {
	
	public static void main(String[] args) {
		
		Prediction.prediction("OFDT-mf", "manufacture/trainingset-mf.csv", 200, "src/main/resources/manufacture/NN.bin", true);
		Prediction.prediction("OFDT-mf", "manufacture/testingset-mf.csv", 20, "src/main/resources/manufacture/NN.bin", true); 
		Prediction.prediction("OFDT-mf", "manufacture/extrascaledtestset-mf.csv", 20, "src/main/resources/manufacture/NN.bin", true); 
		
		
//		Prediction.prediction("OFDT-mf", "manufacture/trainingset-mf.csv", 200, "src/main/resources/latestModel.bin", true);
//		Prediction.prediction("OFDT-mf", "manufacture/testingset-mf.csv", 20, "src/main/resources/latestModel.bin", true); 
//		Prediction.prediction("OFDT-mf", "manufacture/extrascaledtestset-mf.csv", 20, "src/main/resources/latestModel.bin", true); 
//		Prediction.prediction("OFDT-mf", "manufacture/trainingset-mf.csv", 200, "src/main/resources/bestModel.bin", true);
//		Prediction.prediction("OFDT-mf", "manufacture/testingset-mf.csv", 20, "src/main/resources/bestModel.bin", true); 
//		Prediction.prediction("OFDT-mf", "manufacture/extrascaledtestset-mf.csv", 20, "src/main/resources/bestModel.bin", true); 
	}
}
