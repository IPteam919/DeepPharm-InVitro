package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.md;

import net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.base.Prediction;

public class FinalTestResult {
	
	public static void main(String[] args) {
		
		Prediction.prediction("OFDT-manual", "md/trainingset.csv", 200, "src/main/resources/md/mbestModel.bin", false);
		Prediction.prediction("OFDT-manual", "md/testingset.csv", 20, "src/main/resources/md/mbestModel.bin", false); 
	}
}
