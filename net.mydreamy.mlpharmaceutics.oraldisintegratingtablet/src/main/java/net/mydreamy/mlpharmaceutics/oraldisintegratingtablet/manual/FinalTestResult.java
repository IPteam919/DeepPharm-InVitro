package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.manual;

import net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.base.Prediction;

public class FinalTestResult {
	
	public static void main(String[] args) {
		
		

		Prediction.prediction("OFDT-manual", "manual/trainsetapiparams.csv", 200, "src/main/resources/manual/latestModel.bin", false);
		Prediction.prediction("OFDT-manual", "manual/testsetapiparams.csv", 20, "src/main/resources/manual/latestModel.bin", false); 
	}
}
