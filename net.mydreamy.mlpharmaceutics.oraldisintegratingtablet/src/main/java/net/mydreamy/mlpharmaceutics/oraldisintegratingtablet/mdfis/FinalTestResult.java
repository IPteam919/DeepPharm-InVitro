package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.mdfis;

import net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.base.Prediction;

public class FinalTestResult {
	
	public static void main(String[] args) {
		
		

		Prediction.prediction("OFDT-mdfis", "mdfis/trainingset.csv", 200, "src/main/resources/mdfis/bestModel.bin", false);
		Prediction.prediction("OFDT-mdfis", "mdfis/testingset.csv", 20, "src/main/resources/mdfis/bestModel.bin", false); 
	}
}
