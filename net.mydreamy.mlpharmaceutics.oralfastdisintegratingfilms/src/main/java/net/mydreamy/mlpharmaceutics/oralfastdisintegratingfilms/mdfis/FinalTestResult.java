package net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.mdfis;

import net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		

		//best Model testing
		Prediction.prediction("OFDF", "mdfis/trainingset.csv", 100, "src/main/resources/mdfis/bestModel.bin", false);
		Prediction.prediction("OFDF", "mdfis/testingset.csv", 20, "src/main/resources/mdfis/bestModel.bin", false); 
		
	}
}
