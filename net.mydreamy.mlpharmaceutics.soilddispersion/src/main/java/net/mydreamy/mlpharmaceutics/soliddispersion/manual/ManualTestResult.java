package net.mydreamy.mlpharmaceutics.soliddispersion.manual;

import net.mydreamy.mlpharmaceutics.soliddispersion.base.Prediction;

public class ManualTestResult {
	public static void main(String[] args) {
		
		//best Model testing
		Prediction.prediction("SD-Single", "manual/3m-code-tranning.csv", 200, "src/main/resources/manual/3mlatestModel.bin", false);
		Prediction.prediction("SD-Single", "manual/3m-code-testing.csv", 20, "src/main/resources/manual/3mlatestModel.bin", false);
		
		//best Model testing
		Prediction.prediction("SD-Single", "manual/6m-code-training.csv", 200, "src/main/resources/manual/6mlatestModel.bin", false);
		Prediction.prediction("SD-Single", "manual/6m-code-test.csv", 20, "src/main/resources/manual/6mlatestModel.bin", false);
		
	}
}
 