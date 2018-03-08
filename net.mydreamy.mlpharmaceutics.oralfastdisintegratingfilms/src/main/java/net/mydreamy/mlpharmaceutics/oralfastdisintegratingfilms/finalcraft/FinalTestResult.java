package net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.finalcraft;

import net.mydreamy.mlpharmaceutics.oralfastdisintegratingfilms.base.Prediction;

public class FinalTestResult {
	public static void main(String[] args) {
		
		Prediction p = new Prediction();
		//best Model testing
		p.prediction("OFDF-final-craft", "final-craft/trainset.csv", 100, "src/main/resources/final-craft/bestModel.bin", true);
		p.prediction("OFDF-final-craft", "final-craft/devset.csv", 20, "src/main/resources/final-craft/bestModel.bin", true); 
		p.prediction("OFDF-final-craft", "final-craft/testset.csv", 20, "src/main/resources/final-craft/bestModel.bin", true); 
		
	}
}
