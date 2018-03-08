package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet.base;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class EvaluationMCloud {
	
	
	public static Logger log = LoggerFactory.getLogger(EvaluationMCloud.class);
	
	public static void f2(INDArray lablesTest, INDArray PredictionTest)
	{
        int size = lablesTest.size(0);
        
        INDArray allf2 = Nd4j.zeros(size);
        double correctnum = 0;
        
        for (int i = 0; i < size; i++)
        {
        	INDArray labelrow = lablesTest.getRow(i);
        	INDArray perdictrow = PredictionTest.getRow(i);
        	INDArray subrow = labelrow.sub(perdictrow);
        	INDArray subrow100 = subrow.mul(100);
        	double meanerror = Transforms.pow(subrow100, 2).mean(0).getDouble(0);
        	//log.info("mean error: " + String.valueOf(meanerror));
        	
        	double f2 = Math.log10(Math.pow((meanerror + 1), -0.5)*100)*50;
        	
        	//log.info("f2: "+ String.valueOf(f2) + "\n");
        	
        	allf2.putScalar(i, f2);
        	
        	if (f2 >= 50)
        	{
        		correctnum++;
        	}
	        }
        
        log.info(allf2.toString());
        log.info("F2 accurecy: " + (correctnum / size));
	}
	
	public static void AccuracyMAE(INDArray lablesTest, INDArray PredictionTest, double therdsold) {
		
		
		
        INDArray absErrorMatrix = Transforms.abs(lablesTest.sub(PredictionTest)).sum(1).div(PredictionTest.size(1));
        int size = absErrorMatrix.size(0);
		INDArray allAE = Nd4j.zeros(size);

        double correct = 0;
        for (int i = 0; i < size; i++)
        {
        	if (absErrorMatrix.getDouble(i) <= therdsold)
        	{
        		correct++;
        	}
        	allAE.putScalar(i, absErrorMatrix.getDouble(i));
        }
        log.info(allAE.toString());
        log.info("AccuracyMAE  <= " + therdsold*100 + "%: " + String.format("%.4f", correct/size));
	}
	
	public static void AccuracyR(INDArray lablesTest, INDArray PredictionTest,  RegressionEvaluation evalTest) {

//	     log.info(lablesTest.shapeInfoToString());
//	     log.info(PredictionTest.shapeInfoToString());
	     evalTest.eval(lablesTest, PredictionTest);	  
	     
	  //  log.info("testing set MSE is: " + String.format("%.10f", evalTest.accuracy())); 
	    log.info("R Score is: " + String.format("%.4f", evalTest.correlationR2(0)));
	}
	
	public static void AccuracyRSquare(INDArray lablesTest, INDArray PredictionTest) {
		
        Double labelmean = lablesTest.mean(0).getDouble(0);
        
        Double SSS = Transforms.pow(lablesTest.sub(PredictionTest), 2).sum(0).getDouble(0);
        Double SST = Transforms.pow(lablesTest.sub(labelmean), 2).sum(0).getDouble(0);
        Double SSG = Transforms.pow(PredictionTest.sub(labelmean), 2).sum(0).getDouble(0);
        
//        log.info("label mean: " + labelmean);
        
//        log.info("SSE: " + SSE);
//        log.info("SST: " + SST);

        log.info("R square: " + (1 - (SSS/SST)));
        log.info("R square: " + (SSG/SST));
	}
}
