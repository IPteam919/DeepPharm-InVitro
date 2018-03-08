package net.mydreamy.mlpharmaceutics.oraldisintegratingtablet;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class PredictionDisintegrationTime {

	
	//Random number generator seed, for reproducability
    public static final int seed = 12345;
    
    //Number of iterations per minibatch
    public static final int iterations = 1;
    
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 200;

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 143;
    
    //Network learning rate
    public static final double learningRate = 0.01;
    
    public static final int numInputs = 27;
    public static final int numOutputs = 1;
    public static final int numHiddenNodes = 300;
    	
	public static void main(String[] args) {
		Logger log = LoggerFactory.getLogger(PredictionDisintegrationTime.class);


		//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReadertrain = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertrain.initialize(new FileSplit(new ClassPathResource("trainingset.csv").getFile()));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        DataSetIterator iteratortrain = new RecordReaderDataSetIterator(recordReadertrain,batchSize,27,27,true);
//        log.info("training set:" + trainningData.getFeatureMatrix().toString());
//        log.info("training set:" + trainningData.getLabels().toString());

        
        RecordReader recordReadertest = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertest.initialize(new FileSplit(new ClassPathResource("testingset.csv").getFile()));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

        DataSetIterator iteratortest = new RecordReaderDataSetIterator(recordReadertest,20,27,27,true);

 

        
        MultiLayerNetwork bestModel = null;

        try {
        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File("net.mydreamy.mlpharmaceutics.oraldisintegratingtablet\\src\\main\\resources\\bestModel.bin"));
        //	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File(getClass().getResource("/example.csv").toURI()));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        //Get the best model:
       //  bestModel =  result.getBestModel();
        
//        //Train the network on the full data set, and evaluate in periodically
//        for( int i=0; i<nEpochs; i++ ){
//            net.fit(iteratortrain);
//            iteratortrain.reset();
//        }
        
        iteratortrain.reset();
        DataSet trainningData = iteratortrain.next();
        
        // evaluation training set
        RegressionEvaluation evalTrain = new RegressionEvaluation(1);
        
        INDArray featuresTrain = trainningData.getFeatureMatrix();
        INDArray lablesTrain = trainningData.getLabels();
        
        INDArray PredictionTrain = bestModel.output(featuresTrain);
        log.info("train label set:\n" + lablesTrain.toString());
        log.info("train prediction set:\n" + PredictionTrain.toString());
        evalTrain.eval(lablesTrain, PredictionTrain);	  
        
        log.info("training set MSE is:" + String.format("%.4f", evalTrain.meanSquaredError(0)));
        log.info("training set R is:" + String.format("%.4f", evalTrain.correlationR2(0)));
        
        iteratortest.reset();
        DataSet testData = iteratortest.next();
        
        // evluation test set
        RegressionEvaluation evalTest = new RegressionEvaluation(1);
        
        INDArray featuresTest = testData.getFeatureMatrix();
    //    log.info("featuresTest" + featuresTest.shapeInfoToString());
     //   log.info("\n" + featuresTest.toString());

        INDArray lablesTest = testData.getLabels();
        
        
//        log.info(evalTest.stats());
        INDArray PredictionTest = bestModel.output(featuresTest);
        
       log.info("test label value: \n" + lablesTest.toString());
        log.info("test prediction value: \n" + PredictionTest.toString());

        evalTest.eval(lablesTest, PredictionTest);	  
        
       log.info("testing set MSE is: " + String.format("%.4f", evalTest.meanSquaredError(0))); 
       log.info("testing set R is: " + String.format("%.4f", evalTest.correlationR2(0)));
	}


}
