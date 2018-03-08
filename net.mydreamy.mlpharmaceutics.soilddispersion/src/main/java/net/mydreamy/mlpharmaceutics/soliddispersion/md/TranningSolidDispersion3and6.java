package net.mydreamy.mlpharmaceutics.soliddispersion.md;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

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
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TranningSolidDispersion3and6 {
	public static Logger log = LoggerFactory.getLogger(TranningSolidDispersion3and6.class);

	//Random number generator seed, for reproducability
    public static final int seed = 1234567890;
    
    public static boolean isRegression = false;
    
    //Number of iterations per minibatch
    public static final int iterations = 1;
    
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 200;

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int trainsetsize = 200;
    public static final int testsetsize = 20;
    
    //Network learning rate
    public static final double learningRate = 0.01;
    
    //with api properties
    public static final int numInputs = 15;
    //
    //public static final int numInputs = 18;
    public static final int numOutputs = 2;
    public static final int numHiddenNodes = 50;
    	
	public static void main(String[] args) {
		
//		DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

//        CudaEnvironment.getInstance().getConfiguration()
////            // key option enabled
//            .allowMultiGPU(false)
////
////            // we're allowing larger memory caches
//            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
////
////            // cross-device access is used for faster model averaging over pcie
//            .allowCrossDeviceAccess(false);
        

		
		//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 1;
        String delimiter = ",";
        RecordReader recordReadertrain = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertrain.initialize(new FileSplit(new ClassPathResource("md/trainingset.csv").getFile()));

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

        DataSetIterator iteratortrain = new RecordReaderDataSetIterator(recordReadertrain,trainsetsize,numInputs,numInputs+1,true);
 //       DataSet trainningData =  iteratortrain.next();
 //       log.info("training set:\n" + trainningData.getFeatureMatrix().toString());
 //       log.info("training set:\n" + trainningData.getLabels().toString());
 //       iteratortrain.reset();

        
        RecordReader recordReadertest = new CSVRecordReader(numLinesToSkip,delimiter);
        try {
        	recordReadertest.initialize(new FileSplit(new ClassPathResource("md/testingset.csv").getFile()));

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

        DataSetIterator iteratortest = new RecordReaderDataSetIterator(recordReadertest,testsetsize,numInputs,numInputs+1,true);
//
// //       log.info("testData set:" + testData.toString());
//        
////        // Normalization
//        NormalizerStandardize normalizer = new NormalizerStandardize();
//   
        DataSet trainningSet = iteratortrain.next();
        
        if (trainningSet.hasMaskArrays()) {
            log.info("mask array shape:" + trainningSet.getLabelsMaskArray().shapeInfoToString());
            log.info("mask array: \n" + trainningSet.getLabelsMaskArray().toString());       	
        } else {
        	log.info("no mask array");
        }


        
        DataSet testSet = iteratortest.next();
//
//        log.info("train set\n" + trainningSet.getFeatureMatrix().toString());
//        log.info("train set\n" + trainningSet.getLabels().toString());
//        log.info("test set\n" + testSet.getFeatureMatrix().toString());
//        log.info("test set\n" + testSet.getLabels().toString());
//        
////        normalizer.fitLabel(false);
////        normalizer.fit(trainningData); 
////        normalizer.transform(trainningData); 
////        normalizer.transform(testData); 
//        
        iteratortrain.reset();
        iteratortest.reset();
////        log.info("training data features:\n" + trainningData.getFeatureMatrix().toString());
////        log.info("training data label:\n" + trainningData.getLabels().toString());
////        normalizer.transform(testData); 
////        log.info("training data features:\n" + testData.getFeatureMatrix().toString());
////        log.info("training data label:\n" + testData.getLabels().toString());
//        
        // Network Configuration
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.RELU)
//                .learningRateDecayPolicy(LearningRatePolicy.Step)
//                .lrPolicyDecayRate(0.1)
//                .lrPolicySteps(100000)
                .regularization(true)
                .l2(1e-3)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
              //  .dropOut(0.5)
                .updater(Updater.NESTEROVS).momentum(0.8)
              //  .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(5, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(6, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(7, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(8, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation("sigmoid")
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        
        
        List<EpochTerminationCondition> terminationconditions = new LinkedList<EpochTerminationCondition>();
  //      terminationconditions.add(new ScoreImprovementEpochTerminationCondition(10, 1E-10));
        terminationconditions.add(new BestScoreEpochTerminationCondition(0.01));
        terminationconditions.add(new MaxEpochsTerminationCondition(2000));

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        		.epochTerminationConditions(terminationconditions)
        		.scoreCalculator(new DataSetLossCalculator(iteratortest, true))
                .evaluateEveryNEpochs(100)
                .saveLastModel(true)
        		.modelSaver(new LocalFileModelSaver("src/main/resources"))
        		.build();
        
        
        
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iteratortrain);

        //Conduct early stopping training:
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

       

       //Print out the results:
        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());
        
        
        MultiLayerNetwork bestModel = null;
 //   	bestModel = result.getBestModel();
//
        try {
        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/bestModel.bin"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        MultiLayerNetwork latestModel = null;
    //	bestModel = result.getBestModel();

        try {
        	latestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/latestModel.bin"));
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
        
//        iteratortrain.reset();
 //       DataSet trainningData = iteratortrain.next();
        
 //       iteratortest.reset();
  //      DataSet testData = iteratortest.next();
        
        log.info("========================== testing =========================");
        log.info("========================== latest model =========================");
        //test on latest model
         testOnDiffModel(latestModel, trainningSet, testSet);
        
        log.info("========================== best model =========================");
        //test on best model
      //  testOnDiffModel(bestModel, trainningSet, testSet);
        testOnDiffModel(bestModel, trainningSet, testSet);

 
	}

	public static void testOnDiffModel(MultiLayerNetwork net, DataSet trainningData,  DataSet testData)
	{

    //     log.info(testData.toString());
	     
	     INDArray featuresTrain = trainningData.getFeatureMatrix();
	     INDArray lablesTrain = trainningData.getLabels();
	     
	     INDArray PredictionTrain = net.output(featuresTrain);
	    
	     INDArray trainr = lablesTrain.sub(PredictionTrain);
	         
	     int tlen = trainr.size(0);
	     
	     double tcorrectnumber = 0;
	     for (int i = 0; i < tlen; i++)
	     {
	    	 if (Math.abs(trainr.getDouble(i, 0)) <= 0.5 && Math.abs(trainr.getDouble(i, 1)) <= 0.5)
	    	 {
	    		 tcorrectnumber++;
	    	 }
	     }

	     INDArray featuresTest = testData.getFeatureMatrix();
         INDArray lablesTest = testData.getLabels();
	     
	     INDArray PredictionTest = net.output(featuresTest);
	     
	     log.info("test label value: \n" + lablesTest.toString());
	     log.info("test prediction value: \n" + PredictionTest.toString());
	     
	     INDArray testr = lablesTest.sub(PredictionTest);
         
	     int testlen = testr.size(0);
	     
	     double testcorrectnumber = 0;
	     for (int i = 0; i < testlen; i++)
	     {
	    	 if (Math.abs(testr.getDouble(i, 0)) <= 0.5 && Math.abs(testr.getDouble(i, 1)) <= 0.5)
	    	 {
	    		 testcorrectnumber++;
	    	 }
	     }
	     
	     net.printConfiguration();
	     log.info("train 3-6 correctness: " +  String.format("%.4f", tcorrectnumber/(double)tlen));
	     log.info("test 3-6 correctness: " + String.format("%.4f", testcorrectnumber/(double)testlen));
//	     log.info("testcorrectnumber:" + testcorrectnumber);
//	     log.info(testr.toString());
	     
	}
}
