/*
 * Anyone can use, I hope you enjoy!
 */

import java.io.File;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Hashtable;

/**
 *
 * @author matthewsokoloff
 * This file contains an implementation of a neural network
 * The main class instantiates a the neural network to perform digit classification.
 * While this is configured for digit classification, it can be used for any classification problem.
 */

public class NeuralNet {

    int[][] inputMatrix;
    double[] weights_1;
    double[] weights_2;
    int numHidden;
    int numOutputs;
    int bias = 1;
    double ALPHA = 0.1;

    public NeuralNet(int numInput, int numHidden, int numOutputs, boolean load) throws IOException {
        this.weights_1 = new double[numHidden * (numInput + bias)];
        this.weights_2 = new double[numOutputs * (numHidden + bias)];
        this.numOutputs = numOutputs;
        this.numHidden = numHidden;
        if (!load) {
            initWeights();
        } else {
            File file = new File("../lib/weights.csv");
            List < String > lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
            int lineCount = 0;
            for (String line: lines) {
                String[] array = line.split(",");
                for (int i = 0; i < array.length; i++) {
                    if (lineCount == 0) {
                        weights_1[i] = Double.parseDouble(array[i]);
                    } else {
                        weights_2[i] = Double.parseDouble(array[i]);
                    }
                }
                lineCount++;
            }
        }
    }

    /*
     * @iteration : Current iterations number
     * Returns the learning rate used to apply gradients
     */
    private double computeLearningRate(int iteration) {
        if (iteration < 1000) {
            return 0.05;
        }
        return 0.1;
    }

    /*
     * @value : the index of the correct output node
     * Returns the one hot encoded value
     */
    private double[] oneHotEncode(int value) {
        double[] encodedVector = new double[numOutputs];

        //For each output class
        for (int outputClass = 0; outputClass < encodedVector.length; outputClass++) {
            encodedVector[outputClass] = (outputClass == value) ? 1.0 : 0.0;
        }
        return encodedVector;
    }

    /*
     * @inputExample : Array of pixel intensities
     * Returns the predicted class number
     */
    public int predict(double[] inputExample) {
        int classNum = 0;
        double max = -100;
        double[][] pred = forwardProp(inputExample);

        for (int prediction = 0; prediction < numOutputs; prediction++) {
            if (pred[1][prediction] > max) {
                max = pred[1][prediction];
                classNum = prediction;
            }
        }
        return classNum;
    }

    /*
     * @inputMatrix : A matrix with each row containing a single example
     * @answers : List of the correct output classess corresponding to each row in the inputMatrix
     * Returns the accuracy representing the performance of the model on the inputMatrix data
     */
    public double getAccuracy(double[][] inputMatrix, int[] answers) throws IOException {
    	
        double errors = 0;
        double correct = 0;
        
        PrintWriter br = new PrintWriter(new File("../lib/errors.csv"));
        StringBuilder sb = new StringBuilder();
        
        for (int scoringExample = 0; scoringExample < inputMatrix.length; scoringExample++) {
            int classPred = predict(inputMatrix[scoringExample]);
            if (answers[scoringExample] == classPred) {
                System.out.println("CORRECT : " + correct + " , " + classPred + " , " + answers[scoringExample]);
                correct++;
            } else {
                System.out.println("ERROR : " + errors + " , " + classPred + " , " + answers[scoringExample]);
                for(double element : inputMatrix[scoringExample]){
                	sb.append((int) element);
                	sb.append(",");
                }
                sb.append(classPred);
                sb.append(",");
                sb.append(answers[scoringExample]);
                sb.append("\n");
                errors++;
            }
       
        }
        br.write(sb.toString());
        br.close();
        return (correct / (errors + correct));
    }

    /*
     * @inputMatrix : A matrix with each row containing a single example
     * @numIters : Number of iterations to train the model
     * @answers : List of the correct output classess corresponding to each row in the inputMatrix
     * Updates the weights and then saves them to a file.
     */
    public void train(double[][] inputMatrix, int numIters, int[] answers) throws FileNotFoundException {
        int startIter = numIters;
        while (numIters > 0) {
            System.out.println("Iters Remaining : " + numIters);
            backProp(inputMatrix, answers);
            //ALPHA = computeLearningRate((startIter-numIters));
            ALPHA = computeLearningRate(numIters);
            numIters--;
        }

        PrintWriter br = new PrintWriter(new File("../lib/weights.csv"));
        StringBuilder sb = new StringBuilder();
        double[][] weights = new double[2][];
        weights[0] = weights_1;
        weights[1] = weights_2;

        for (double[] example: weights) {
            for (double feature: example) {
                sb.append(feature);
                sb.append(",");
            }
            sb.append("\n");
        }
        br.write(sb.toString());
        br.close();
    }

    /*
     * Initializes the models weights to follow a random uniform distribution
     */
    private void initWeights() {
        double min = -1.0;
        double max = 1.0;

        Random random = new Random();
        for (int weightIndex = 0; weightIndex < weights_1.length; weightIndex++) {
            double randomValue = min + (max - min) * random.nextDouble();
            weights_1[weightIndex] = randomValue;
        }

        for (int weightIndex = 0; weightIndex < weights_2.length; weightIndex++) {
            double randomValue = min + (max - min) * random.nextDouble();
            weights_2[weightIndex] = randomValue;
        }
    }

    /*
     * @z : The dot product of the weights connected to this node and the input activations
     * Returns the activated node value
     */
    private double sigmoid(double z) {
        return (1 / (1 + Math.exp(-z)));
    }

    /*
     * @out : The activation of this node
     * Returns the derivative of the node's activation
     */
    private double sigPrime(double out) {
        return ((1 - out) * (out));
    }

    /*
     * @inputVector : A single example for the neural network
     * Returns the hidden and output activations
     */
    private double[][] forwardProp(double[] inputVector) {

        //Create input data with spot for bias
        double[] inputVectorBias = new double[inputVector.length + bias];

        for (int i = 0; i < inputVectorBias.length; i++) {
            if (i == inputVector.length && bias == 1) {
                //Add bias
                inputVectorBias[i] = 1.0;
            } else {
                //Add inputs
                inputVectorBias[i] = inputVector[i];
            }
        }

        //Vector containing a value for each output node output
        double[] outputPreds = new double[numOutputs];
        //Vector containing a value for each hidden node output
        double[] hiddenPreds = new double[numHidden + bias];
        //Matrix containing both output predictions and hidden predictions so both can be returned
        double[][] returnValues = new double[2][];

        //For each of the hidden nodes              
        for (int hiddenNodeIndex = 0; hiddenNodeIndex < (hiddenPreds.length - bias); hiddenNodeIndex++) {
            double sum = 0;
            //For each of the weights connected to the current hidden node
            for (int inputNode = 0; inputNode < (inputVectorBias.length); inputNode++) {
                sum += inputVectorBias[inputNode] * weights_1[inputNode + (hiddenNodeIndex * (inputVectorBias.length))];
            }
            hiddenPreds[hiddenNodeIndex] = sigmoid(sum);
        }
        
        //Add bias
        hiddenPreds[numHidden] = 1;

        //For each of the output nodes                 
        for (int outputNodeIndex = 0; outputNodeIndex < outputPreds.length; outputNodeIndex++) {
            double sum = 0;
            //For each of the weights connected to the current output node
            for (int hiddenNodeIndex = 0; hiddenNodeIndex < (hiddenPreds.length); hiddenNodeIndex++) {
                sum += hiddenPreds[hiddenNodeIndex] * weights_2[hiddenNodeIndex + (outputNodeIndex * (numHidden + bias))];
            }
            outputPreds[outputNodeIndex] = sigmoid(sum);
        }

        returnValues[0] = hiddenPreds;
        returnValues[1] = outputPreds;
        return returnValues;
    }

    /*
     * @inputMatrix : A matrix with each row containing a single example
     * @answers : List of the correct output classess corresponding to each row in the inputMatrix
     * Computes and applies the gradients
     */
    private void backProp(double[][] inputMatrix, int[] answers) {
        //Initialize gradient for connections between input and hidden
        double[] grad1 = new double[this.weights_1.length];
        //Initialize gradient for connections between hidden and output
        double[] grad2 = new double[this.weights_2.length];

        //Iterate over the entire training set
        for (int trainingExample = 0; trainingExample < inputMatrix.length; trainingExample++) {
            //One hot encode answer
            double[] answerVector = oneHotEncode(answers[trainingExample]);
            //Get activations for hidden and output layers
            double[][] predictions = forwardProp(inputMatrix[trainingExample]);
            //Calculate each output nodes' error
            double[] errors = getError(predictions[1], answerVector);
            //Calculate output delta
            double[] outputDelta = getOutputDelta(errors, predictions);
            //Calculate hidden delta
            double[] hiddenDelta = getHiddenDelta(outputDelta, predictions);
            
            //accumulate output deltas
            for (int hiddenNode = 0; hiddenNode < (numHidden + bias); hiddenNode++) {
                for (int outputNode = 0; outputNode < numOutputs; outputNode++) {
                    grad2[hiddenNode + (outputNode * (numHidden + bias))] += predictions[0][hiddenNode] * outputDelta[outputNode];
                }
            }

            //accumulate hidden deltas
            for (int inputNode = 0; inputNode < (inputMatrix[trainingExample].length + bias); inputNode++) {
                for (int hiddenNode = 0; hiddenNode < /*numOutputs*/ numHidden; hiddenNode++) {
                    if (inputNode == inputMatrix[trainingExample].length) {
                        grad1[inputNode + (hiddenNode * (inputMatrix[trainingExample].length + bias))] += 1 * hiddenDelta[hiddenNode];
                    } else {
                        grad1[inputNode + (hiddenNode * (inputMatrix[trainingExample].length + bias))] += inputMatrix[trainingExample][inputNode] * hiddenDelta[hiddenNode];
                    }
                }
            }
            //End of code calculating batch gradient    
        }

        //hidden to output weight updates
        for (int hiddenNode = 0; hiddenNode < (numHidden + bias); hiddenNode++) {
            for (int outputNode = 0; outputNode < numOutputs; outputNode++) {
                grad2[hiddenNode + (outputNode * (numHidden + bias))] /= inputMatrix.length;
                weights_2[hiddenNode + (outputNode * (numHidden + bias))] -= grad2[hiddenNode + (outputNode * (numHidden + bias))] * ALPHA;
            }
        }

        //input to hidden weight updates
        for (int inputNode = 0; inputNode < (inputMatrix[0].length + bias); inputNode++) {
            for (int hiddenNode = 0; hiddenNode < /*numOutputs*/ numHidden; hiddenNode++) {
                weights_1[inputNode + (hiddenNode * (inputMatrix[0].length + bias))] -= grad1[inputNode + (hiddenNode * (inputMatrix[0].length + bias))] * ALPHA;
            }
        }
    }
    
    
    /*
     * @errors : The error for each output node
     * @predictions : The predictions for the output and hidden nodes 
     * Returns an array containing the output deltas
     */
    private double[] getOutputDelta(double[] errors, double[][] predictions) {
        double[] outputDelta = new double[errors.length];
        //Calculate output deltas
        for (int outputNode = 0; outputNode < numOutputs; outputNode++) {
            outputDelta[outputNode] = errors[outputNode] * sigPrime(predictions[1][outputNode]);
        }

        return outputDelta;
    }

    /*
     * @outputDelta : The output deltas
     * @predictions : The predictions for the output and hidden nodes 
     * Returns an array containing the hidden deltas
     */
    private double[] getHiddenDelta(double[] outputDelta, double[][] predictions) {
        double[] hiddenDelta = new double[numHidden + bias];
        for (int hiddenNode = 0; hiddenNode < (numHidden + bias); hiddenNode++) {
            for (int outputNode = 0; outputNode < numOutputs; outputNode++) {
                hiddenDelta[hiddenNode] += weights_2[hiddenNode + (outputNode * (numHidden + bias))] * outputDelta[outputNode];
            }
            hiddenDelta[hiddenNode] *= sigPrime(predictions[0][hiddenNode]);
        }
        return hiddenDelta;
    }

    /*
     * @preds : The predictions of the network
     * @answers : The correct one hot encoded vector  
     * Returns an error vector
     */
    private double[] getError(double[] preds, double[] answers) {
        double[] error = new double[preds.length];
        for (int nodeNum = 0; nodeNum < preds.length; nodeNum++) {
            error[nodeNum] = (preds[nodeNum] - answers[nodeNum]);
        }
        return error;
    }

    public static void main(String[] args) throws IOException {
    	Hashtable<String, String> commandArgs = new Hashtable<String, String>();
    	commandArgs.put("-input_nodes", "784");
    	commandArgs.put("-hidden_nodes", "25");
    	commandArgs.put("-output_classes", "3");
    	commandArgs.put("-num_iters", "200");
    	commandArgs.put("-input_path", "../lib/012mnist.csv");
    	commandArgs.put("-training_examples", "8000");
    	commandArgs.put("-testing_examples", "4993");
    	commandArgs.put("-load_saved", "false");
    	
    	String[] commandLineArgs = {"-input_nodes","-hidden_nodes", "-output_classes","-num_iters","-input_path","-training_examples","-testing_examples","-load_saved","-help"};
    	    	
    	if(args.length > 0) {
			if(args[0].equals("-help")) {
				System.out.println(
						   "-input_nodes       : Number of input nodes \n"
						+  "-hidden_nodes      : Number of hidden nodes\n"
						+  "-output_classes    : Number of output classes\n"
						+  "-num_iters         : Number of iterations to run back propogation\n"
						+  "-input_path        : CSV to be used for training and testing data\n"
						+  "-training_examples : Number of examples in input file to use for training\n"
						+  "-testing_examples  : Number of examples in input file to use for testing\n"
						+  "-load_saved        : Boolean indicating whether or not to load saved weights");
			System.exit(0);
	    	}
    		
			for(int index = 0; index < (args.length - 1); index++){
				commandArgs.put(args[index], args[index+1]);
			}
    	}
    	
    	int inputNodes = Integer.parseInt(commandArgs.get("-input_nodes")); 
    	int hiddenNodes = Integer.parseInt(commandArgs.get("-hidden_nodes"));
    	int numOutputClasses = Integer.parseInt(commandArgs.get("-output_classes"));
    	int numIters = Integer.parseInt(commandArgs.get("-num_iters"));
    	String inputDataPath = commandArgs.get("-input_path");
    	int trainingSetSize = Integer.parseInt(commandArgs.get("-training_examples"));
    	int testingSetSize = Integer.parseInt(commandArgs.get("-testing_examples"));
    	boolean loadSavedModel = Boolean.parseBoolean(commandArgs.get("-load_saved"));
    	
				    	
    	//Instantiate the neuarl network class
        NeuralNet nn = new NeuralNet(inputNodes, hiddenNodes, numOutputClasses, loadSavedModel);

        //Create plaeholders for training and testing set (66/33 split)
        double[][] trainingData = new double[trainingSetSize][inputNodes];
        double[][] testingData = new double[testingSetSize][inputNodes];
        int[] actual = new int[trainingSetSize+testingSetSize];
        File file = new File(inputDataPath);
        List < String > lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
        int lineCount = 0;
        for (String line: lines) {
            if (lineCount != 0 && lineCount < (trainingSetSize+testingSetSize)) {
                String[] array = line.split(",");
                actual[lineCount] = Integer.parseInt(array[0]);
                for (int i = 1; i < array.length; i++) {
                    if (lineCount >= trainingSetSize) {
                        int arrayIndex = lineCount - trainingSetSize;
                        testingData[arrayIndex][i - 1] = Double.parseDouble(array[i]);
                    } else {
                        trainingData[lineCount][i - 1] = Double.parseDouble(array[i]);
                    }
                }
            }
            lineCount++;
        }

        int[] trainingLabels = Arrays.copyOfRange(actual, 0, trainingSetSize);
        int[] testingLabels = Arrays.copyOfRange(actual, trainingSetSize, (trainingSetSize+testingSetSize));
        //Train the network
        nn.train(trainingData, numIters, trainingLabels);
        //Print the error
        System.out.println(nn.getAccuracy(testingData, testingLabels));

    }

}
