package recog;

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

/**
 *
 * @author matthewsokoloff
 * 
 */

public class NeuralNet {
    
    int[][] inputMatrix;
    double[] weights_1;
    double[] weights_2;
    int numHidden;
    int numOutputs;
    int bias = 1;
    double ALPHA = 0.1;
    
    public NeuralNet(int numInput, int numHidden, int numOutputs, boolean load) throws IOException{
        this.weights_1 = new double[numHidden*(numInput+bias)];
        this.weights_2 = new double[numOutputs*(numHidden+bias)];
        this.numOutputs = numOutputs;
        this.numHidden = numHidden;
        if(!load){
            initWeights();
        }else{
            File file = new File("/Users/matthewsokoloff/Downloads/weights.csv");
            List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
            int lineCount = 0;
            for (String line : lines) {
                String[] array = line.split(",");
                for(int i = 0 ; i<array.length; i++){
                    if(lineCount==0){
                    weights_1[i] = Double.parseDouble(array[i]);    
                    }else{
                     weights_2[i] = Double.parseDouble(array[i]);      
                    }
            }
                lineCount++;
        }
        }
    }
    
    private double computeLearningRate(int iteration) {
        
        if(iteration<1000){
           return 0.05;
        }
        return 0.1;
        //return -0.000075 * iteration + 0.495;
    }
    
    private double[] oneHotEncode(int value){
        double[] encodedVector = new double[numOutputs];
        
        //For each output class
        for(int outputClass = 0; outputClass<encodedVector.length; outputClass++){
         encodedVector[outputClass] = (outputClass == value) ? 1.0 : 0.0;
        }
                
       return encodedVector;
    }
    
    public int predict(double[] inputExample){
        int classNum =0;
        double max = -100;
        double[][] pred = forwardProp(inputExample);
        
       for(int prediction = 0; prediction <numOutputs; prediction++){
          // System.out.println("Prediction Number : " + prediction + ", acutal prediction : " + pred[1][prediction]);
           if(pred[1][prediction] > max){
               
               max = pred[1][prediction];
               classNum = prediction;
           }
       }  
        return classNum;
    }
    
    public double getAccuracy(double[][] inputMatrix, int[] answers){
        double errors = 0;
        double correct = 0;
        for(int scoringExample = 0; scoringExample < inputMatrix.length; scoringExample++){
            int classPred = predict(inputMatrix[scoringExample]);
            if(answers[scoringExample] == classPred){
                System.out.println("CORRECT : " + correct + " , " + classPred + " , " + answers[scoringExample]);
                correct++;
            }else{
                System.out.println("ERROR : " +errors + " , " + classPred + " , " + answers[scoringExample]);
                errors++;
            }
        }
        return (correct/ (errors+correct));
    }

    public void train(double[][] inputMatrix, int numIters, int[] answers) throws FileNotFoundException{
        int startIter = numIters;
        while(numIters>0){
            System.out.println("Iters Remaining : " + numIters);
            backProp(inputMatrix, answers);
            //ALPHA = computeLearningRate((startIter-numIters));
            ALPHA = computeLearningRate(numIters);
            numIters--;
        } 
        
         PrintWriter br = new PrintWriter(new File("/Users/matthewsokoloff/Downloads/weights.csv"));
        StringBuilder sb = new StringBuilder();
        double[][] weights = new double[2][];
        weights[0] = weights_1;
        weights[1] = weights_2;
        
        for(double[] example : weights){    
            for (double feature : example) {
                sb.append(feature);
                sb.append(",");
            }
            sb.append("\n");
        }
        br.write(sb.toString());
        br.close(); 
    }
    
    
    private void initWeights(){
        double min = -1.0;
        double max = 1.0;
        
         Random random = new Random();
         for(int weightIndex = 0; weightIndex < weights_1.length; weightIndex++){
               double randomValue = min + (max - min) * random.nextDouble();
               weights_1[weightIndex] = randomValue;
         }
         
         for(int weightIndex = 0; weightIndex < weights_2.length; weightIndex++){
               double randomValue = min + (max - min) * random.nextDouble();
               weights_2[weightIndex] = randomValue;
         }
         
    }
    
    
    private double sigmoid(double z){
        return (1/(1+Math.exp(-z)));
    }
    
    private double sigPrime(double out){
        return ((1-out)*(out));
    }
    
    private double[][] forwardProp(double[] inputVector){
        
        //Create input data with spot for bias
        double[] inputVectorBias = new double[inputVector.length + bias];
        
        for(int i =0; i<inputVectorBias.length; i++){
            if(i == inputVector.length && bias == 1){
                //Add bias
                inputVectorBias[i] = 1.0;
            }else{
                //Add inputs
                inputVectorBias[i] = inputVector[i];
            }
        }
        
        //Vector containing a value for each output node output
        double[] outputPreds = new double[numOutputs];
        //Vector containing a value for each hidden node output
        double[] hiddenPreds = new double[numHidden+bias];
        //Matrix containing both output predictions and hidden predictions so both can be returned
        double[][] returnValues = new double[2][];
        
        
        //For each of the hidden nodes                ----- Bias isn't getting a calculated input----
        for(int hiddenNodeIndex = 0; hiddenNodeIndex<(hiddenPreds.length-bias); hiddenNodeIndex++){ // 0-10
            double sum = 0;
            //For each of the weights connected to the current hidden node
            for(int inputNode = 0; inputNode < (inputVectorBias.length); inputNode++){ // 0-10
               sum += inputVectorBias[inputNode] * weights_1[inputNode+(hiddenNodeIndex*(inputVectorBias.length))];
            }
            hiddenPreds[hiddenNodeIndex] = sigmoid(sum);
            
        }
        
    //    hiddenNode + (outputNode*(numHidden+bias))
        
        
        
        //Add bias
        hiddenPreds[numHidden] = 1;
        
        //For each of the output nodes                  ----- No bias so we do not have to subtract it off ------
        for(int outputNodeIndex = 0; outputNodeIndex<outputPreds.length; outputNodeIndex++){
            double sum = 0;
            //For each of the weights connected to the current output node
            for(int hiddenNodeIndex = 0; hiddenNodeIndex < (hiddenPreds.length); hiddenNodeIndex++){
                sum += hiddenPreds[hiddenNodeIndex] * weights_2[hiddenNodeIndex+(outputNodeIndex*(/*outputPreds.length*/numHidden+bias))];
            }
            outputPreds[outputNodeIndex] = sigmoid(sum);
        }
        
        returnValues[0] = hiddenPreds;
        returnValues[1] = outputPreds;
     return returnValues;   
    }
    
    private void backProp(double[][] inputMatrix, int[] answers){
        //Initialize gradient for connections between input and hidden
        double[] grad1 = new double[this.weights_1.length];
        //Initialize gradient for connections between hidden and output
        double[] grad2 = new double[this.weights_2.length];

        
        
        //Iterate over the entire training set
        for(int trainingExample = 0; trainingExample<inputMatrix.length; trainingExample++){
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
            for(int hiddenNode = 0; hiddenNode<(numHidden+bias); hiddenNode++){
                for(int outputNode = 0; outputNode<numOutputs; outputNode++){
                    grad2[hiddenNode + (outputNode*(numHidden+bias))] += predictions[0][hiddenNode]*outputDelta[outputNode];
                }
            }
            
            //accumulate hidden deltas
            
            /*
            *
            *
            *
            *
            */
            for(int inputNode = 0; inputNode<(inputMatrix[trainingExample].length+bias); inputNode++){
                for(int hiddenNode = 0; hiddenNode</*numOutputs*/numHidden; hiddenNode++){
                    if(inputNode == inputMatrix[trainingExample].length){
                        grad1[inputNode + (hiddenNode*(inputMatrix[trainingExample].length+bias))] += 1*hiddenDelta[hiddenNode];
                    }else{
                        grad1[inputNode + (hiddenNode*(inputMatrix[trainingExample].length+bias))] += inputMatrix[trainingExample][inputNode]*hiddenDelta[hiddenNode];
                    }
                }
            }
        //End of code calculating batch gradient    
        }

        //hidden to output weight updates
        for(int hiddenNode = 0; hiddenNode<(numHidden+bias); hiddenNode++){
            for(int outputNode = 0; outputNode<numOutputs; outputNode++){
                grad2[hiddenNode + (outputNode*(numHidden+bias))] /= inputMatrix.length;
                weights_2[hiddenNode + (outputNode*(numHidden+bias))] -= grad2[hiddenNode + (outputNode*(numHidden+bias))]*ALPHA;
            }
        }

        //input to hidden weight updates
        for(int inputNode = 0; inputNode<(inputMatrix[0].length+bias); inputNode++){
            for(int hiddenNode = 0; hiddenNode</*numOutputs*/numHidden; hiddenNode++){
                        weights_1[inputNode + (hiddenNode*(inputMatrix[0].length+bias))] -= grad1[inputNode + (hiddenNode*(inputMatrix[0].length+bias))]*ALPHA;
            }
        }

    }
    
    private double[] getOutputDelta(double[] errors, double[][] predictions){
        double[] outputDelta = new double[errors.length];
        //Calculate output deltas
        for(int outputNode = 0; outputNode<numOutputs; outputNode++){
                   outputDelta[outputNode] = errors[outputNode]*sigPrime(predictions[1][outputNode]);
        }
        
       return outputDelta;
    }
    
    private double[] getHiddenDelta(double[] outputDelta, double[][] predictions){
        double[] hiddenDelta = new double[numHidden+bias];
        for(int hiddenNode = 0; hiddenNode<(numHidden+bias); hiddenNode++){
                for(int outputNode = 0; outputNode<numOutputs; outputNode++){
                    hiddenDelta[hiddenNode] += weights_2[hiddenNode + (outputNode*(numHidden+bias))] * outputDelta[outputNode];
                }
                hiddenDelta[hiddenNode] *= sigPrime(predictions[0][hiddenNode]);
            }
        return hiddenDelta;
    }
        
    private double[] getError(double[] preds, double[] answers){
        double[] error = new double[preds.length];
        for(int nodeNum = 0; nodeNum<preds.length; nodeNum++){
            error[nodeNum] = (preds[nodeNum] - answers[nodeNum]);
        }
        
        return error;
    }
    
    public static void main(String[] args) throws IOException{
        NeuralNet nnn = new NeuralNet(784, 10, 3, false); 

        Random rand = new Random();
        double[][] trainingData = new double[8000][784];
        double[][] testingData = new double[4993][784];
        int[] actual = new int[12993];
        File file = new File("/Users/matthewsokoloff/Downloads/123mnist.csv");
        List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
            int lineCount = 0;
            for (String line : lines) {
                if(lineCount != 0 && lineCount<12993){
                String[] array = line.split(",");
                actual[lineCount] = Integer.parseInt(array[0]);
                for(int i = 1 ; i<array.length; i++){
                    if(lineCount>=8000){
                        int arrayIndex = lineCount-8000;
                        testingData[arrayIndex][i-1] = Double.parseDouble(array[i]);
                    }else{
                        trainingData[lineCount][i-1] = Double.parseDouble(array[i]);
                    }
                }            
            }
                lineCount++;
        }
            
         int[] trainingLabels = Arrays.copyOfRange(actual, 0, 8000);  
         int[] testingLabels = Arrays.copyOfRange(actual, 8000, 12993);  
         
         nnn.train(trainingData, 200, trainingLabels);
         
       
         System.out.println(nnn.getAccuracy(testingData, testingLabels));
        
    } 

}
