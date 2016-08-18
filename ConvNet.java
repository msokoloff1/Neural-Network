/*
 * Anyone can use, I hope you enjoy!
 */
package nnet;
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
    
    double[][][] imageToConvWeights;
    double[][][][] convToInputWeights;
    
    int numHidden;
    int numOutputs;
    int bias = 1;
    double ALPHA = 0.1;
    double g_sum = 0.0;
    double[] preds;
    int  imageXDim, imageYDim;
    int strideX = 5;
    int strideY = 5;
    int numFilters;
    
    double[][] inputImage;
	double[] imageToConvBias;
	double[] convToInputBias;
	int numInput;
    
    public NeuralNet(int numInput, int numHidden, int numOutputs, boolean load, int imageXDim, int imageYDim, int numFilters) throws IOException{
    	this.numInput = numInput;
    	this.numFilters = numFilters;
    	this.imageXDim = imageXDim;
    	this.imageYDim = imageYDim;
        this.weights_1 = new double[numHidden*(numInput+bias)];
        this.weights_2 = new double[numOutputs*(numHidden+bias)];
        this.imageToConvWeights = new double[numFilters][strideY][strideX];
        this.imageToConvBias = new double[numFilters];
        this.convToInputWeights = new double[numInput][numFilters][imageYDim-strideY][imageXDim-strideX];
        this.numOutputs = numOutputs;
        this.convToInputBias = new double[numInput];
        this.numHidden = numHidden;
        if(!load){
            initWeights();
        }else{
            File file = new File("weights.csv");
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
        
        if(iteration<50){
           return 0.001;
        }else{
        	return 0.01;
        }
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
    	this.inputImage = new double[this.imageYDim][this.imageXDim];
    	
    	
    	
    	
        int startIter = numIters;
        while(numIters>0){
            System.out.println("Iters Remaining : " + numIters);
            backProp(inputMatrix, answers);
            //ALPHA = computeLearningRate((startIter-numIters));
            ALPHA = computeLearningRate(numIters);
            numIters--;
        } 
        
         PrintWriter br = new PrintWriter(new File("weights.csv"));
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
        double min = -0.01;
        double max = 0.01;
        Random random = new Random();
        
        for(int z = 0; z<this.imageToConvWeights.length; z++){
        	for(int y = 0; y<this.imageToConvWeights[0].length; y++){
        		for(int x = 0; x<this.imageToConvWeights[0].length; x++){
        			this.imageToConvWeights[z][y][x] = min + (max - min) * random.nextDouble();
        		}
        	}
        	this.imageToConvBias[z] =  min + (max - min) * random.nextDouble();
        }
        
        for(int neuronNum = 0; neuronNum<this.numInput; neuronNum++){
	        for(int z = 0; z<this.convToInputWeights[0].length; z++){
	        	for(int y = 0; y<this.convToInputWeights[0][0].length; y++){
	        		for(int x = 0; x<this.convToInputWeights[0][0][0].length; x++){
	        			this.convToInputWeights[neuronNum][z][y][x] = min + (max - min) * random.nextDouble();
	        		}
	        	}
	        }
	        this.convToInputBias[neuronNum] =  min + (max - min) * random.nextDouble();
        }
        
        
        
        
        
        
         for(int weightIndex = 0; weightIndex < weights_1.length; weightIndex++){
               double randomValue = min + (max - min) * random.nextDouble();
               weights_1[weightIndex] = randomValue;
         }
         
         for(int weightIndex = 0; weightIndex < weights_2.length; weightIndex++){
               double randomValue = min + (max - min) * random.nextDouble();
               weights_2[weightIndex] = randomValue;
         }
         
    }
    
    
    
    private double relu(double z){
        //return (1/(1+Math.exp(-z)));

    	if(z<0){
    		return 0.3*(Math.exp(z)-1);
    		
    	}else{
    		if(z>10){
    			return 10;
    		}else{
    			return z;
    		}
       	}
    	//return ((2/(1+Math.exp(-2*z)))-1);
    }
    
    private double reluPrime(double out){
    	if(out<0){
    		return out+0.3;
    		
    	}else{
    		return 1;
    	}
    	
    	//return (1-(out*out));
        //return ((1-out)*(out));
    }
    
    private double sigmoid(double z){
    	return (1/(1+Math.exp(-z)));
    }
    
    private double sigPrime(double out){
    	return (1-out)*out;
    }
    
    private void prepSoftMax(double[] preds){
    	this.preds = preds;
    	double sum = 0;
    	for(double pred : preds){
    		sum+= Math.exp(pred);
    	}
    	g_sum = sum;
    }
    
    private double getSoftMax(int index){
    	
    	return Math.exp(preds[index])/g_sum;
    }
    
    
    /*
     * returns 3d array z,y,x (z=0 for 2d parts of net)
     */
    //Make private
    public double[][] forwardProp(double[][] inputImage){
    	double[][][] convPreds = new double[convToInputWeights.length][convToInputWeights[0].length][convToInputWeights[0][0].length];
    	

    	
    	for(int filterNum = 0; filterNum<this.numFilters; filterNum++){
    		double sum = 0;
    		int startX = 0, startY = 0;
    		boolean finished = false;
    		while(!finished){
    			
    			
	    		for(int currentX = startX; currentX<(startX+strideX); currentX++){
	    			for(int currentY = startY; currentY<(startY+strideY); currentY++){
	    				sum+= imageToConvWeights[filterNum][(currentY-startY)][(currentX-startX)]*inputImage[currentY][currentX];
	    			}
	    		}
	    		sum+=this.imageToConvBias[filterNum];
	    		convPreds[filterNum][startY][startX] = relu(sum);
	    		
	    		//System.out.println(sum);
	    		//System.out.println("startX " + startX + " startY " + startY);
	    		if( (startX == (convToInputWeights[0][0].length-1)) && (startY == (convToInputWeights[0].length-1))){finished = true; break;}
	    		if(startX == (convToInputWeights[0][0].length-1)){
	    			startX = 0;
	    			startY += 1;
	    		}else{
	    			startX++;
	    		}		
    		}
    	}
    		
    	
    	
    	double[] inputActivations = new double[this.numInput];
    	
    	for(int inputNeuron = 0; inputNeuron<inputActivations.length; inputNeuron++){
    		double sum = 0.0;
    		for(int filterNum = 0; filterNum <this.numFilters; filterNum++){
        		for(int y = 0; y<convPreds[0].length; y++){
        			for(int x=0; x<convPreds[0][0].length; x++){
        				 sum += convPreds[filterNum][y][x]*this.convToInputWeights[inputNeuron][filterNum][y][x];
        			}
        		}
        	}
    		sum += this.convToInputBias[inputNeuron];
    		inputActivations[inputNeuron] = relu(sum);
    	}
    	
    	
    	   	
    	
    	/*
    	 *OLD CODE BELOW 
    	 */
    	
        
      double[] inputVectorBias = new double[inputActivations.length + bias];
        
        for(int i =0; i<inputVectorBias.length; i++){
            if(i == inputActivations.length && bias == 1){
                //Add bias
                inputVectorBias[i] = 1.0;
            }else{
                //Add inputs
                inputVectorBias[i] = inputActivations[i];
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
            hiddenPreds[hiddenNodeIndex] = relu(sum);
            
        }
          
        
        //Add bias
        hiddenPreds[numHidden] = 1;
        
        //For each of the output nodes                  ----- No bias so we do not have to subtract it off ------
        for(int outputNodeIndex = 0; outputNodeIndex<outputPreds.length; outputNodeIndex++){
            double sum = 0;
            //For each of the weights connected to the current output node
            for(int hiddenNodeIndex = 0; hiddenNodeIndex < (hiddenPreds.length); hiddenNodeIndex++){
                sum += hiddenPreds[hiddenNodeIndex] * weights_2[hiddenNodeIndex+(outputNodeIndex*(numHidden+bias))];
            }

           // outputPreds[outputNodeIndex] = sum;// sigmoid(sum);
           outputPreds[outputNodeIndex] = sigmoid(sum)
        }
        
       /*prepSoftMax(outputPreds);
        for(int outputNodeIndex = 0; outputNodeIndex<outputPreds.length; outputNodeIndex++){
        	System.out.println(getSoftMax(outputNodeIndex));
        }*/
       
        returnValues[0] = hiddenPreds;
        returnValues[1] = outputPreds;
        
//===================================END OF COMMENT FOR TESTING ===================================
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
                   outputDelta[outputNode] = errors[outputNode];//*sigPrime(predictions[1][outputNode]);
        	//outputDelta[outputNode] = errors[outputNode];//*sigPrime(predictions[1][outputNode]);
        }
        
       return outputDelta;
    }
    
    private double[] getHiddenDelta(double[] outputDelta, double[][] predictions){
        double[] hiddenDelta = new double[numHidden+bias];
        for(int hiddenNode = 0; hiddenNode<(numHidden+bias); hiddenNode++){
                for(int outputNode = 0; outputNode<numOutputs; outputNode++){
                    hiddenDelta[hiddenNode] += weights_2[hiddenNode + (outputNode*(numHidden+bias))] * outputDelta[outputNode];
                }
                hiddenDelta[hiddenNode] *= reluPrime(predictions[0][hiddenNode]);
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
        NeuralNet nnn = new NeuralNet(784, 10, 3, false,28,28, 8);
        double[][] testImage = new double[28][28];
        Random rand = new Random();
        
        for(int y = 0; y<28; y++){
        	for(int x =0; x<28;x++){
        		testImage[y][x] = rand.nextDouble()*255;
        	}
        }
        
        nnn.forwardProp(testImage);
     /*   Random rand = new Random();
        double[][] trainingData = new double[8000][784];
        double[][] testingData = new double[4993][784];
        int[] actual = new int[12993];
        File file = new File("C:\\Users\\maristuser\\workspace\\nnet\\src\\mnist.csv");
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
         
         nnn.train(trainingData, 300, trainingLabels, 28,28);
         
       
         System.out.println(nnn.getAccuracy(testingData, testingLabels));
        */
    } 

}

