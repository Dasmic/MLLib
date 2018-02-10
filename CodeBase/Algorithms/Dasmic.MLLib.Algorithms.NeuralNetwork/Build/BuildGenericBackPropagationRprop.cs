using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    /// <summary>
    /// Build a Generic Backpropagation Network
    /// 
    /// The number of layers and units are specified by the user using SetParameters function
    /// </summary>
    public class BuildGenericBackPropagationRprop : BuildBackPropagationBase
    {
       
        protected IActivationFunction _activationFunctionOutputLayer;
        protected double _factorIncrease;
        protected double _factorDecrease;
        protected double _maxUpdateFactor;
        protected double _minUpdateFactor;


        public BuildGenericBackPropagationRprop()
        {
            _model = new ModelBackPropagationBase();
            _activationFunctionOutputLayer = new Sigmoid();
            _factorDecrease = .5;
            _factorIncrease = 1.2;
            _alpha = .1;
            _maxUpdateFactor = Double.MaxValue;
            _minUpdateFactor = Double.MinValue;          
        }


        /// <summary>
        /// Set model building Parameters:
        ///    
        /// <para>0 - Mode;default=0:0 - Regression (one output unit): 1 - Classification (multiple output units)</para>
        /// <para>1 - Number of hidden layers; default=1</para>
        /// <para>2 - Stopping threshold;default=.01</para>
        /// <para>3 - Maximum Epoch/Iterations; default=1000</para>
        /// <para>4 - Initial Update Value;default=.1</para>
        /// <para>5 - Increase (Multiplication) Factor;default=1.2</para>
        /// <para>6 - Decrease (Multiplication) Factor;default=.5</para>        
        /// <para>7 - Maximum Update Value;default=unlimited</para>
        /// <para>8 - Minimum Update Value;default=unlimited</para>                
        /// </summary> 
        public override void
            SetParameters(params double[] values)
        {
            int idx = 0;
            if (values.Length > 0)
                if (values[idx] != double.NaN)
                    _mode = (ModelBackPropagationBase.EnumMode)(int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _model.SetNumberOfLayers((int)(values[idx]) + 1); //1 for output layer
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _threshold = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _noOfEpoch = (int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _alpha = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _factorIncrease = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _factorDecrease = (double)values[idx];           
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _maxUpdateFactor = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _minUpdateFactor = (double)values[idx];           
        }
      

        /// <summary>
        /// Adds the Activation Function for Output Layer
        /// </summary>
        /// <param name="activationFunction"></param>
        public void SetOutputLayerActivationFunction(IActivationFunction activationFunction)
        {
            //Add 1 for output layer, which Input layer is added by default
            _activationFunctionOutputLayer = activationFunction;
        }

        /// <summary>
        /// Builds the model. Uses constant Learning Rate
        /// </summary>
        /// <param name="trainingData"></param>
        /// <param name="attributeHeaders"></param>
        /// <param name="indexTargetAttribute"></param>
        /// <returns></returns>
        public override Common.MLCore.ModelBase
           BuildModel(double[][] trainingData,
                        string[] attributeHeaders,
                        int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            //Proceed with computations
            int noOfHiddenLayers = _model.GetNumberOfLayers() - 2;

            _model.SetValues(_missingValue,
                                   _indexTargetAttribute,
                                   _trainingData.Length - 1);
            _model.Mode = _mode; //Classification or Regression 
                                 //Find out number of categories
                                 //Get unique value in Target Attribute
            _model.TargetValues =
                GetNumberOfTargetValues(_mode, trainingData, indexTargetAttribute);
            _noOfUnitsOutputLayer = _model.TargetValues.Length;


            //Add Input Layer
            _model.SetInputLayer(_trainingData, _noOfAttributes);


            //Sanity Check - Verify if all layers except output layer are in place         
            if (VerifyLayers(_model))
                throw new NeuralNetworkConfigurationNotReady();

            //Initialize weights and setup Upstream Layer for Hidden Layers
            Parallel.For(1, noOfHiddenLayers + 1,
                            new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                            idxLayer =>
            {
                _model.GetLayer(idxLayer).SetUpstreamLayer(_model.GetLayer(idxLayer - 1));
                _model.GetLayer(idxLayer).InitializeWeights();
            });

            //Add Output Layer
            _model.AddLayer(noOfHiddenLayers + 1, _noOfUnitsOutputLayer, //Output Layer
            _weightBaseValue, _activationFunctionOutputLayer,
            _maxParallelThreads);
            
            //Sanity Check - Verify if all Upstream layers are in place - do it after adding output layer            
            if (VerifyUpstreamLayers(_model))
                throw new NeuralNetworkConfigurationNotReady();

            //Start the training
            //Do not parallelize as computation has to be sequential
            //double[] computedValuesOutput = new double[_model.GetOutputUnitCount()];
            double[][] computedValuesAllLayers = new double[_model.GetNumberOfLayers()][]; //Index 0 will be null
            double[][] errorLayers = new double[_model.GetNumberOfLayers()][];
            //double[][] weightUpdates = new double[_model.GetNumberOfLayers()][];
            double[][] upStreamUnitValues = new double[_model.GetNumberOfLayers()][]; //Leave left most value as 0
            //sumGradient[IdxCurrentValue][IdxLayer][IdxLayer][IdxWeight]
            double[][][] factors = new double[_model.GetNumberOfLayers()][][];
            double[][][] deltas = new double[_model.GetNumberOfLayers()][][];
            double[][][][] sumGradients = new double[2][][][];
            sumGradients[0] = new double[_model.GetNumberOfLayers()][][];
            sumGradients[1] = new double[_model.GetNumberOfLayers()][][];

            //Initialize Layer specific data, and init weights
            InitVariables(computedValuesAllLayers, errorLayers, sumGradients,factors,deltas);
        
            int idxOutputLayer = _model.GetNumberOfLayers() - 1; //Single Input layer is not included, and this is only for compute and error arrays

            //Start iterations
            for (int epoch = 0; epoch < _noOfEpoch; epoch++) //Do not parallelize
            {
                double totalThreshold=0;
                for (long row = 0; row < _noOfDataSamples; row++) //For each data sample
                {
                    
                    ComputeUpstreamValue(upStreamUnitValues, row);
                    totalThreshold += ComputeError(computedValuesAllLayers, errorLayers, idxOutputLayer, row);
                    ComputeSumGradient(computedValuesAllLayers, errorLayers, sumGradients,upStreamUnitValues);                 
                } //For row            
                if (totalThreshold < _threshold) break;//Stopping condition reached
                UpdateWeights(sumGradients,upStreamUnitValues, factors,deltas);                
            }//epoch

            return _model;
        }

        protected void InitVariables(double[][] computedValuesAllLayers,
                                    double[][] errorLayers,                                    
                                    double[][][][] sumGradient,
                                    double[][][] factors,
                                    double[][][] delta)
        {
            //Note that all parameters have index 0 for input layer
            //for(int idxLayer=1;idxLayer<errorLayers.Length;idxLayer++)
            //{
            Parallel.For(1, errorLayers.Length,
                              new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                              idxLayer => 
            { 
                errorLayers[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)]; //Input layer is not included in original index
                computedValuesAllLayers[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)];                
                factors[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)][]; //Extra unit for bias term
                delta[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)][]; //Extra unit for bias term

                sumGradient[0][idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)][];
                sumGradient[1][idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)][];

                //Update weights and gradient sum
                for (int idxUnits = 0; idxUnits < _model.GetNumberOfUnits(idxLayer); idxUnits++)
                {
                    sumGradient[0][idxLayer][idxUnits] = new double[_model.GetNumberOfUpstreamUnits(idxLayer)+1]; //One extra for bias
                    sumGradient[1][idxLayer][idxUnits] = new double[_model.GetNumberOfUpstreamUnits(idxLayer)+1];
                    factors[idxLayer][idxUnits] = new double[_model.GetNumberOfUpstreamUnits(idxLayer)+1];
                    delta[idxLayer][idxUnits] = new double[_model.GetNumberOfUpstreamUnits(idxLayer) + 1];

                    for (int idxUpUnits = 0; idxUpUnits <= _model.GetNumberOfUpstreamUnits(idxLayer); idxUpUnits++) //Add extra for bias term
                    {
                        sumGradient[0][idxLayer][idxUnits][idxUpUnits] = 0;
                        factors[idxLayer][idxUnits][idxUpUnits] = _alpha;
                        delta[idxLayer][idxUnits][idxUpUnits] = 0;
                    }
                }
            });

        } //InitVariables

        protected void ComputeUpstreamValue(double[][] upStreamUnitValues,long row)
        {
            //_model.GetLayer(0)

            //for (int idxLayer = 1; idxLayer < _model.GetNumberOfLayers(); idxLayer++)
            Parallel.For(1, _model.GetNumberOfLayers(),
                    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idxLayer =>
            {
                //Compute values of upstream units, this will preent repetitive computations                        
                long noOfUpstreamUnits = _model.GetNumberOfUpstreamUnits(idxLayer);
                //double[] upStreamUnitValue = new double[noOfUpstreamUnits];
                upStreamUnitValues[idxLayer] = new double[noOfUpstreamUnits];

                // For each upstream unit in layer, Can be parallelized
                for (int idxUpUnit = 0; idxUpUnit < noOfUpstreamUnits; idxUpUnit++)
                {
                    //upStreamUnitValue[idxUpUnit] =
                    upStreamUnitValues[idxLayer][idxUpUnit] =
                            _model.GetUpstreamUnitValue(idxLayer, idxUpUnit, row);
                }
            });
        }

        /// <summary>
        /// Folllow ok - tk since weight is computed differently
        /// </summary>
        /// <param name="computedValuesAllLayers"></param>
        /// <param name="expectedOutput"></param>
        /// <returns></returns>
        protected override double GetOutputUnitDifference(double computedValuesAllLayers, double expectedOutput)
        {
            //returns ok- tk
            return computedValuesAllLayers - expectedOutput;
        }

        protected void ComputeSumGradient(double[][] computedValuesAllLayers,
                                   double[][] errorLayers,
                                   double[][][][] sumGradient,
                                   double[][] upStreamUnitValues)

        {
            //--- Compute sumGradient
            //for (int idxLayer = 1; idxLayer < _model.GetNumberOfLayers(); idxLayer++)
            Parallel.For(1, _model.GetNumberOfLayers(),
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idxLayer =>
            {
                //Compute values of upstream units, this will preent repetitive computations                        
                long noOfUpstreamUnits = _model.GetNumberOfUpstreamUnits(idxLayer);
                
                //For each unit in current layer
                for (int idxUnit = 0;
                                idxUnit < _model.GetNumberOfUnits(idxLayer);
                                    idxUnit++)
                {
                    //Do for each weight in the unit
                    //Bias will be computed in same value
                    //double delta = 0;
                    //wji =  wji + dwji

                    for (int idxUpUnit = 0; idxUpUnit < noOfUpstreamUnits;
                                    idxUpUnit++)
                    {
                        /*delta = _learningRate *
                                        errorLayers[idxLayer - 1][idxUnit] *
                                        upStreamUnitValue[idxUpUnit];*/
                        //PROBLEM IN LOGIC HERE
                        sumGradient[1][idxLayer][idxUnit][idxUpUnit] +=
                            errorLayers[idxLayer][idxUnit]
                                    * upStreamUnitValues[idxLayer][idxUpUnit];                       
                    }
                    
                    //Bias term
                    sumGradient[1][idxLayer][idxUnit][noOfUpstreamUnits] +=
                            errorLayers[idxLayer][idxUnit];
                    //Update for bias term
                    /*delta = _learningRate *
                                       errorLayers[idxLayer - 1][idxUnit] * 1.0;

                    _model.SetWeight(idxLayer, noOfUpstreamUnits, idxUnit,
                                _model.GetWeight(idxLayer, noOfUpstreamUnits, idxUnit)
                                + delta);*/
                }
                //Print values
            }); //idxLayer
        }

        
        protected void UpdateWeights(double[][][][] sumGradient,
                                    double[][] upStreamUnitValues,
                                    double[][][] factors,
                                    double[][][] deltas)
        {
            //Update weights after all training samples
            //for (int idxLayer = 1; idxLayer < _model.GetNumberOfLayers(); idxLayer++)
            Parallel.For(1, _model.GetNumberOfLayers(),
                    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idxLayer =>
            {
                //Compute values of upstream units, this will preent repetitive computations                        
                long noOfUpstreamUnits = _model.GetNumberOfUpstreamUnits(idxLayer);
                
                //For each unit in current layer
                for (int idxUnit = 0;
                                idxUnit < _model.GetNumberOfUnits(idxLayer);
                                    idxUnit++)
                {
                    //Do for each weight in the unit
                    //Bias will be computed in same value 
                    //Do update here
                    for (int idxUpUnit = 0; idxUpUnit <= noOfUpstreamUnits; //Last is for Bias term
                                    idxUpUnit++)
                    {
                        double delta = 0;
                        double mulValue = sumGradient[0][idxLayer][idxUnit][idxUpUnit]
                            * sumGradient[1][idxLayer][idxUnit][idxUpUnit];

                        if (mulValue >0 )
                        {
                            factors[idxLayer][idxUnit][idxUpUnit]
                                = factors[idxLayer][idxUnit][idxUpUnit] * _factorIncrease;

                            if (factors[idxLayer][idxUnit][idxUpUnit] > _maxUpdateFactor)
                                factors[idxLayer][idxUnit][idxUpUnit] = _maxUpdateFactor;

                            deltas[idxLayer][idxUnit][idxUpUnit] =  -Math.Sign(sumGradient[1][idxLayer][idxUnit][idxUpUnit]) * 
                                factors[idxLayer][idxUnit][idxUpUnit];

                            delta = deltas[idxLayer][idxUnit][idxUpUnit];
                            sumGradient[0][idxLayer][idxUnit][idxUpUnit]
                                    = sumGradient[1][idxLayer][idxUnit][idxUpUnit];
                        }
                        else if(mulValue < 0)
                        {
                            factors[idxLayer][idxUnit][idxUpUnit]
                                = factors[idxLayer][idxUnit][idxUpUnit] * _factorDecrease;

                            if (factors[idxLayer][idxUnit][idxUpUnit] < _minUpdateFactor)
                                factors[idxLayer][idxUnit][idxUpUnit] = _minUpdateFactor;

                            //Revert to old weight by reversing sign of old delta
                            delta = -deltas[idxLayer][idxUnit][idxUpUnit];

                            sumGradient[0][idxLayer][idxUnit][idxUpUnit] = 0;
                        }
                        else if (mulValue == 0)
                        {
                            deltas[idxLayer][idxUnit][idxUpUnit] = 
                                 -Math.Sign(sumGradient[1][idxLayer][idxUnit][idxUpUnit]) *
                                       factors[idxLayer][idxUnit][idxUpUnit];

                            delta = deltas[idxLayer][idxUnit][idxUpUnit];
                            sumGradient[0][idxLayer][idxUnit][idxUpUnit]
                                    = sumGradient[1][idxLayer][idxUnit][idxUpUnit];
                        }
                        //Reset sumGradient since it is end of epoch
                        //This is a very important step as otherwise this value will be inherited in next epoch
                        sumGradient[1][idxLayer][idxUnit][idxUpUnit] = 0;

                        if (delta != 0)
                            _model.SetWeight(idxLayer, idxUpUnit, idxUnit,
                                    _model.GetWeight(idxLayer, idxUpUnit, idxUnit)
                                        + delta);
                    }

                    //For bias term
                    //_model.SetWeight(idxLayer, noOfUpstreamUnits, idxUnit,
                    //            _model.GetWeight(idxLayer, noOfUpstreamUnits, idxUnit)
                    //            + delta);
                }
                //Print values
            }); //idxLayer         
        } //Function end
    }
}
