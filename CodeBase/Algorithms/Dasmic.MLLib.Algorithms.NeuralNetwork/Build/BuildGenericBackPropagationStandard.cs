using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    /// <summary>
    /// Build a Generic Standard Backpropagation Network. Standard Back Progragation uses 
    /// a Fixed Learning Rate
    /// 
    /// The number of layers and units are specified by the user
    /// </summary>
    public class BuildGenericBackPropagationStandard : BuildBackPropagationBase
    {       
        IActivationFunction _activationFunctionOutputLayer;

        public BuildGenericBackPropagationStandard()
        {
            _model = new ModelBackPropagationBase();
            _activationFunctionOutputLayer = new Sigmoid();
        }

        /// <summary>
        /// <para>0 - Mode;default=0:0 - Regression (one output unit): 1 - Classification (multiple output neurons)</para>
        /// <para>1 - Number of hidden layers; default=1</para>
        /// <para>2 - Threshold; default=.01</para>
        /// <para>3 - Maximum Epoch/Iterations; default=1000</para>
        /// <para>4 - Alpha (Initial Learning Rate);default=.3</para>
        /// </summary>
        /// <param name="values"></param>
        public override void
            SetParameters(params double[] values)
        {
            int idx = 0;
            if (values.Length > idx)
                if (values[idx] != double.NaN)
                    _mode = (ModelBackPropagationBase.EnumMode)(int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _model.SetNumberOfLayers((int)(values[idx]) + 1);
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _threshold = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _noOfEpoch = (int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _alpha = (double)values[idx];
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
        /// 
        /// This is as close to the algorithm in Mitchell's book as possible
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
            
            //In Classification number of units same as number of categories
            _noOfUnitsOutputLayer = _model.TargetValues.Length;

            //Add Input Layer
            _model.SetInputLayer(_trainingData, _noOfAttributes);
            
            if (VerifyLayers(_model))
                throw new NeuralNetworkConfigurationNotReady();

            //Initialize weights and setup Upstream Layer for Hidden Layers
            //Dont parallelize as InitWeigts is already Parallel
            for(int idxLayer=1;idxLayer< noOfHiddenLayers + 1;idxLayer++)            
            {
                _model.GetLayer(idxLayer).SetUpstreamLayer(_model.GetLayer(idxLayer - 1));
                _model.GetLayer(idxLayer).InitializeWeights();
            }

            //Add Output Layer
            _model.AddLayer(noOfHiddenLayers + 1, _noOfUnitsOutputLayer, //Output Layer
            _weightBaseValue, _activationFunctionOutputLayer,
                _maxParallelThreads);


            //Sanity Check - Verify if all Upstream layers are in place - do it after adding output layer            
            if (VerifyUpstreamLayers(_model))
                throw new NeuralNetworkConfigurationNotReady();

            double learningRate = 0;

            //Start the training
            //Do not parallelize as computation has to be sequential
            //double[] computedValuesOutput = new double[_model.GetOutputUnitCount()];
            double[][] computedValuesAllLayers = new double[_model.GetNumberOfLayers()][];
            double[][] errorLayers = new double[_model.GetNumberOfLayers()][];

            //Initialize Layer specific data
            //for(int idxLayer=0;idxLayer< errorLayers.Length;idxLayer++)
            Parallel.For(0, errorLayers.Length,
                            new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                            idxLayer => 
                            {
                                errorLayers[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)]; //Input layer is not included in original index
                                computedValuesAllLayers[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)];
                            });


            int idxOutputLayer = _model.GetNumberOfLayers() - 1; 
            //Start iterations
            for (int epoch = 0; epoch < _noOfEpoch; epoch++) //Do not parallelize
            {
                double totalThreshold = 0;
                learningRate = _alpha;// * (1.0 - (epoch /_noOfEpoch));/R also keeps learning rate same                
                for (long row = 0; row < _noOfDataSamples; row++) //For each data sample
                {
                    totalThreshold += ComputeError(computedValuesAllLayers, errorLayers, idxOutputLayer, row);
                    //Fill input layer values in computedValuesAllLayers
                    for (int idxUnit = 0; idxUnit < _model.GetNumberOfUnits(0); idxUnit++)
                    {
                        computedValuesAllLayers[0][idxUnit] = //CAUTION: Upstream value is dependent on weights
                                _model.GetUnitValue(0, idxUnit, row);
                    }

                    UpdateWeights(computedValuesAllLayers,errorLayers, learningRate,row);
                } //For row                    
                if (totalThreshold < _threshold)
                    break;//Stopping condition reached
            }//epoch

            return _model;
        }

        
        protected void UpdateWeights( double[][] computedValuesAllLayers,
                                    double[][] errorLayers,
                                    double learningRate,
                                    long row)
        {
            //Update network weight for each layer.
            //for (int idxLayer = 1; idxLayer < _model.GetNumberOfLayers(); idxLayer++)
            Parallel.For(1,_model.GetNumberOfLayers(),
                    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idxLayer =>
            {
                //Compute values of upstream units, this will preent repetitive computations                        
                long noOfUpstreamUnits = _model.GetNumberOfUpstreamUnits(idxLayer);
                double[] upStreamUnitValues = computedValuesAllLayers[idxLayer-1];// new double[noOfUpstreamUnits];

                //NOTE: Use old values of weights, do not recompte values here
                //as they may use the updated weights

                //For each unit in current layer
                for (int idxUnit = 0;
                                idxUnit < _model.GetNumberOfUnits(idxLayer);
                                    idxUnit++)
                {
                    //Do for each weight in the unit
                    //Bias will be computed in same value
                    double delta = 0;
                    //wji =  wji + dwji
                    for (int idxUpUnit = 0; idxUpUnit < noOfUpstreamUnits;
                                    idxUpUnit++)
                    {
                        delta = learningRate *
                                        errorLayers[idxLayer][idxUnit] *
                                        upStreamUnitValues[idxUpUnit];
                        //Update the weights
                        _model.SetWeight(idxLayer, idxUpUnit, idxUnit,
                                    _model.GetWeight(idxLayer, idxUpUnit, idxUnit)
                                    + delta);
                    }

                    //Update for bias term
                    delta = learningRate *
                                       errorLayers[idxLayer][idxUnit] * 1.0;

                    _model.SetWeight(idxLayer, noOfUpstreamUnits, idxUnit,
                                _model.GetWeight(idxLayer, noOfUpstreamUnits, idxUnit)
                                + delta);
                }

                //Print values

            }); //idxLayer                    
             
        }

        /// <summary>
        /// Follow tk - ok as given in Mitchell's book
        /// </summary>
        /// <param name="computedValuesAllLayers"></param>
        /// <param name="expectedOutput"></param>
        /// <returns></returns>
        protected override double GetOutputUnitDifference(double computedValuesAllLayers, double expectedOutput)
        {
            //returns tk- ok
            return expectedOutput- computedValuesAllLayers;
        }
        
    }
}
