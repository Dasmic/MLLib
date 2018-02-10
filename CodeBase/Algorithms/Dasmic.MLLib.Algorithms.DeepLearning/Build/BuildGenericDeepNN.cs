using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;

namespace Dasmic.MLLib.Algorithms.DeepLearning
{
    /// <summary>
    /// Builds a Generic NN using Rprop with capability of unlimited Hidden Layers
    ///     
    /// The number of units in the Hidden layer can be specifed or auto-computed
    /// The number of hidden layers can be specified through the SetParameters Functions
    /// </summary>
    public class BuildGenericDeepNN : BuildBackPropagationBase
    {        
        int _noOfHiddenLayers;
        int _scalingFactor;
        BuildGenericBackPropagationRprop _buildGenericRprop;
        
        /// <summary>
        /// Sets number units in all Hidden Layers. 
        /// Default is -1, which imples it will be computed 
        /// automatically
        /// </summary>
        public int NumberOfUnitsHiddenLayer { get; set; }

        public BuildGenericDeepNN()
        {
            _buildGenericRprop =
                new BuildGenericBackPropagationRprop();
            _buildGenericRprop.SetParameters(0, 2);
            NumberOfUnitsHiddenLayer = -1;
            _scalingFactor = 2;
            _noOfHiddenLayers = 2;  //1 Output, 2 Hidden
        }

        /// <summary>
        /// Set model building Parameters:
        ///    
        /// <para>0 - Mode;default=0:0 - Regression (one output unit): 1 - Classification (multiple output units)</para>
        /// <para>1 - Number of hidden layers; default=2</para>
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
            _noOfHiddenLayers = (int)values[1];
            _buildGenericRprop.SetParameters(values);
        }


        /// <summary>
        /// Builds the model, with all hidden units having SigMoid activation function        
        /// while output units have Linear function 
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
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);
           

            double[] targetValues =
                GetNumberOfTargetValues(_mode, trainingData, indexTargetAttribute);

            _noOfUnitsOutputLayer = targetValues.Length;

            if (NumberOfUnitsHiddenLayer < 0)
            { //Compute this number automatically
                NumberOfUnitsHiddenLayer =
                   GetNumberOfHiddenUnits(_noOfUnitsOutputLayer, 
                                            _scalingFactor,_noOfAttributes);
            }
            
            //Add Hidden Layers
            for (int idx=0;idx<_noOfHiddenLayers;idx++)
                _buildGenericRprop.AddHiddenLayer(idx, NumberOfUnitsHiddenLayer, 
                                new Sigmoid());

            _buildGenericRprop.SetOutputLayerActivationFunction(new Linear());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)_buildGenericRprop.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            return model;
        }
    }
}
