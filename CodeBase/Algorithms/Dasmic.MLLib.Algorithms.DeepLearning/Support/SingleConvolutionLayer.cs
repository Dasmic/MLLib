using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SingleConvolutionLayer: SingleConvolutionLayerInput
    {
        private IActivationFunction _activationFunction;
      
        public SingleConvolutionLayer(LayerBase upStreamLayer, 
                                        int noOfFilterUnits,
                                        int sizeWindow, 
                                        int sizeStride)
        {
            _upstreamLayer = upStreamLayer;
            _activationFunction = new RectifiedLinearUnit();
            WeightBaseValue = .05;

            SetupFilterUnits(noOfFilterUnits, 
                                sizeWindow, //This also specifies the Filter dimensions
                                sizeStride);
        }

        /// <summary>
        /// Default activation function is ReLU.Use this function to
        /// overwrite the default value
        /// </summary>
        /// <param name="activationFunction"></param>
        public void SetActivationFunction(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;

            //Update activation in each filter
            foreach(SingleFilterUnit sfn in FilterUnits)
            {
                sfn.SetActivationFunction(_activationFunction);
            }
        }
        
        //TODO:Set Weight Base function
        private void SetupFilterUnits(int noOfFilterUnits,
                                        int sizeWindow, 
                                        int sizeStride)
        {
            FilterUnits = new SingleFilterUnit[noOfFilterUnits];

            for(int ii=0;ii<FilterUnits.Length;ii++)
            {
                FilterUnits[ii] = new SingleFilterUnit(
                                        _upstreamLayer,                                        
                                        sizeWindow, 
                                        sizeStride, 
                                        _maxParallelThreads, 
                                        WeightBaseValue);                                        
            }                                    
        }

        public void ComputeValueOfFilters()
        {
            //For each filter neuron in Layer
            foreach (SingleFilterUnit sfn in FilterUnits)
            {
                //Each FilterUnit has fix output
                //Each filter neuron is connected to another Filter Nruron in Upstream Layer
                sfn.ComputeValueMap();
            }
        }

    }
}
