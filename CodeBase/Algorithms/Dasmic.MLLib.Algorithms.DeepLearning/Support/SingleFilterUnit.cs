using System;
using System.Threading.Tasks;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;


namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    /// <summary>
    /// In CNN each Filter is like a Unit in a single layer
    /// 
    /// In the standard NN we did not implement a class for each Unit
    /// However, to better separate the complexity in CNNs later's we have created
    /// a separate class for each Unit (also called as Filter in CNN) here.
    /// </summary>
    public class SingleFilterUnit:UnitBase
    {
        private double[][] _weights;
        public double _bias; //Keep bias separate
        private double _weightBaseValue;
        
        IActivationFunction _activationFunction;

        /// <summary>
        /// Overloaded constructor. Mainly meant to be called from Input layer
        /// </summary>
        /// <param name="noOfInputColumns"></param>
        /// <param name="noOfInputRows"></param>
        /// <param name="windowSize"></param>
        /// <param name="strideSize"></param>
        /// <param name="maxParallelThreads"></param>
        /// <param name="weightBaseValue"></param>
        public SingleFilterUnit(int noOfInputColumns,
                                int noOfInputRows,
                                  int windowSize,  //FilterSize is always square
                                  int strideSize,
                                  int maxParallelThreads,
                                  double weightBaseValue) : base(null, strideSize, maxParallelThreads)
        {
            SetValueMap(noOfInputColumns, noOfInputRows, windowSize);
        }

        public SingleFilterUnit(LayerBase upStreamLayer,
                                  int windowSize,  //FilterSize is always square
                                  int strideSize,
                                  int maxParallelThreads,
                                  double weightBaseValue) :base(upStreamLayer,strideSize,maxParallelThreads)
        {            
            //The weight matrix should equal FilterSize             
            _weights = SupportFunctions.Get2DArray(windowSize, 
                                            windowSize);
            _weightBaseValue = weightBaseValue;
            SetValueMap(windowSize);
        }
      
        public void SetActivationFunction(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
        }

        public double GetValueMapAtIndex(int idxCol, int idxRow)
        {
            return ValueMap[idxCol][idxRow];
        }

        /// <summary>
        /// Initialize weights for in filterUnit
        /// </summary>
        /// <param name="initValue"></param>
        public void InitializeWeights()
        {         
            //Weights array should already be allocated before coming here
            Parallel.For(0, _weights.Length,
                      new ParallelOptions
                      {
                          MaxDegreeOfParallelism =
                          _maxParallelThreads
                      }, col =>
                      {                          
                          Random rnd = new Random();
                          int mul = 0;
                          for (int row = 0; row < _weights[col].Length;
                                          row++)
                          {
                            //Use different init values
                            mul = rnd.Next(1, 10);
                              _weights[col][row] = _weightBaseValue *
                                                      mul; //.005
                        }
                      });
        }

        /// <summary>
        /// Computes the FilterMap by convolution over each Input array
        /// from Upstream layer
        /// 
        /// This function contains the sliding convolution window logic
        /// </summary>
        /// <param name="upStreamLayer"></param>
        public override void ComputeValueMap()
        {           
            int filterSize = _weights.Length; //Filter is same as window
            //Slide the filter Window to compute value for each Cell in ValueMap (VM)                        
            for (int idxRowVM = 0; idxRowVM < GetValueMapNoOfRows(); idxRowVM++)
            {//idxRowVM
                for (int idxColVM = 0; idxColVM < GetValueMapNoOfColumns(); idxColVM++)
                { //idxColVM
                    double sum = 0;
                    int idxUpValueMapCol=0, idxUpValueMapRow=0;
                    //Set the filter window co-ordinates in the upstream value map
                    //relative to the current Value Map
                    int upMapLeftCol = idxColVM * _stride;
                    int upMapLeftRow = idxRowVM * _stride;
                    //TODO: Assign value
                    for(int idxFilter=0; idxFilter < _upStreamLayer.GetNumberOfFilterUnits();
                                idxFilter++) //For each Filter Unit Upstream
                    {
                        SingleFilterUnit upFilterUnit =
                                            (SingleFilterUnit)_upStreamLayer.GetFilterUnit(idxFilter);

                        for (int idxFilterRow = 0; idxFilterRow < filterSize; idxFilterRow++)
                        { 
                            for (int idxFilterCol = 0; idxFilterCol < filterSize; idxFilterCol++)                            
                            {
                                //Update value                               
                                idxUpValueMapCol = upMapLeftCol + idxFilterCol;
                                idxUpValueMapRow = upMapLeftRow + idxFilterRow;
                                //Multiply upStream FilterMap by current Filter weights
                                sum+= _weights[idxFilterCol][idxFilterRow] *
                                            upFilterUnit.GetValueMapAtIndex(idxUpValueMapCol, 
                                                                    idxUpValueMapRow);
                            } //idxFilterCol
                              //Do for bias term
                            sum += _bias * 1.0;
                        } //idxFilterRow
                    }//idxFilter

                    //Normalize Sum
                    sum = sum / (filterSize * 2); //divide sum by filter Grid size
                    //Apply activation function - this saves time
                    sum = _activationFunction.GetValue(sum);
                    ValueMap[idxColVM][idxRowVM] = sum; //Assign value to FilterMap
                } //idxColFM
            } //idxRowFM  
        }


        public double GetFilterWeight(int idxCol, int idxRow)
        {
            return _weights[idxCol][idxRow];
        }


    }


}
