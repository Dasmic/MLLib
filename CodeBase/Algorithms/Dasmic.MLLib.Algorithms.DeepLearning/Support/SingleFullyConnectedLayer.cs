using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support;


namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    /// <summary>
    /// Intuitively the SingleLayer class used in Neutral Network
    /// can be used here. However, the upsteams layer for  the SingleLayer has
    /// linear arrays for weight which will not work correctly here
    /// </summary>
    public class SingleFullyConnectedLayer:LayerBase
    {
        private LayerBase _upStreamLayer; //Cannot use _upStreamLayer in base class
        private double[][][][] _weights; //_weights[idxUnit][idxUpunit][idxCol][idxRow]
        private int _numberOfUnits;

        int _noUpStreamUnits, _noUpStreamVMRows, _noUpStreamVMCols;
        //Upstream layer value map rows and cols        
        
        /// <summary>
        /// Upstream layer will generally be pooling layer
        /// 
        /// </summary>
        /// <param name="upStreamLayer"></param>
        /// <param name="uniqueTargetValues"></param>
        public SingleFullyConnectedLayer(LayerBase upStreamLayer,
                                            int uniqueTargetValues,
                                              int maxParallelThreads=-1)
        { 
            _upStreamLayer = upStreamLayer;
            _numberOfUnits = uniqueTargetValues;
            _maxParallelThreads = maxParallelThreads;

            int _noUpStreamUnits = _upstreamLayer.GetNumberOfUpstreamUnits();
            //Upstream layer value map rows and cols 
            int _noUpStreamVMRows = _upstreamLayer.GetValueMapNoOfRows();
            int _noUpStreamVMCols = _upstreamLayer.GetValueMapNoOfColumns();

            //Call after all privat2 vars have been assigned values
            InitializeWeights();
        }

        /// <summary>
        /// _weights[idxUnit][idxUpunit][idxCol][idxRow]
        /// </summary>
        public void InitializeWeights()
        {
            if (_weights == null) //Will not set if Weight is already assigned
            {
                _weights = new double[_numberOfUnits][][][];                            
            }
          

            //for(int col=0;col< Weights.Length;col++)
            Parallel.For(0, _weights.Length,
                      new ParallelOptions
                      {
                          MaxDegreeOfParallelism =
                          _maxParallelThreads
                      }, unit =>
                      { //For each unit
                          _weights[unit] = new double[_noUpStreamUnits][][];
                          Random rnd = new Random();
                          int mul = 0;
                          for (int idxUpUnit = 0; idxUpUnit < _noUpStreamUnits; idxUpUnit++)
                          {
                              _weights[unit][idxUpUnit] = new double[_noUpStreamVMCols][];
                              for (int idxCol = 0; idxCol < _noUpStreamVMCols; idxCol++)
                              {
                                  _weights[unit][idxUpUnit][idxCol] = new double[_noUpStreamVMCols];
                                  for (int idxRow = 0; idxRow < _noUpStreamVMRows; idxRow++)
                                  { // row
                                    //Use different init values
                                      
                                      mul = rnd.Next(1, 10);
                                      _weights[unit][idxUpUnit][idxCol][idxRow] = 
                                                                WeightBaseValue * mul; //.005
                                  }//row
                              } //col
                          } //idxUpUnit
                      });
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="idxUnit"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public double GetValue(long idxUnit)
        {
            Parallel.For(0, _noUpStreamUnits,
                      new ParallelOptions
                      {
                          MaxDegreeOfParallelism =
                          _maxParallelThreads
                      }, idxUpUnit =>
                      { //For each unit                          
                          {
                              for (int idxCol = 0; idxCol < _noUpStreamVMCols; idxCol++)
                              {
                                  for (int idxRow = 0; idxRow < _noUpStreamVMRows; idxRow++)
                                  { // row
                                      //Use different init values
                                      
                                      _weights[unit][idxUpUnit][idxCol][idxRow] =
                                                                WeightBaseValue * mul; //.005
                                  }//row
                              } //col
                          } //idxUpUnit
                      });

            
        }
    }
}
