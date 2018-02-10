using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support
{
    public class SingleLayer
    {
        protected SingleLayer _upstreamLayer; 
        protected IActivationFunction _activationFunction;
        protected int _maxParallelThreads;
        protected long _numberOfUnits;
        protected double _weightBaseValue;

        //Weights are for each unit in current layer x unit in upstream layer
        public double[][] Weights;  //column is for each upstream weight and bias, while row is for every unit in current layer

        /// <summary>
        /// Overload constructor when used in some derived classes
        /// like SinglyFullyConnectedLayer
        /// </summary>
        public SingleLayer()
        {

        }

        public SingleLayer(long numberOfUnits, 
                           double weightBaseValue,
                           SingleLayer upstreamLayer,
                            IActivationFunction activationFunction,
                            int maxParallelThreads)
        {
            _numberOfUnits = numberOfUnits;
            _weightBaseValue = weightBaseValue;
            _upstreamLayer = upstreamLayer;
            _activationFunction = activationFunction;
            _maxParallelThreads = maxParallelThreads;
            //Number of weights corresponds to  number of units upstream
            if (upstreamLayer != null)
            {                 
                InitializeWeights();
            }
        }


        /// <summary>
        /// Set Upstream Layer is not set in the constructor
        /// </summary>
        /// <param name="upstreamLayer"></param>
        public void SetUpstreamLayer(SingleLayer upstreamLayer)
        {
            _upstreamLayer = upstreamLayer;
        }

        /// <summary>
        /// Initialize weight for the layer
        /// </summary>
        /// <param name="initValue"></param>
        public void InitializeWeights()
        {
            if(Weights == null) //Will not set if Weight is already assigned
                Weights = new double[GetNumberOfUpstreamUnits() + 1][]; //One extra for bias

          //for(int col=0;col< Weights.Length;col++)

          Parallel.For(0, Weights.Length,
                    new ParallelOptions { MaxDegreeOfParallelism = 
                        _maxParallelThreads }, col =>
          {
              Weights[col] = new double[GetNumberOfUnits()];
              Random rnd = new Random();
              int mul=0;
              for (int row = 0; row < Weights[col].Length; 
                                row++)
              {
                  //Use different init values
                  mul = rnd.Next(1, 10);
                  Weights[col][row] =  _weightBaseValue * 
                                            mul; //.005
              }
          });          
        }

        /// <summary>
        /// Returns the computed value of each unit in the layer
        /// 
        /// For each unit, take input from upstream layer, multiply by weights
        /// and then apply transfer function
        /// </summary>
        /// <returns></returns>
        public virtual double GetValue(long idxUnit,long row)
        {
            double net = 0;
            for (int idxUp = 0; idxUp < Weights.Length-1; idxUp++)
            {
                net += Weights[idxUp][idxUnit]
                    * GetValueUpstreamLayer(idxUp,row);
            }
            
            //For Bias term
            net += Weights[Weights.Length - 1][idxUnit] * 1.0;
            return _activationFunction.GetValue(net);
        }

        public virtual double GetValueUpstreamLayer(long upIdx, long row)
        {
            return _upstreamLayer.GetValue(upIdx,row);
        }

        /// <summary>
        /// Gets the derivation value from the Activation Function
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double GetDerivativeValue(double value)
        {
            return _activationFunction.GetDerivativeValue(value);
        }

        /// <summary>
        /// Returns number of units
        /// 
        /// NOTE: Does not incluse bias term
        /// </summary>
        /// <returns></returns>
        public long GetNumberOfUnits()
        {
            return _numberOfUnits;
        }


        /// <summary>
        /// Does not include bias term
        /// </summary>
        /// <returns></returns>
        public long GetNumberOfUpstreamUnits()
        {
            return _upstreamLayer.GetNumberOfUnits();
        }


    }
}
