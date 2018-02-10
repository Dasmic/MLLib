using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support
{
    public class SingleLayerInput:SingleLayer
    {
        private double[][] _data;
        //private long _currentRow;

        public SingleLayerInput(int numberOfUnits,
                                int maxParallelThreads):
                           base(numberOfUnits,1, null,
                           null,maxParallelThreads)
        {
            
        }


        public void SetData(double[][] data)
        {
            _data = data;
            //Last column of data will include target value
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="unitIdx">idx of unit, each column/feature in training data is an inputr neuron</param>
        /// <param name="trainingDataRow">Training data row</param>
        /// <returns></returns>
        public override double GetValue(long unitIdx,long trainingDataRow)
        {
            if (unitIdx >= _data.Length) //Do not give back target column
                throw new IndexOutOfRangeException();
            return _data[unitIdx][trainingDataRow];
        }
        
    }
}
