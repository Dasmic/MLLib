using System.Threading.Tasks;
using System.Linq;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    /// <summary>
    /// Single layer Perceptron
    /// </summary>
    public class ModelMultiUnitPerceptron :ModelPerceptronBase
    {
        public double[] TargetValues; //Directly initialized from build system

        public ModelMultiUnitPerceptron(double missingValue,
                          int indexTargetAttribute, 
                          int countAttributes) :
                                base(missingValue, indexTargetAttribute, countAttributes)
        {
            
        }
              
        /// <summary>
        /// Runs the model for passed data
        /// 
        /// Perceptron will return 1 if y > 0
        /// else will return -1
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public override 
           double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            double[][] data2D = Convert1Dto2D(data);
            ChangeInputLayerData(data2D);
            
            //Get max value
            int idx = GetIndexOfMaxOutput(0);
            return TargetValues[idx];
        }

        /// <summary>
        /// Returns the idx of the higest value
        /// </summary>         
        /// <param name="row"></param>        
        /// <returns></returns>
        public int GetIndexOfMaxOutput(long row)
        {
            double[] computedValue = GetOutput(row);
            
            //Find max Idx in computedValue
            int maxIdx = computedValue.Select((item, indx) =>
                new { Item = item, Index = indx }).
                OrderByDescending(x => x.Item).Select(x => x.Index).First();

            return maxIdx;            
        }
    }
}
