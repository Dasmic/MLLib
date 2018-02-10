using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    /// <summary>
    /// Single layer Perceptron
    /// </summary>
    public class ModelSingleUnitPerceptron : ModelPerceptronBase
    {
        public ModelSingleUnitPerceptron(double missingValue,
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
        /// 
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public override
           double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            double[][] data2D = Convert1Dto2D(data);

            ChangeInputLayerData(data2D);
            return GetOutput(0, 0);

        }        
    }
}
