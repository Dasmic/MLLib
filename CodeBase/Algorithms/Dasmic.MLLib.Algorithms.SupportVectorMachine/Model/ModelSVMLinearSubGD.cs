using Dasmic.MLLib.Common.Exceptions;
using System.Threading.Tasks;
using System.Threading;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    /// <summary>
    /// Solve SVM using Sub-Gradient Descent
    /// 
    /// Can only be used with Linear Kernel
    /// </summary>
    public class ModelSVMLinearSubGD:ModelBase
    {
        private double[] _coeffs; //Need to be array for parallelization

        public ModelSVMLinearSubGD(double missingValue,
                                int indexTargetAttribute, int countAttributes) :
                                base(missingValue, indexTargetAttribute, countAttributes)
        {
            _coeffs = new double[countAttributes + 1]; //account for 1 extra bias attribute
        }

        /// <summary>
        /// Resets all Coefficients to 0 value
        /// </summary>
        public void resetCoeffs()
        {
            Parallel.For(0, _coeffs.Length, idx =>
            {
                _coeffs[idx] = 0;
            });
        }

        public void setCoeff(int index, double value)
        {
            _coeffs[index] = value;
        }

        public double getCoeff(int index)
        {
            return _coeffs[index];
        }

        public int getCoeffCount()
        {
            return _coeffs.Length;
        }

        public override
            double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            double val = 0;
            int col;
            for (col = 0; col < getCoeffCount() - 1; col++)
                val += getCoeff(col) * data[col];

            //Bias
            val += getCoeff(col) * 1;
            return val > 0 ? 1 : -1; ; //Change this when SVM is not binary
        }


        /// <summary>
        /// Returns the current accuracy of model/coeffs
        /// based on passed training data
        /// </summary>
        /// <param name="trainingData"></param>
        /// <returns></returns>
        public double getAccuracy(double[][] trainingData)
        {
            //Number of coeffs = number of Attributes
            if (_coeffs.Length != trainingData.Length)
                throw new AttributesCountMismatchException();
            int correctCount = 0;
            //Run in Parallel
            //for (int row=0;row<trainingData[0].Length;row++)
            Parallel.For(0, trainingData[0].Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, row =>
            {
                double val = 0;
                int result;
                int col;
                for (col = 0; col < trainingData.Length - 1;
                                    col++)
                {
                    val += _coeffs[col] * trainingData[col][row];
                }
                val += _coeffs[col] * 1; //bias term
                result = val > 0 ? 1 : -1; //For binary classifications only
                if (result ==
                    trainingData[_origTargetAttributeIndex][row])
                    Interlocked.Increment(ref correctCount);
            }); //Change to parallel

            return (double)correctCount / (double)trainingData[0].Length;
        }

        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }

    }
}
