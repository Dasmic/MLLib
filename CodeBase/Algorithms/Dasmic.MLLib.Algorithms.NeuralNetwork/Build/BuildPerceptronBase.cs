 

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    public abstract class BuildPerceptronBase : BuildBase
    {
        public abstract override Common.MLCore.ModelBase
            BuildModel(double[][] trainingData,
                         string[] attributeHeaders,
                         int indexTargetAttribute);

        /// <summary>
        /// 0 - Alpha (Initial Learning Rate); default=.3
        /// 1 - Maximum Epoch/Iterations;default=100
        /// 2 - Initial value of weights, default .05
        /// </summary>
        /// <param name="values"></param>
        public override void
            SetParameters(params double[] values)
        {
            if (values.Length > 0)
                if (values[0] != double.NaN)
                    _alpha = (int)values[0];
            if (values.Length > 1)
                if (values[1] != double.NaN)
                    _noOfEpoch = (int)values[1];
            if (values.Length > 2)
                if (values[2] != double.NaN)
                    _weightBaseValue = (int)values[2];
        }
    }
}
