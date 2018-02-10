using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    public abstract class BuildBase: MLLib.Common.MLCore.BuildBase
    {
        protected long _maxIterations;

        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute);
    }
}
