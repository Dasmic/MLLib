

namespace Dasmic.MLLib.Algorithms.Bayesian
{
    public abstract class BuildBase : Common.MLCore.BuildBase
    {
        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                     string[] attributeHeaders,
                     int indexTargetAttribute);

    }
}
