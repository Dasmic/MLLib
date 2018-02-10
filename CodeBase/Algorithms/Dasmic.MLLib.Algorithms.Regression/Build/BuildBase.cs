using System; 
using Dasmic.MLLib.Common.MLCore;
using System.Threading.Tasks;


namespace Dasmic.MLLib.Algorithms.Regression
{
    public abstract class BuildBase : Common.MLCore.BuildBase
    {
        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute);

    }
}
