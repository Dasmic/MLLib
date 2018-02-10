using Dasmic.MLLib.Common.MLCore;

namespace UnitTests.MLLib.Common.MLCore
{
    class BuildBaseObject:BuildBase
    {
        #region IBuildModel Methods
        public override ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            return null;
        }

 

        #endregion
    }
}
