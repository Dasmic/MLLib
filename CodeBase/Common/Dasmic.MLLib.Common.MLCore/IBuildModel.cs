

namespace Dasmic.MLLib.Common.MLCore
{
    public interface IBuildModel
    {
        ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute);
        
        void SetMissingValue(double value);
        
        //double runModelForSingleData(double[] data,
        //                             int targetAttributeIndex);

        void SetParameters(params double[] values);
    }
}
