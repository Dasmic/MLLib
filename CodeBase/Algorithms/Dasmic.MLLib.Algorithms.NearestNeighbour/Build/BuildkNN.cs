using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{
    public class BuildkNN: Common.MLCore.BuildBase
    {
        public override Common.MLCore.ModelBase 
            BuildModel(double[][] trainingData,
                     string[] attributeHeaders,
                     int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);           

            ModelkNN model = new ModelkNN(_missingValue,
                                                indexTargetAttribute,
                                                trainingData.Length - 1,
                                                trainingData);
            return model;
        }


    }
}
