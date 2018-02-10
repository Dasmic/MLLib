using System;
using System.Collections.Generic;

namespace Dasmic.MLLib.Algorithms.DeepLearning
{
    public class BuildConvNetBase: MLLib.Common.MLCore.BuildBase
    {
        public override Common.MLCore.ModelBase
          BuildModel(double[][] trainingData,
                       string[] attributeHeaders,
                       int indexTargetAttribute)
        {
            throw new NotImplementedException();
        }        


        public override Common.MLCore.ModelBase BuildModelSingle(double[][][] trainingData,
                   Dictionary<double, string> targetValueMapping) //Mapping between target values and their string values
        {
            throw new NotImplementedException();
        }
       
        public override void InitializeModel(string[] targetHeaders)
        {
            throw new NotImplementedException();
        }


    }
}
